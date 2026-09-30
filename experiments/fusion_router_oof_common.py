"""
fusion_router_oof_common.py — Primary ML router fusion via nested stratified CV + OOF stacking.

Protocol (see ``theisis overview.md`` §12C):
  * Outer loop: ``K`` sentence-level multilabel-stratified folds (default 5).
  * Inner loop on outer-train: ``K-1`` folds produce OOF Exp01 + Exp04 predictions for router fit.
  * Retrain Exp01 + Exp04 on full outer-train; evaluate fused NER on outer-test (held out).

``06_*_ready`` remains an in-sample routing upper bound (appendix only).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import get_experiment_output_dir, write_result_excel, write_result_json
from exp07_split_artifacts import multilabel_stratified_kfold_assignments
from fusion_ready_sources import (
    FUSION_ROUTER_CATEGORICAL_FEATURES,
    FUSION_ROUTER_NUMERIC_FEATURES,
    compute_metrics,
    load_cascade_from_exp04,
    load_regular_from_exp01,
    merge_regular_cascade,
)
from fusion_router_ready_common import ReadyRouterConfig, _build_classifier, _build_preprocess_pipeline, _exp06_configs
from split_io import load_split, save_split

try:
    from sklearn.pipeline import Pipeline
except ImportError:
    Pipeline = None  # type: ignore[misc, assignment]

_NUMERIC_FEATURES = list(FUSION_ROUTER_NUMERIC_FEATURES)
_CATEGORICAL_FEATURES = list(FUSION_ROUTER_CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class OofRouterConfig:
    config_key: str
    ready: ReadyRouterConfig
    experiment_id: str
    experiment_name: str
    result_basename: str


def _oof_configs() -> dict[str, OofRouterConfig]:
    mapping = {
        "linear_svm": "06_svm_oof",
        "kernel_svm": "06_svm_kernel_oof",
        "nb": "06_nb_oof",
        "lr": "06_lr_oof",
        "rf": "06_rf_oof",
        "mlp": "06_mlp_oof",
    }
    out: dict[str, OofRouterConfig] = {}
    for key, cfg in _exp06_configs().items():
        exp_id = mapping[key]
        out[key] = OofRouterConfig(
            config_key=key,
            ready=cfg,
            experiment_id=exp_id,
            experiment_name=cfg.experiment_name.replace("(Ready)", "(OOF CV)"),
            result_basename=cfg.result_basename.replace("_ready", "_oof"),
        )
    return out


def _resolve_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or str(default)).strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return default


def _load_corpus_sentences() -> list[dict]:
    train_path = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
    eval_path = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
    if not train_path or not eval_path:
        raise RuntimeError("OOF router fusion requires THESIS_PRESPLIT_TRAIN_JSON and THESIS_PRESPLIT_EVAL_JSON")
    train = load_split(Path(train_path))
    eval_ = load_split(Path(eval_path))
    return list(train) + list(eval_)


def _oof_cache_root() -> Path:
    output_dir = (os.environ.get("THESIS_CROSS_OUTPUT_DIR") or "").strip()
    base = Path(output_dir) if output_dir else get_experiment_output_dir("exp06_oof")
    model = (os.environ.get("THESIS_MODEL_NAME") or "model").replace("/", "_")
    cond = (os.environ.get("THESIS_CURRENT_CONDITION_KEY") or "default").replace("/", "_")
    seed = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    root = base / "oof_router_cache" / f"{model}_{cond}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_base_pair(train_json: Path, eval_json: Path, cache_key: str) -> tuple[Path, Path]:
    cache_dir = _oof_cache_root() / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    exp01_xlsx = cache_dir / "exp01_metrics.xlsx"
    exp04_xlsx = cache_dir / "exp04_metrics.xlsx"
    marker = cache_dir / "complete.json"
    if marker.exists() and exp01_xlsx.exists() and exp04_xlsx.exists():
        return exp01_xlsx, exp04_xlsx

    prev_train = os.environ.get("THESIS_PRESPLIT_TRAIN_JSON")
    prev_eval = os.environ.get("THESIS_PRESPLIT_EVAL_JSON")
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(train_json)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(eval_json)

    try:
        import experiment_01_regular_ner as exp01_mod
        import experiment_04_auc_cascaded_pipeline as exp04_mod

        p1 = exp01_mod.run()
        p4 = exp04_mod.run()
        src1 = Path(str(p1.get("metrics_file") or p1.get("result_file") or ""))
        src4 = Path(str(p4.get("metrics_file") or p4.get("result_file") or ""))
        if not src1.exists() or not src4.exists():
            raise RuntimeError(f"Missing base model outputs for cache key {cache_key}")
        shutil.copy2(src1, exp01_xlsx)
        shutil.copy2(src4, exp04_xlsx)
        marker.write_text(
            json.dumps({"exp01": str(exp01_xlsx), "exp04": str(exp04_xlsx)}, indent=2),
            encoding="utf-8",
        )
        try:
            from core.model_cleanup import cleanup_training_artifacts_if_enabled

            cleanup_training_artifacts_if_enabled()
        except Exception:
            pass
        return exp01_xlsx, exp04_xlsx
    finally:
        if prev_train is not None:
            os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = prev_train
        else:
            os.environ.pop("THESIS_PRESPLIT_TRAIN_JSON", None)
        if prev_eval is not None:
            os.environ["THESIS_PRESPLIT_EVAL_JSON"] = prev_eval
        else:
            os.environ.pop("THESIS_PRESPLIT_EVAL_JSON", None)


def _remap_sentence_ids(df: pd.DataFrame, local_to_global: list[int]) -> pd.DataFrame:
    out = df.copy()
    mapping = {i + 1: local_to_global[i] for i in range(len(local_to_global))}
    out["sentence_id"] = out["sentence_id"].astype(int).map(mapping)
    return out


def _merged_predictions(exp01_xlsx: Path, exp04_xlsx: Path, local_to_global: list[int]) -> pd.DataFrame:
    reg = _remap_sentence_ids(load_regular_from_exp01(exp01_xlsx), local_to_global)
    cas = _remap_sentence_ids(load_cascade_from_exp04(exp04_xlsx), local_to_global)
    return merge_regular_cascade(reg, cas)


def _train_router_from_merged(merged: pd.DataFrame, config: OofRouterConfig):
    disag = merged[merged["disagree"]].copy()
    disag["regular_correct"] = disag["regular_pred_label"].astype(str) == disag["true_label"].astype(str)
    disag["cascade_correct"] = disag["cascade_pred_label"].astype(str) == disag["true_label"].astype(str)

    def _target(row):
        r = bool(row["regular_correct"])
        c = bool(row["cascade_correct"])
        if r and not c:
            return "regular"
        if c and not r:
            return "cascade"
        return None

    disag["target_source"] = disag.apply(_target, axis=1)
    usable = disag[disag["target_source"].notna()].copy()
    if usable.empty or usable["target_source"].nunique() < 2 or Pipeline is None:
        return None, {"trained": False, "reason": "insufficient_router_training_data"}

    model = Pipeline(
        steps=[
            ("pre", _build_preprocess_pipeline(config.config_key)),
            ("clf", _build_classifier(config.config_key)),
        ]
    )
    x_train = usable[_NUMERIC_FEATURES + _CATEGORICAL_FEATURES]
    y_train = usable["target_source"].astype(str)
    model.fit(x_train, y_train)
    return model, {
        "trained": True,
        "router": config.ready.classifier_label,
        "training_samples": int(len(usable)),
        "class_distribution": y_train.value_counts().to_dict(),
    }


def _apply_router(merged: pd.DataFrame, router, config: OofRouterConfig) -> pd.DataFrame:
    out = merged.copy()
    reg_label = out["regular_pred_label"].values
    cas_label = out["cascade_pred_label"].values
    reg_prob = out["regular_prob"].values
    cas_prob = out["cascade_prob"].values
    agree = ~out["disagree"].values

    fused = reg_label.copy()
    source = np.full(len(out), "agree", dtype=object)
    confidence = np.maximum(reg_prob, cas_prob)
    disag_idx = np.where(~agree)[0]

    ready_cfg = config.ready
    if router is not None and len(disag_idx) > 0:
        x_disag = out.iloc[disag_idx][_NUMERIC_FEATURES + _CATEGORICAL_FEATURES]
        predictions = router.predict(x_disag)
        for i, idx in enumerate(disag_idx):
            choice = str(predictions[i])
            if choice == "regular":
                fused[idx] = reg_label[idx]
                source[idx] = ready_cfg.source_tag_regular
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = ready_cfg.source_tag_cascade
                confidence[idx] = cas_prob[idx]
    else:
        for idx in disag_idx:
            if reg_prob[idx] >= cas_prob[idx]:
                fused[idx] = reg_label[idx]
                source[idx] = ready_cfg.source_tag_fallback_regular
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = ready_cfg.source_tag_fallback_cascade
                confidence[idx] = cas_prob[idx]

    out["fused_pred_label"] = fused
    out["selected_source"] = source
    out["selected_confidence"] = confidence
    return out


def run_oof_router_fusion(config_key: str) -> dict:
    config = _oof_configs()[config_key]
    sentences = _load_corpus_sentences()
    n_sentences = len(sentences)
    if n_sentences < 10:
        raise RuntimeError(f"OOF router fusion needs at least 10 sentences; got {n_sentences}")

    split_seed = _resolve_int_env("THESIS_SPLIT_SEED", 42)
    outer_folds = _resolve_int_env("THESIS_ROUTER_OOF_OUTER_FOLDS", 5)
    inner_folds = _resolve_int_env("THESIS_ROUTER_OOF_INNER_FOLDS", max(2, outer_folds - 1))

    outer_assign = multilabel_stratified_kfold_assignments(sentences, outer_folds, split_seed + 1000)
    split_scratch = _oof_cache_root() / "splits"
    split_scratch.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict] = []
    pooled_test_parts: list[pd.DataFrame] = []

    for outer_idx in range(outer_folds):
        test_local_idx = [i for i, a in enumerate(outer_assign) if a == outer_idx]
        train_local_idx = [i for i, a in enumerate(outer_assign) if a != outer_idx]
        if not test_local_idx or not train_local_idx:
            continue

        train_sents = [sentences[i] for i in train_local_idx]
        test_sents = [sentences[i] for i in test_local_idx]
        train_global = [i + 1 for i in train_local_idx]
        test_global = [i + 1 for i in test_local_idx]

        inner_assign = multilabel_stratified_kfold_assignments(
            train_sents, inner_folds, split_seed + 2000 + outer_idx
        )
        oof_parts: list[pd.DataFrame] = []
        for inner_idx in range(inner_folds):
            inner_train = [train_sents[j] for j, a in enumerate(inner_assign) if a != inner_idx]
            inner_val = [train_sents[j] for j, a in enumerate(inner_assign) if a == inner_idx]
            inner_val_global = [train_global[j] for j, a in enumerate(inner_assign) if a == inner_idx]
            if not inner_val:
                continue

            train_path = split_scratch / f"outer{outer_idx}_inner{inner_idx}_train.json"
            eval_path = split_scratch / f"outer{outer_idx}_inner{inner_idx}_eval.json"
            save_split(inner_train, train_path)
            save_split(inner_val, eval_path)
            cache_key = f"outer{outer_idx}_inner{inner_idx}"
            e1, e4 = _run_base_pair(train_path, eval_path, cache_key)
            oof_parts.append(_merged_predictions(e1, e4, inner_val_global))

        if oof_parts:
            oof_merged = pd.concat(oof_parts, ignore_index=True)
        else:
            oof_merged = pd.DataFrame()

        router, router_info = _train_router_from_merged(oof_merged, config) if not oof_merged.empty else (None, {"trained": False})

        outer_train_path = split_scratch / f"outer{outer_idx}_final_train.json"
        outer_test_path = split_scratch / f"outer{outer_idx}_final_test.json"
        save_split(train_sents, outer_train_path)
        save_split(test_sents, outer_test_path)
        final_cache = f"outer{outer_idx}_final"
        fe1, fe4 = _run_base_pair(outer_train_path, outer_test_path, final_cache)
        test_merged = _merged_predictions(fe1, fe4, test_global)
        fused_test = _apply_router(test_merged, router, config)
        pooled_test_parts.append(fused_test)

        f1_fold, p_fold, r_fold = compute_metrics(fused_test)
        fold_summaries.append(
            {
                "outer_fold": outer_idx,
                "train_sentences": len(train_local_idx),
                "test_sentences": len(test_local_idx),
                "router_info": router_info,
                "f1": f1_fold,
                "precision": p_fold,
                "recall": r_fold,
            }
        )

    if not pooled_test_parts:
        raise RuntimeError("OOF router fusion produced no outer-test predictions")

    pooled = pd.concat(pooled_test_parts, ignore_index=True)
    f1, precision, recall = compute_metrics(pooled)

    model_display = (os.environ.get("THESIS_MODEL_NAME") or "model").split("/")[-1]
    split_condition = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
    split_seed_str = os.environ.get("THESIS_SPLIT_SEED", "42")

    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "oof_router_pooled_outer_tests",
                "model": model_display,
                "split_condition": split_condition,
                "seed": split_seed_str,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "corpus_sentences": n_sentences,
                "outer_folds": outer_folds,
                "inner_folds": inner_folds,
                "router": config.ready.classifier_label,
                "protocol": "nested_stratified_cv_oof",
            }
        ]
    )
    fold_df = pd.DataFrame(fold_summaries)

    out_xlsx = write_result_excel(
        config.experiment_id,
        config.result_basename,
        metrics_df,
        pooled,
        extra_sheets={"fold_metrics": fold_df},
    )
    payload = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "metrics_file": str(out_xlsx),
        "result_file": str(out_xlsx),
        "experiment_id": config.experiment_id,
        "protocol": "nested_stratified_cv_oof",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "corpus_sentences": n_sentences,
    }
    write_result_json(config.experiment_id, config.result_basename, payload)
    return payload


def exp06_oof_run_factory(config_key: str) -> Callable[[], dict]:
    def run() -> dict:
        return run_oof_router_fusion(config_key)

    return run


OOF_ML_ROUTER_EXP_IDS: frozenset[str] = frozenset(
    {
        "06_svm_oof",
        "06_svm_kernel_oof",
        "06_nb_oof",
        "06_lr_oof",
        "06_rf_oof",
        "06_mlp_oof",
    }
)
