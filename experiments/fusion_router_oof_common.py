"""
fusion_router_oof_common.py — Primary ML router fusion via nested stratified CV + OOF stacking.

Protocol (see ``thesis_overview.md`` §12C and §12A.6):
  * Outer loop: ``K`` sentence-level multilabel-stratified folds (default 5).
  * Inner loop on outer-train: ``K-1`` folds produce OOF regular + cascade predictions for router fit.
  * Retrain both base models on full outer-train; evaluate fused NER on outer-test (held out).

Base tracks: ``non_crf`` (Exp01 + Exp04 -> ``06_*_oof``) and ``crf`` (Exp10 regular + Exp10
cascade -> ``10_*_oof``). With ``THESIS_OOF_AUGMENT=1`` (``exp07+aug`` conditions) every
inner-train / outer-train set is augmented inside its fold; inner-val and outer-test never are.

``06_*_ready`` / ``10_*_ready`` remain in-sample routing upper bounds (appendix only).
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import get_experiment_output_dir, suppress_output_if_needed, write_result_excel, write_result_json
from exp07_split_artifacts import multilabel_stratified_kfold_assignments
from fusion_ready_sources import (
    FUSION_ROUTER_CATEGORICAL_FEATURES,
    FUSION_ROUTER_NUMERIC_FEATURES,
    compute_metrics,
    load_cascade_from_exp04,
    load_regular_from_exp01,
    merge_regular_cascade,
)
from fusion_router_ready_common import (
    ReadyRouterConfig,
    _build_classifier,
    _build_preprocess_pipeline,
    _exp06_configs,
    _exp10_configs,
)
from split_io import load_split, save_split

try:
    from sklearn.pipeline import Pipeline
except ImportError:
    Pipeline = None  # type: ignore[misc, assignment]

_NUMERIC_FEATURES = list(FUSION_ROUTER_NUMERIC_FEATURES)
_CATEGORICAL_FEATURES = list(FUSION_ROUTER_CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class BaseTrack:
    name: str
    regular_module: str
    cascade_module: str
    regular_cache_file: str
    cascade_cache_file: str
    cache_prefix: str
    lambda_bio_envs: tuple[str, ...]
    lambda_type_envs: tuple[str, ...]


NON_CRF_TRACK = BaseTrack(
    name="non_crf",
    regular_module="experiment_01_regular_ner",
    cascade_module="experiment_04_auc_cascaded_pipeline",
    regular_cache_file="exp01_metrics.xlsx",
    cascade_cache_file="exp04_metrics.xlsx",
    cache_prefix="",
    lambda_bio_envs=("THESIS_EXP04_LAMBDA_BIO",),
    lambda_type_envs=("THESIS_EXP04_LAMBDA_TYPE",),
)
CRF_TRACK = BaseTrack(
    name="crf",
    regular_module="experiment_10_regular_ner_crf",
    cascade_module="experiment_10_cascaded_pipeline_crf",
    regular_cache_file="exp10_regular_metrics.xlsx",
    cascade_cache_file="exp10_cascade_metrics.xlsx",
    cache_prefix="crf_",
    # Same fallback order as core/cascaded_crf_runtime.py when no CRF grid selection exists.
    lambda_bio_envs=("THESIS_EXP10_LAMBDA_BIO", "THESIS_EXP04_LAMBDA_BIO"),
    lambda_type_envs=("THESIS_EXP10_LAMBDA_TYPE", "THESIS_EXP04_LAMBDA_TYPE"),
)
BASE_TRACKS: dict[str, BaseTrack] = {track.name: track for track in (NON_CRF_TRACK, CRF_TRACK)}

_OOF_EXPERIMENT_IDS: dict[str, dict[str, str]] = {
    "non_crf": {
        "linear_svm": "06_svm_oof",
        "kernel_svm": "06_svm_kernel_oof",
        "nb": "06_nb_oof",
        "lr": "06_lr_oof",
        "rf": "06_rf_oof",
        "mlp": "06_mlp_oof",
    },
    "crf": {
        "linear_svm": "10_svm_oof",
        "kernel_svm": "10_svm_kernel_oof",
        "nb": "10_nb_oof",
        "lr": "10_lr_oof",
        "rf": "10_rf_oof",
        "mlp": "10_mlp_oof",
    },
}


@dataclass(frozen=True)
class OofRouterConfig:
    config_key: str
    ready: ReadyRouterConfig
    experiment_id: str
    experiment_name: str
    result_basename: str
    track: BaseTrack


def _oof_configs(track: str = "non_crf") -> dict[str, OofRouterConfig]:
    base_track = BASE_TRACKS[track]
    ready_configs = _exp10_configs() if base_track is CRF_TRACK else _exp06_configs()
    out: dict[str, OofRouterConfig] = {}
    for key, cfg in ready_configs.items():
        out[key] = OofRouterConfig(
            config_key=key,
            ready=cfg,
            experiment_id=_OOF_EXPERIMENT_IDS[track][key],
            experiment_name=cfg.experiment_name.replace("(Ready)", "(OOF CV)"),
            result_basename=cfg.result_basename.replace("_ready", "_oof"),
            track=base_track,
        )
    return out


def _resolve_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or str(default)).strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return default


def _env_float_chain(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return default


def _augmentation_enabled() -> bool:
    return (os.environ.get("THESIS_OOF_AUGMENT") or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_corpus_sentences() -> list[dict]:
    train_path = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
    eval_path = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
    if not train_path or not eval_path:
        raise RuntimeError("OOF router fusion requires THESIS_PRESPLIT_TRAIN_JSON and THESIS_PRESPLIT_EVAL_JSON")
    train = load_split(Path(train_path))
    eval_ = load_split(Path(eval_path))
    return list(train) + list(eval_)


def _oof_cache_base() -> Path:
    output_dir = (os.environ.get("THESIS_CROSS_OUTPUT_DIR") or "").strip()
    base = Path(output_dir) if output_dir else get_experiment_output_dir("exp06_oof")
    return base / "oof_router_cache"


def _condition_and_seed() -> tuple[str, str]:
    cond = (os.environ.get("THESIS_CURRENT_CONDITION_KEY") or "default").replace("/", "_")
    seed = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    return cond, seed


def _model_cache_token() -> str:
    raw = (os.environ.get("THESIS_MODEL_NAME") or "model").strip()
    if os.path.isabs(raw):
        # Local checkpoint paths would otherwise replace the cache root when joined.
        raw = Path(raw).name
    return raw.replace("/", "_").replace("\\", "_")


def _oof_cache_root(track: BaseTrack) -> Path:
    cond, seed = _condition_and_seed()
    root = _oof_cache_base() / f"{track.cache_prefix}{_model_cache_token()}_{cond}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _augmentation_settings() -> tuple[str, int]:
    import experiment_08_llm_augmentation as exp08_mod

    aug_model = (os.environ.get("THESIS_OOF_AUGMENTATION_MODEL") or "").strip()
    if not aug_model:
        aug_model = exp08_mod._resolve_augmentation_model_name((os.environ.get("THESIS_MODEL_NAME") or "").strip())
    try:
        multiplier = max(1, int((os.environ.get("THESIS_EXP08_MULTIPLIER") or "3").strip()))
    except ValueError:
        multiplier = 3
    return aug_model, multiplier


def _augmented_fold_train(
    train_sents: list[dict],
    fold_key: str,
    aug_seed: int,
    aug_model: str,
    multiplier: int,
) -> list[dict]:
    """Fold-train sentences plus mask-fill variants generated from them alone (vocabulary included).

    Cached per (condition, seed, fold) and shared by every encoder and both tracks, because the
    fold partition and the augmentation model do not depend on the encoder being trained.
    """
    cond, seed = _condition_and_seed()
    cache_dir = _oof_cache_base() / f"aug_{cond}_seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "augmentation_meta.json"
    if meta_path.exists():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(stored.get("multiplier", multiplier)) != multiplier:
            raise RuntimeError(
                f"{meta_path} was built with multiplier={stored.get('multiplier')} but "
                f"THESIS_EXP08_MULTIPLIER={multiplier}; use a fresh --output-dir or delete {cache_dir}."
            )
        if stored.get("augmentation_model") != aug_model:
            print(
                f"[oof-aug] WARNING: {cache_dir.name} was built with augmentation model "
                f"{stored.get('augmentation_model')!r}; current {aug_model!r}. Reusing cached fold sets."
            )
    else:
        meta_path.write_text(
            json.dumps(
                {"augmentation_model": aug_model, "multiplier": multiplier, "vocabulary_scope": "fold_train"},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    out_path = cache_dir / f"{fold_key}_train.json"
    if out_path.exists():
        return load_split(out_path)

    import experiment_08_llm_augmentation as exp08_mod

    with suppress_output_if_needed():
        generated, _ = exp08_mod._augment_training_data(
            train_sents,
            exp08_mod.sentences_to_token_df(train_sents),
            (os.environ.get("THESIS_MODEL_NAME") or "").strip(),
            multiplier=multiplier,
            rng_seed=aug_seed,
            augmentation_model_name=aug_model,
        )
    augmented = list(train_sents) + list(generated)
    save_split(augmented, out_path)
    return augmented


def _run_base_pair(train_json: Path, eval_json: Path, cache_key: str, track: BaseTrack) -> tuple[Path, Path]:
    cache_dir = _oof_cache_root(track) / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    regular_xlsx = cache_dir / track.regular_cache_file
    cascade_xlsx = cache_dir / track.cascade_cache_file
    marker = cache_dir / "complete.json"
    if marker.exists() and regular_xlsx.exists() and cascade_xlsx.exists():
        return regular_xlsx, cascade_xlsx

    prev_train = os.environ.get("THESIS_PRESPLIT_TRAIN_JSON")
    prev_eval = os.environ.get("THESIS_PRESPLIT_EVAL_JSON")
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(train_json)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(eval_json)

    try:
        regular_mod = importlib.import_module(track.regular_module)
        cascade_mod = importlib.import_module(track.cascade_module)

        p_reg = regular_mod.run()
        p_cas = cascade_mod.run()
        src_reg = Path(str(p_reg.get("metrics_file") or p_reg.get("result_file") or ""))
        src_cas = Path(str(p_cas.get("metrics_file") or p_cas.get("result_file") or ""))
        if not src_reg.exists() or not src_cas.exists():
            raise RuntimeError(f"Missing {track.name} base model outputs for cache key {cache_key}")
        # Fold workbooks are only read back from this cache; moving avoids a second copy per training.
        shutil.move(str(src_reg), str(regular_xlsx))
        shutil.move(str(src_cas), str(cascade_xlsx))
        marker.write_text(
            json.dumps(
                {"track": track.name, "regular": str(regular_xlsx), "cascade": str(cascade_xlsx)},
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            from core.model_cleanup import cleanup_training_artifacts_if_enabled

            cleanup_training_artifacts_if_enabled()
        except Exception:
            pass
        return regular_xlsx, cascade_xlsx
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


def _merged_predictions(regular_xlsx: Path, cascade_xlsx: Path, local_to_global: list[int]) -> pd.DataFrame:
    # Exp10 workbooks share the Exp01 / Exp04 sheet layouts (see fusion_crf_ready_sources.py).
    reg = _remap_sentence_ids(load_regular_from_exp01(regular_xlsx), local_to_global)
    cas = _remap_sentence_ids(load_cascade_from_exp04(cascade_xlsx), local_to_global)
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


def run_oof_router_fusion(config_key: str, track: str = "non_crf") -> dict:
    config = _oof_configs(track)[config_key]
    base_track = config.track
    sentences = _load_corpus_sentences()
    n_sentences = len(sentences)
    if n_sentences < 10:
        raise RuntimeError(f"OOF router fusion needs at least 10 sentences; got {n_sentences}")

    split_seed = _resolve_int_env("THESIS_SPLIT_SEED", 42)
    outer_folds = _resolve_int_env("THESIS_ROUTER_OOF_OUTER_FOLDS", 5)
    inner_folds = _resolve_int_env("THESIS_ROUTER_OOF_INNER_FOLDS", max(2, outer_folds - 1))
    augment = _augmentation_enabled()
    aug_model, aug_multiplier = _augmentation_settings() if augment else ("", 0)

    outer_assign = multilabel_stratified_kfold_assignments(sentences, outer_folds, split_seed + 1000)
    split_scratch = _oof_cache_root(base_track) / "splits"
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

            cache_key = f"outer{outer_idx}_inner{inner_idx}"
            if augment:
                inner_train = _augmented_fold_train(
                    inner_train,
                    cache_key,
                    split_seed + 3000 + 100 * outer_idx + inner_idx,
                    aug_model,
                    aug_multiplier,
                )
            train_path = split_scratch / f"outer{outer_idx}_inner{inner_idx}_train.json"
            eval_path = split_scratch / f"outer{outer_idx}_inner{inner_idx}_eval.json"
            save_split(inner_train, train_path)
            save_split(inner_val, eval_path)
            reg_xlsx, cas_xlsx = _run_base_pair(train_path, eval_path, cache_key, base_track)
            oof_parts.append(_merged_predictions(reg_xlsx, cas_xlsx, inner_val_global))

        if oof_parts:
            oof_merged = pd.concat(oof_parts, ignore_index=True)
        else:
            oof_merged = pd.DataFrame()

        router, router_info = _train_router_from_merged(oof_merged, config) if not oof_merged.empty else (None, {"trained": False})

        final_cache = f"outer{outer_idx}_final"
        outer_train_sents = train_sents
        if augment:
            outer_train_sents = _augmented_fold_train(
                train_sents,
                final_cache,
                split_seed + 3000 + 100 * outer_idx + 99,
                aug_model,
                aug_multiplier,
            )
        outer_train_path = split_scratch / f"outer{outer_idx}_final_train.json"
        outer_test_path = split_scratch / f"outer{outer_idx}_final_test.json"
        save_split(outer_train_sents, outer_train_path)
        save_split(test_sents, outer_test_path)
        reg_xlsx, cas_xlsx = _run_base_pair(outer_train_path, outer_test_path, final_cache, base_track)
        test_merged = _merged_predictions(reg_xlsx, cas_xlsx, test_global)
        fused_test = _apply_router(test_merged, router, config)
        pooled_test_parts.append(fused_test)

        original_texts = {str(s.get("text", "")) for s in train_sents}
        novel_synthetic_texts = {
            str(s.get("text", "")) for s in outer_train_sents[len(train_sents):]
        } - original_texts
        test_texts = {str(s.get("text", "")) for s in test_sents}

        f1_fold, p_fold, r_fold = compute_metrics(fused_test)
        fold_summaries.append(
            {
                "outer_fold": outer_idx,
                "train_sentences": len(train_local_idx),
                "train_sentences_augmented": len(outer_train_sents),
                # Expected 0; non-zero only if the corpus holds near-duplicate sentences.
                "synthetic_test_text_overlap": len(novel_synthetic_texts & test_texts),
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

    model_display = (os.environ.get("THESIS_MODEL_NAME") or "model").replace("\\", "/").split("/")[-1]
    split_condition = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
    split_seed_str = os.environ.get("THESIS_SPLIT_SEED", "42")

    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "oof_router_pooled_outer_tests",
                "model": model_display,
                "split_condition": split_condition,
                "seed": split_seed_str,
                "experiment_id": config.experiment_id,
                "track": base_track.name,
                "augmentation": "in_fold_mask_fill" if augment else "none",
                "augmentation_model": aug_model,
                "augmentation_multiplier": aug_multiplier if augment else None,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "corpus_sentences": n_sentences,
                "outer_folds": outer_folds,
                "inner_folds": inner_folds,
                "router": config.ready.classifier_label,
                "protocol": "nested_stratified_cv_oof",
                "lambda_bio": _env_float_chain(base_track.lambda_bio_envs, 10.0),
                "lambda_type": _env_float_chain(base_track.lambda_type_envs, 5.0),
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
        "track": base_track.name,
        "augmentation": "in_fold_mask_fill" if augment else "none",
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


def exp10_oof_run_factory(config_key: str) -> Callable[[], dict]:
    def run() -> dict:
        return run_oof_router_fusion(config_key, track="crf")

    return run


OOF_ML_ROUTER_EXP_IDS: frozenset[str] = frozenset(
    exp_id for ids in _OOF_EXPERIMENT_IDS.values() for exp_id in ids.values()
)
