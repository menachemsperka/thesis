"""
experiment_06_fusion_svm.py - Fusion with SVM-based disagreement routing

Fusion strategy:
- If regular and cascaded predictions agree, keep the agreed label.
- If they disagree, use an SVM router trained on train-split disagreements to
  choose which source (regular or cascade) is more likely correct.

Training protocol:
- Requires pre-split train/eval JSON files.
- Train regular and cascaded models on train split.
- Build disagreement rows on train split and label router targets by which
  source matches the ground-truth label.
- Fit a linear SVM with numeric + categorical features.

Evaluation protocol:
- Apply agreement passthrough.
- Route disagreements through the trained SVM.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from common import (
    configure_model_environment,
    get_experiment_output_dir,
    is_debug_enabled,
    now_timestamp,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from NERtraining import PrepDataSetNERTraining  # type: ignore


def _safe_label_name(label_idx: int, label_list: list[str]) -> str | None:
    if label_idx == -100:
        return None
    if 0 <= int(label_idx) < len(label_list):
        return label_list[int(label_idx)]
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits)
    exp_vals = np.exp(stable)
    return exp_vals / np.sum(exp_vals)


def _top2_margin(probs: np.ndarray) -> float:
    if probs.size == 0:
        return 0.0
    ordered = np.sort(probs)
    if ordered.size == 1:
        return float(ordered[-1])
    return float(ordered[-1] - ordered[-2])


def _build_regular_token_predictions(eval_ds, trainer, tokenizer, label_list: list[str]) -> pd.DataFrame:
    preds, _, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds, axis=2)

    rows = []
    for sentence_idx, item in enumerate(eval_ds, start=1):
        input_ids = item["input_ids"]
        true_ids = item["labels"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_id = 0
        for token, true_id, pred_id, token_logits in zip(tokens, true_ids, pred_ids[sentence_idx - 1], preds[sentence_idx - 1]):
            true_label = _safe_label_name(int(true_id), label_list)
            pred_label = _safe_label_name(int(pred_id), label_list)
            if int(true_id) == -100 or true_label is None or pred_label is None or str(token).startswith("["):
                continue

            token_id += 1
            probs = _softmax(np.asarray(token_logits, dtype=np.float64))
            rows.append(
                {
                    "sentence_id": sentence_idx,
                    "token_idx": token_id,
                    "token": str(token),
                    "true_label": true_label,
                    "regular_pred_label": pred_label,
                    "regular_prob": float(np.max(probs)),
                    "regular_margin": _top2_margin(probs),
                }
            )

    return pd.DataFrame(rows)


def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _load_cascaded_predicted_tokens(excel_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(excel_path, sheet_name=None)
    detailed = sheets.get("detailed_results")
    if detailed is None or detailed.empty:
        raise ValueError(f"Missing or empty detailed_results sheet in: {excel_path}")

    if "eval_mode" in detailed.columns:
        detailed = detailed[detailed["eval_mode"].astype(str) == "predicted"].copy()

    required_columns = {
        "sentence_id", "token_idx", "token",
        "true_bio", "true_etype", "pred_bio", "pred_etype",
        "entity_prob", "bio_prob",
    }
    missing = required_columns - set(detailed.columns)
    if missing:
        raise ValueError(f"Missing columns in cascaded results: {missing}")

    rows = []
    for _, row in detailed.iterrows():
        true_label = _bio_type_to_label(row.get("true_bio"), row.get("true_etype"))
        pred_label = _bio_type_to_label(row.get("pred_bio"), row.get("pred_etype"))
        entity_prob = float(row.get("entity_prob", 0))
        bio_prob = float(row.get("bio_prob", 0))
        combined_prob = entity_prob * bio_prob

        rows.append(
            {
                "sentence_id": int(row["sentence_id"]),
                "token_idx": int(row["token_idx"]),
                "token": str(row["token"]),
                "cascade_true_label": true_label,
                "cascade_pred_label": pred_label,
                "cascade_prob": combined_prob,
                # Proxy margin in [0,1], larger means farther from uncertainty 0.5.
                "cascade_margin": abs(combined_prob - 0.5) * 2.0,
            }
        )

    return pd.DataFrame(rows)


def _to_seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str) -> tuple[list[list[str]], list[list[str]]]:
    true_lists = []
    pred_lists = []
    for sentence_id in df["sentence_id"].unique():
        sentence_df = df[df["sentence_id"] == sentence_id]
        true_lists.append(sentence_df[true_col].astype(str).tolist())
        pred_lists.append(sentence_df[pred_col].astype(str).tolist())
    return true_lists, pred_lists


def _run_cascaded_pipeline_subprocess(extra_env: dict[str, str] | None = None):
    run_kwargs = {}
    if is_debug_enabled():
        run_kwargs = {
            "capture_output": False,
            "text": True,
        }
    else:
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }

    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)

    return subprocess.run([sys.executable, str(CORE_DIR / "auc_cascaded_pipeline.py")], env=env, **run_kwargs)


def _raise_if_subprocess_failed(completed: subprocess.CompletedProcess, context: str) -> None:
    if completed.returncode == 0:
        return

    stderr_tail = ""
    stdout_tail = ""
    if isinstance(completed.stderr, str):
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-40:])
    if isinstance(completed.stdout, str):
        stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-40:])
    details = stderr_tail or stdout_tail or "No subprocess output captured."
    raise RuntimeError(
        f"Cascaded pipeline failed during {context}. "
        f"Exit code: {completed.returncode}. "
        f"Last output lines:\n{details}"
    )


def _archive_cascaded_results(exp_dir: Path, suffix: str) -> Path:
    source_metrics_path = CORE_DIR / "cascaded_pipeline_results.xlsx"
    archived_source_metrics = exp_dir / f"fusion_svm_source_cascaded_{suffix}_{now_timestamp()}.xlsx"
    if source_metrics_path.exists():
        shutil.move(str(source_metrics_path), str(archived_source_metrics))
        return archived_source_metrics
    raise FileNotFoundError("cascaded_pipeline_results.xlsx was not generated by cascaded pipeline")


def _split_label_parts(label: str) -> tuple[str, str]:
    if not label or label == "O":
        return "O", "NONE"
    if "-" not in label:
        return label, "NONE"
    bio, etype = label.split("-", 1)
    return bio, etype


def _build_router_frame(merged: pd.DataFrame) -> pd.DataFrame:
    disag = merged[merged["regular_pred_label"].astype(str) != merged["cascade_pred_label"].astype(str)].copy()
    if disag.empty:
        return disag

    disag["regular_correct"] = disag["regular_pred_label"].astype(str) == disag["true_label"].astype(str)
    disag["cascade_correct"] = disag["cascade_pred_label"].astype(str) == disag["true_label"].astype(str)

    def _target_source(row) -> str | None:
        reg_ok = bool(row["regular_correct"])
        cas_ok = bool(row["cascade_correct"])
        if reg_ok and not cas_ok:
            return "regular"
        if cas_ok and not reg_ok:
            return "cascade"
        return None

    disag["target_source"] = disag.apply(_target_source, axis=1)

    reg_parts = disag["regular_pred_label"].astype(str).map(_split_label_parts)
    cas_parts = disag["cascade_pred_label"].astype(str).map(_split_label_parts)

    disag["regular_bio"] = reg_parts.map(lambda t: t[0])
    disag["regular_etype"] = reg_parts.map(lambda t: t[1])
    disag["cascade_bio"] = cas_parts.map(lambda t: t[0])
    disag["cascade_etype"] = cas_parts.map(lambda t: t[1])

    disag["prob_diff"] = disag["regular_prob"] - disag["cascade_prob"]
    disag["abs_prob_diff"] = (disag["regular_prob"] - disag["cascade_prob"]).abs()
    disag["max_prob"] = disag[["regular_prob", "cascade_prob"]].max(axis=1)

    return disag


def _train_router(train_router_df: pd.DataFrame):
    usable = train_router_df[train_router_df["target_source"].notna()].copy()
    if usable.empty:
        return None, {
            "trained": False,
            "reason": "no_train_disagreements_with_single_correct_source",
            "train_disagreements_total": int(len(train_router_df)),
            "train_disagreements_usable": 0,
        }

    classes = sorted(usable["target_source"].unique().tolist())
    if len(classes) < 2:
        return None, {
            "trained": False,
            "reason": "single_class_router_targets",
            "train_disagreements_total": int(len(train_router_df)),
            "train_disagreements_usable": int(len(usable)),
            "class_counts": usable["target_source"].value_counts().to_dict(),
        }

    numeric_features = [
        "regular_prob",
        "cascade_prob",
        "regular_margin",
        "cascade_margin",
        "prob_diff",
        "abs_prob_diff",
        "max_prob",
    ]
    categorical_features = [
        "regular_bio",
        "regular_etype",
        "cascade_bio",
        "cascade_etype",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", preprocess),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)),
        ]
    )

    x_train = usable[numeric_features + categorical_features]
    y_train = usable["target_source"].astype(str)

    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    train_acc = float((train_pred == y_train).mean()) if len(y_train) else None

    info = {
        "trained": True,
        "router": "LinearSVC",
        "train_disagreements_total": int(len(train_router_df)),
        "train_disagreements_usable": int(len(usable)),
        "class_counts": usable["target_source"].value_counts().to_dict(),
        "train_accuracy": train_acc,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return model, info


def _row_router_features(row: pd.Series) -> pd.DataFrame:
    reg_bio, reg_etype = _split_label_parts(str(row["regular_pred_label"]))
    cas_bio, cas_etype = _split_label_parts(str(row["cascade_pred_label"]))
    reg_prob = float(row["regular_prob"])
    cas_prob = float(row["cascade_prob"])
    return pd.DataFrame(
        [
            {
                "regular_prob": reg_prob,
                "cascade_prob": cas_prob,
                "regular_margin": float(row.get("regular_margin", 0.0)),
                "cascade_margin": float(row.get("cascade_margin", 0.0)),
                "prob_diff": reg_prob - cas_prob,
                "abs_prob_diff": abs(reg_prob - cas_prob),
                "max_prob": max(reg_prob, cas_prob),
                "regular_bio": reg_bio,
                "regular_etype": reg_etype,
                "cascade_bio": cas_bio,
                "cascade_etype": cas_etype,
            }
        ]
    )


def _compute_agreement_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    agree = int((df["regular_pred_label"] == df["cascade_pred_label"]).sum())
    disagree = total - agree
    return {
        "total_tokens": total,
        "agreement_count": agree,
        "disagreement_count": disagree,
        "agreement_percent": 100 * agree / total if total > 0 else 0,
    }


def run() -> dict:
    dataset_path = resolve_dataset("ner_dataset.csv")
    model_name, is_local_model = configure_model_environment()

    seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 42

    with suppress_output_if_needed():
        worker = PrepDataSetNERTraining()
        data = worker.load_and_prepare_data(str(dataset_path))

        presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
        presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
        has_presplit = bool(
            presplit_train
            and presplit_eval
            and Path(presplit_train).exists()
            and Path(presplit_eval).exists()
        )

        if not has_presplit:
            raise RuntimeError(
                "SVM fusion requires pre-split train/eval JSON files. "
                "Set THESIS_PRESPLIT_TRAIN_JSON and THESIS_PRESPLIT_EVAL_JSON to existing files."
            )

        from split_io import load_split

        train_sentences = load_split(Path(presplit_train))
        eval_sentences = load_split(Path(presplit_eval))

        trainer, eval_results, label_list, ds_eval = worker.run_training_with_presplit(
            data,
            train_sentences,
            eval_sentences,
        )

        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        regular_eval_tokens_df = _build_regular_token_predictions(ds_eval, trainer, processor, label_list)

        train_ds = getattr(trainer, "train_dataset", None)
        if train_ds is None:
            raise RuntimeError("Trainer does not expose train_dataset; cannot train SVM disagreement router.")
        regular_train_tokens_df = _build_regular_token_predictions(train_ds, trainer, processor, label_list)

    exp_dir = get_experiment_output_dir("exp06_fusion_svm")

    completed_eval = _run_cascaded_pipeline_subprocess()
    _raise_if_subprocess_failed(completed_eval, "eval inference")
    archived_eval_metrics = _archive_cascaded_results(exp_dir, "eval")
    cascaded_eval_tokens_df = _load_cascaded_predicted_tokens(archived_eval_metrics)

    completed_train = _run_cascaded_pipeline_subprocess(
        {
            "THESIS_PRESPLIT_TRAIN_JSON": presplit_train,
            "THESIS_PRESPLIT_EVAL_JSON": presplit_train,
        }
    )
    _raise_if_subprocess_failed(completed_train, "training-split router learning")
    archived_train_metrics = _archive_cascaded_results(exp_dir, "train_router")
    cascaded_train_tokens_df = _load_cascaded_predicted_tokens(archived_train_metrics)

    merged_train = regular_train_tokens_df.merge(
        cascaded_train_tokens_df,
        on=["sentence_id", "token_idx"],
        how="inner",
        suffixes=("_regular", "_cascade"),
    )
    if merged_train.empty:
        raise RuntimeError("Fusion failed: no aligned train tokens between regular and cascaded outputs.")

    merged_eval = regular_eval_tokens_df.merge(
        cascaded_eval_tokens_df,
        on=["sentence_id", "token_idx"],
        how="inner",
        suffixes=("_regular", "_cascade"),
    )
    if merged_eval.empty:
        raise RuntimeError("Fusion failed: no aligned eval tokens between regular and cascaded outputs.")

    merged_eval["true_label"] = merged_eval["true_label"].astype(str)
    merged_eval["cascade_true_label"] = merged_eval["cascade_true_label"].astype(str)
    mismatched_truth = int((merged_eval["true_label"] != merged_eval["cascade_true_label"]).sum())

    train_router_df = _build_router_frame(merged_train)
    router, router_info = _train_router(train_router_df)

    def _svm_fusion(row) -> tuple[str, str, float]:
        regular_label = str(row["regular_pred_label"])
        cascade_label = str(row["cascade_pred_label"])
        regular_prob = float(row["regular_prob"])
        cascade_prob = float(row["cascade_prob"])

        if regular_label == cascade_label:
            return regular_label, "agree", max(regular_prob, cascade_prob)

        if router is None:
            if regular_prob >= cascade_prob:
                return regular_label, "fallback_regular", regular_prob
            return cascade_label, "fallback_cascade", cascade_prob

        x_row = _row_router_features(row)
        choice = str(router.predict(x_row)[0])
        if choice == "regular":
            return regular_label, "svm_regular", regular_prob
        return cascade_label, "svm_cascade", cascade_prob

    selected = merged_eval.apply(_svm_fusion, axis=1, result_type="expand")
    selected.columns = ["fused_pred_label", "selected_source", "selected_confidence"]
    merged_eval = pd.concat([merged_eval, selected], axis=1)

    merged_eval["disagree"] = merged_eval["regular_pred_label"].astype(str) != merged_eval["cascade_pred_label"].astype(str)

    y_true, y_pred = _to_seqeval_lists(merged_eval, "true_label", "fused_pred_label")
    precision = float(precision_score(y_true, y_pred)) if y_true else None
    recall = float(recall_score(y_true, y_pred)) if y_true else None
    f1 = float(f1_score(y_true, y_pred)) if y_true else None

    agreement_stats = _compute_agreement_stats(merged_eval)

    disagreement_count = int(merged_eval["disagree"].sum())
    selected_regular = int((merged_eval["selected_source"].isin(["svm_regular", "fallback_regular"])) .sum())
    selected_cascade = int((merged_eval["selected_source"].isin(["svm_cascade", "fallback_cascade"])) .sum())
    selected_agree = int((merged_eval["selected_source"] == "agree").sum())

    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tokens_aligned": int(len(merged_eval)),
                "disagreements": disagreement_count,
                "selected_regular": selected_regular,
                "selected_cascade": selected_cascade,
                "agreements": selected_agree,
                "truth_label_mismatch_between_models": mismatched_truth,
                "router_trained": bool(router_info.get("trained")),
                "router_train_disagreements_total": int(router_info.get("train_disagreements_total", 0)),
                "router_train_disagreements_usable": int(router_info.get("train_disagreements_usable", 0)),
            }
        ]
    )

    detailed_df = merged_eval[
        [
            "sentence_id",
            "token_idx",
            "token_regular",
            "true_label",
            "regular_pred_label",
            "regular_prob",
            "regular_margin",
            "cascade_pred_label",
            "cascade_prob",
            "cascade_margin",
            "disagree",
            "selected_source",
            "selected_confidence",
            "fused_pred_label",
        ]
    ].rename(columns={"token_regular": "token"})

    agreement_df = pd.DataFrame([agreement_stats])
    router_info_df = pd.DataFrame([router_info])

    metrics_file = write_result_excel(
        "exp06_fusion_svm",
        "fusion_svm_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "agreement_stats": agreement_df,
            "router_info": router_info_df,
            "regular_tokens_eval": regular_eval_tokens_df,
            "cascaded_tokens_eval": cascaded_eval_tokens_df,
            "regular_tokens_train": regular_train_tokens_df,
            "cascaded_tokens_train": cascaded_train_tokens_df,
            "router_train_rows": train_router_df,
        },
    )

    result_data = {
        "experiment_id": "exp06_fusion_svm",
        "experiment_name": "SVM Router Fusion",
        "description": "Agreement passthrough with SVM disagreement routing between regular and cascaded predictions.",
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "metrics_file": str(metrics_file),
        "agreement_stats": agreement_stats,
        "fusion_method": "svm_router",
        "router_info": router_info,
    }

    out_path = write_result_json("exp06_fusion_svm", "fusion_svm", result_data)
    result_data["result_file"] = str(out_path)
    return result_data
