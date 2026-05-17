"""
experiment_06_fusion_ensemble_rules.py — Rules-based ensemble voting

Better fusion strategy: Apply decision rules for different scenarios.
1. Predictions agree: use prediction (confidence is strong)
2. Predictions disagree with clear winner (conf_gap > threshold): use higher confidence
3. Predictions disagree with unclear winner: use cascade (more conservative, better at boundaries)

This addresses: Simple confidence voting doesn't account for disagreement severity.
Cascade is typically more conservative/robust at entity boundaries.
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

from common import (
    configure_model_environment,
    get_experiment_output_dir,
    is_debug_enabled,
    now_timestamp,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
    write_split_runs_excel,
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

    required_columns = {"sentence_id", "token_idx", "token", "true_bio", "true_etype", "pred_bio", "pred_etype", "entity_prob", "bio_prob"}
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

        rows.append({
            "sentence_id": int(row["sentence_id"]),
            "token_idx": int(row["token_idx"]),
            "token": str(row["token"]),
            "cascade_true_label": true_label,
            "cascade_pred_label": pred_label,
            "cascade_prob": combined_prob,
        })

    return pd.DataFrame(rows)


def _to_seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str) -> tuple[list[list[str]], list[list[str]]]:
    true_lists = []
    pred_lists = []
    for sentence_id in df["sentence_id"].unique():
        sentence_df = df[df["sentence_id"] == sentence_id]
        true_lists.append(sentence_df[true_col].astype(str).tolist())
        pred_lists.append(sentence_df[pred_col].astype(str).tolist())
    return true_lists, pred_lists


def _run_cascaded_pipeline_subprocess():
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

    return subprocess.run([sys.executable, str(CORE_DIR / "auc_cascaded_pipeline.py")], **run_kwargs)


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

        # Support pre-computed splits from experiment 07
        presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
        presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
        if presplit_train and presplit_eval and Path(presplit_train).exists() and Path(presplit_eval).exists():
            from split_io import load_split
            train_sentences = load_split(Path(presplit_train))
            eval_sentences = load_split(Path(presplit_eval))
            trainer, eval_results, label_list, ds_eval = worker.run_training_with_presplit(
                data, train_sentences, eval_sentences,
            )
        else:
            trainer, eval_results, label_list, ds_eval = worker.run_training_steps(data)

        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        regular_tokens_df = _build_regular_token_predictions(ds_eval, trainer, processor, label_list)

    completed = _run_cascaded_pipeline_subprocess()
    if completed.returncode != 0:
        stderr_tail = ""
        stdout_tail = ""
        if isinstance(completed.stderr, str):
            stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-40:])
        if isinstance(completed.stdout, str):
            stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-40:])
        details = stderr_tail or stdout_tail or "No subprocess output captured."
        raise RuntimeError(
            "Cascaded pipeline failed. "
            f"Exit code: {completed.returncode}. "
            f"Last output lines:\n{details}"
        )

    source_metrics_path = CORE_DIR / "cascaded_pipeline_results.xlsx"
    exp_dir = get_experiment_output_dir("exp06_fusion_ensemble_rules")
    archived_source_metrics = exp_dir / f"fusion_ensemble_rules_source_cascaded_{now_timestamp()}.xlsx"
    if source_metrics_path.exists():
        shutil.move(str(source_metrics_path), str(archived_source_metrics))
    else:
        raise FileNotFoundError("cascaded_pipeline_results.xlsx was not generated by cascaded pipeline")

    cascaded_tokens_df = _load_cascaded_predicted_tokens(archived_source_metrics)

    merged = regular_tokens_df.merge(
        cascaded_tokens_df,
        on=["sentence_id", "token_idx"],
        how="inner",
        suffixes=("_regular", "_cascade"),
    )
    if merged.empty:
        raise RuntimeError("Fusion failed: no aligned tokens between regular and cascaded outputs.")

    merged["true_label"] = merged["true_label"].astype(str)
    merged["cascade_true_label"] = merged["cascade_true_label"].astype(str)

    mismatched_truth = int((merged["true_label"] != merged["cascade_true_label"]).sum())

    merged["disagree"] = merged["regular_pred_label"].astype(str) != merged["cascade_pred_label"].astype(str)

    # ============================================================================
    # ENSEMBLE RULES FUSION: Decision rules based on agreement and confidence gap
    # ============================================================================
    AGREEMENT_THRESHOLD = 0.2  # Confidence gap threshold

    def _ensemble_rules_fusion(row) -> tuple[str, str, float]:
        """
        Rules-based ensemble voting:
        
        1. Predictions agree: use with high confidence (agreement is strong signal)
        2. Predictions disagree:
           a. Large gap (|conf_diff| > threshold): trust higher confidence (clear winner)
           b. Small gap (|conf_diff| <= threshold): use cascade (conservative at boundaries)
        """
        regular_label = str(row["regular_pred_label"])
        cascade_label = str(row["cascade_pred_label"])
        regular_prob = float(row["regular_prob"])
        cascade_prob = float(row["cascade_prob"])

        pred_agree = (regular_label == cascade_label)
        conf_diff = abs(regular_prob - cascade_prob)

        if pred_agree:
            # Rule 1: Predictions agree → use with average confidence
            return regular_label, "agree", max(regular_prob, cascade_prob)

        # Rule 2a: Large confidence gap → trust higher confidence
        if conf_diff > AGREEMENT_THRESHOLD:
            if regular_prob > cascade_prob:
                return regular_label, "rules_regular_confident", regular_prob
            else:
                return cascade_label, "rules_cascade_confident", cascade_prob

        # Rule 2b: Small confidence gap → use cascade (more conservative)
        return cascade_label, "rules_cascade_conservative", cascade_prob

    selected = merged.apply(_ensemble_rules_fusion, axis=1, result_type="expand")
    selected.columns = ["fused_pred_label", "selected_source", "selected_confidence"]
    merged = pd.concat([merged, selected], axis=1)

    y_true, y_pred = _to_seqeval_lists(merged, "true_label", "fused_pred_label")
    precision = float(precision_score(y_true, y_pred)) if y_true else None
    recall = float(recall_score(y_true, y_pred)) if y_true else None
    f1 = float(f1_score(y_true, y_pred)) if y_true else None

    agreement_stats = _compute_agreement_stats(merged)

    disagreement_count = int(merged["disagree"].sum())
    selected_regular = int((merged["selected_source"] == "rules_regular_confident").sum())
    selected_cascade_confident = int((merged["selected_source"] == "rules_cascade_confident").sum())
    selected_cascade_conservative = int((merged["selected_source"] == "rules_cascade_conservative").sum())
    selected_cascade = selected_cascade_confident + selected_cascade_conservative
    selected_agree = int((merged["selected_source"] == "agree").sum())

    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tokens_aligned": int(len(merged)),
                "disagreements": disagreement_count,
                "selected_regular": selected_regular,
                "selected_cascade": selected_cascade,
                "selected_cascade_confident": selected_cascade_confident,
                "selected_cascade_conservative": selected_cascade_conservative,
                "agreements": selected_agree,
                "truth_label_mismatch_between_models": mismatched_truth,
            }
        ]
    )

    detailed_df = merged[
        [
            "sentence_id",
            "token_idx",
            "token_regular",
            "true_label",
            "regular_pred_label",
            "regular_prob",
            "cascade_pred_label",
            "cascade_prob",
            "disagree",
            "selected_source",
            "selected_confidence",
            "fused_pred_label",
        ]
    ].rename(columns={"token_regular": "token"})

    agreement_df = pd.DataFrame([agreement_stats])

    metrics_file = write_result_excel(
        "exp06_fusion_ensemble_rules",
        "fusion_ensemble_rules_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "agreement_stats": agreement_df,
            "regular_tokens": regular_tokens_df,
            "cascaded_tokens": cascaded_tokens_df,
        },
    )

    result_data = {
        "experiment_id": "exp06_fusion_ensemble_rules",
        "experiment_name": "Ensemble Rules Fusion",
        "description": "Rules-based ensemble: agree→both, clear_winner→higher_conf, ambiguous→cascade (conservative).",
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "metrics_file": str(metrics_file),
        "agreement_stats": agreement_stats,
        "fusion_method": "ensemble_rules",
        "rules_parameters": {
            "agreement_threshold": AGREEMENT_THRESHOLD,
        },
    }

    out_path = write_result_json("exp06_fusion_ensemble_rules", "fusion_ensemble_rules", result_data)
    result_data["result_file"] = str(out_path)

    return result_data


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
