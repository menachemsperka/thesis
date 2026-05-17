"""
experiment_05_ready.py — AUC Cascaded Pipeline Step-3 Consistency (Ready from Exp04)

Loads pre-computed Exp04 (cascaded pipeline) outputs and applies the B/I
entity-type consistency rule as a post-processing step, without retraining.

Consistency rule: when a B-X tag is immediately followed by I-Y with X != Y,
reconcile by choosing the entity type with the higher bio_prob.

Requires Exp04 to have been run first (outputs/exp04/latest.json must exist,
or set THESIS_READY_EXP04_XLSX).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from common import write_result_excel, write_result_json


def _resolve_exp04(env_var: str = "THESIS_READY_EXP04_XLSX") -> Path:
    explicit = (os.environ.get(env_var) or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points to a missing file: {p}")
        return p

    latest_json = Path("outputs") / "exp04" / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(
            "Cannot auto-resolve Exp04 output.  "
            f"Either set {env_var} or run experiment 04 first so that {latest_json} exists."
        )

    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    metrics_file = payload.get("metrics_file")
    if not metrics_file:
        raise ValueError(f"metrics_file key missing in {latest_json}")

    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(f"metrics_file from {latest_json} not found: {p}")
    return p


def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _apply_bi_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Apply B/I entity-type consistency: when B-X is followed by I-Y with X!=Y,
    reconcile using the higher bio_prob."""
    df = df.copy()
    for sentence_id in sorted(df["sentence_id"].unique()):
        mask = df["sentence_id"] == sentence_id
        sentence_df = df.loc[mask].sort_values("token_idx")
        indices = sentence_df.index.tolist()

        for i in range(len(indices) - 1):
            curr_idx = indices[i]
            next_idx = indices[i + 1]

            curr_bio = str(df.at[curr_idx, "pred_bio"])
            next_bio = str(df.at[next_idx, "pred_bio"])
            curr_etype = str(df.at[curr_idx, "pred_etype"])
            next_etype = str(df.at[next_idx, "pred_etype"])

            if curr_bio == "B" and next_bio == "I" and curr_etype != next_etype:
                curr_prob = float(df.at[curr_idx, "bio_prob"] or 0.0)
                next_prob = float(df.at[next_idx, "bio_prob"] or 0.0)

                if curr_prob >= next_prob:
                    df.at[next_idx, "pred_etype"] = curr_etype
                else:
                    df.at[curr_idx, "pred_etype"] = next_etype

    return df


def _extract_final_f1(xlsx_path: Path) -> float | None:
    """Try to extract the pipeline_span_f1 from the metrics sheet."""
    try:
        mdf = pd.read_excel(xlsx_path, sheet_name="metrics")
        final_rows = mdf[
            (mdf["epoch"].astype(str) == "final_optimised")
            & (mdf["eval_mode"] == "predicted")
        ]
        if final_rows.empty:
            return None
        return float(final_rows.iloc[-1]["pipeline_span_f1"])
    except Exception:
        return None


def run() -> dict:
    exp04_xlsx = _resolve_exp04()

    df = pd.read_excel(exp04_xlsx, sheet_name="detailed_results")

    if "eval_mode" in df.columns:
        df = df[df["eval_mode"].astype(str) == "predicted"].copy()

    required = {"sentence_id", "token_idx", "token", "true_bio", "true_etype",
                "pred_bio", "pred_etype", "entity_prob", "bio_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp04 detailed_results missing columns: {sorted(missing)}")

    df["sentence_id"] = df["sentence_id"].astype(int)
    df["token_idx"] = df["token_idx"].astype(int)

    # Apply consistency
    consistent_df = _apply_bi_consistency(df)

    # Build labels before/after for reporting
    consistent_df["pred_label_after"] = [
        _bio_type_to_label(b, t) for b, t in zip(consistent_df["pred_bio"], consistent_df["pred_etype"])
    ]
    consistent_df["true_label"] = [
        _bio_type_to_label(b, t) for b, t in zip(consistent_df["true_bio"], consistent_df["true_etype"])
    ]

    # Original (before consistency) labels for comparison
    orig_labels = [
        _bio_type_to_label(b, t) for b, t in zip(df["pred_bio"], df["pred_etype"])
    ]
    consistent_df["pred_label_before"] = orig_labels
    consistent_df["consistency_changed"] = (
        consistent_df["pred_label_before"] != consistent_df["pred_label_after"]
    )

    # Compute seqeval metrics
    y_true, y_pred = [], []
    for sid in sorted(consistent_df["sentence_id"].unique()):
        sdf = consistent_df[consistent_df["sentence_id"] == sid].sort_values("token_idx")
        y_true.append(sdf["true_label"].astype(str).tolist())
        y_pred.append(sdf["pred_label_after"].astype(str).tolist())

    f1 = float(f1_score(y_true, y_pred)) if y_true else None
    prec = float(precision_score(y_true, y_pred)) if y_true else None
    rec = float(recall_score(y_true, y_pred)) if y_true else None

    tokens_changed = int(consistent_df["consistency_changed"].sum())

    metrics_df = pd.DataFrame([{
        "dataset_name": "ready_from_exp04",
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "total_tokens": len(consistent_df),
        "tokens_changed_by_consistency": tokens_changed,
        "source_exp04_xlsx": str(exp04_xlsx),
    }])

    detailed_out = consistent_df[[
        "sentence_id", "token_idx", "token",
        "true_bio", "true_etype", "true_label",
        "pred_bio", "pred_etype", "pred_label_after",
        "pred_label_before", "consistency_changed",
        "entity_prob", "bio_prob",
    ]].copy()

    metrics_file = write_result_excel(
        "exp05_ready",
        "cascaded_step3_consistent_ready_results",
        metrics_df,
        detailed_out,
    )

    result = {
        "experiment_id": "exp05_ready",
        "name": "AUC Cascaded Pipeline Step-3 Consistency (Ready from Exp04)",
        "description": (
            "Post-processing consistency applied to Exp04 outputs: B/I entity-type "
            "reconciliation using bio_prob.  No model retraining."
        ),
        "mode": "ready",
        "source_exp04_xlsx": str(exp04_xlsx),
        "metrics_file": str(metrics_file),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tokens_changed_by_consistency": tokens_changed,
        "status": "ok",
    }

    out_path = write_result_json("exp05_ready", "cascaded_step3_consistent_ready", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    changed = payload.get("tokens_changed_by_consistency", 0)
    print(f"[exp05_ready] F1={f1_str} | tokens changed by consistency: {changed}")
