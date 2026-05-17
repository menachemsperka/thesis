"""
fusion_ready_sources.py — Shared module for ready-results fusion experiments.

Loads pre-computed outputs from Exp01 (Regular NER) and Exp04 (Cascaded Pipeline),
merges them, and provides a generic entry point for applying any fusion strategy
without retraining.

Requires:
    - Exp01 output with a "token_predictions" sheet (sentence_id, token_idx, token,
      true_label, pred_label, prob, entropy, margin).
    - Exp04 output with a "detailed_results" sheet (sentence_id, token_idx, token,
      true_bio, true_etype, pred_bio, pred_etype, entity_prob, bio_prob).

Environment variables (optional — auto-resolved from latest.json if not set):
    THESIS_READY_EXP01_XLSX  — path to Exp01 result xlsx
    THESIS_READY_EXP04_XLSX  — path to Exp04 result xlsx
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from common import write_result_excel, write_result_json


# ---------------------------------------------------------------------------
# Source file resolution
# ---------------------------------------------------------------------------

def _resolve_source(exp_id: str, env_var: str) -> Path:
    """Find the latest output xlsx for *exp_id*, or use an explicit env var."""
    explicit = (os.environ.get(env_var) or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points to a missing file: {p}")
        return p

    latest_json = Path("outputs") / exp_id / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(
            f"Cannot auto-resolve {exp_id} output.  "
            f"Either set {env_var} or run experiment {exp_id} first so that "
            f"{latest_json} exists."
        )

    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    metrics_file = payload.get("metrics_file")
    if not metrics_file:
        raise ValueError(f"metrics_file key missing in {latest_json}")

    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(f"metrics_file from {latest_json} not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _load_regular_from_exp01_legacy_token_level(xlsx_path: Path) -> pd.DataFrame:
    """Compatibility path for older Exp01 files that only have token_level sheet.

    Legacy files do not store per-token probabilities. We synthesize neutral
    confidence features so ready fusion can still execute without retraining.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name="token_level")
    except ValueError:
        raise ValueError(
            f"Exp01 output {xlsx_path} does not have a 'token_predictions' sheet "
            "or a compatible legacy 'token_level' sheet."
        )

    required = {"sentence_id", "token_id", "token", "true_label", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Exp01 legacy token_level in {xlsx_path} is missing columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "sentence_id": df["sentence_id"],
            "token_idx": df["token_id"],
            "token": df["token"],
            "true_label": df["true_label"],
            "regular_pred_label": df["predicted_label"],
            # Unknown confidence in legacy sheet -> neutral defaults.
            "regular_prob": 0.5,
            "regular_entropy": 1.0,
            "regular_margin": 0.0,
        }
    )

    out["sentence_id"] = pd.to_numeric(out["sentence_id"], errors="coerce").fillna(-1).astype(int)
    out["token_idx"] = pd.to_numeric(out["token_idx"], errors="coerce").fillna(-1).astype(int)
    out = out[(out["sentence_id"] >= 0) & (out["token_idx"] >= 0)].copy()
    out["token"] = out["token"].astype(str)
    out["true_label"] = out["true_label"].astype(str)
    out["regular_pred_label"] = out["regular_pred_label"].astype(str)
    return out


def load_regular_from_exp01(xlsx_path: Path) -> pd.DataFrame:
    """Load regular token predictions from Exp01, with legacy compatibility."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="token_predictions")
    except ValueError:
        return _load_regular_from_exp01_legacy_token_level(xlsx_path)

    required = {"sentence_id", "token_idx", "token", "true_label", "pred_label", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp01 token_predictions missing columns: {sorted(missing)}")

    out = df.copy()
    out["sentence_id"] = out["sentence_id"].astype(int)
    out["token_idx"] = out["token_idx"].astype(int)
    out["token"] = out["token"].astype(str)
    out["true_label"] = out["true_label"].astype(str)
    # Rename for fusion consistency
    out.rename(columns={
        "pred_label": "regular_pred_label",
        "prob": "regular_prob",
    }, inplace=True)
    # Optional columns (entropy, margin) — fill with defaults if missing
    if "entropy" in out.columns:
        out.rename(columns={"entropy": "regular_entropy"}, inplace=True)
    else:
        out["regular_entropy"] = 0.0
    if "margin" in out.columns:
        out.rename(columns={"margin": "regular_margin"}, inplace=True)
    else:
        out["regular_margin"] = 1.0
    out["regular_pred_label"] = out["regular_pred_label"].astype(str)
    out["regular_prob"] = pd.to_numeric(out["regular_prob"], errors="coerce").fillna(0.0)
    out["regular_entropy"] = pd.to_numeric(out["regular_entropy"], errors="coerce").fillna(0.0)
    out["regular_margin"] = pd.to_numeric(out["regular_margin"], errors="coerce").fillna(1.0)
    return out


def load_cascade_from_exp04(xlsx_path: Path) -> pd.DataFrame:
    """Load token-level cascaded predictions from an Exp04 (or Exp05) output file."""
    df = pd.read_excel(xlsx_path, sheet_name="detailed_results")

    if "eval_mode" in df.columns:
        df = df[df["eval_mode"].astype(str) == "predicted"].copy()

    required = {"sentence_id", "token_idx", "token", "true_bio", "true_etype",
                "pred_bio", "pred_etype", "entity_prob", "bio_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp04 detailed_results missing columns: {sorted(missing)}")

    df["sentence_id"] = df["sentence_id"].astype(int)
    df["token_idx"] = df["token_idx"].astype(int)

    cascade_true_label = [
        _bio_type_to_label(b, t) for b, t in zip(df["true_bio"], df["true_etype"])
    ]
    cascade_pred_label = [
        _bio_type_to_label(b, t) for b, t in zip(df["pred_bio"], df["pred_etype"])
    ]

    entity_prob = pd.to_numeric(df["entity_prob"], errors="coerce").fillna(0.0).values
    bio_prob = pd.to_numeric(df["bio_prob"], errors="coerce").fillna(0.0).values
    pred_bio = df["pred_bio"].astype(str).values

    cascade_prob = np.where(pred_bio == "O", 1.0 - entity_prob, entity_prob * bio_prob)
    cascade_entropy = np.array([
        -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10) for p in cascade_prob
    ])
    cascade_margin = np.abs(cascade_prob - 0.5) * 2.0

    out = pd.DataFrame({
        "sentence_id": df["sentence_id"].values,
        "token_idx": df["token_idx"].values,
        "token_cascade": df["token"].astype(str).values,
        "cascade_true_label": cascade_true_label,
        "cascade_pred_label": cascade_pred_label,
        "cascade_prob": cascade_prob,
        "cascade_entropy": cascade_entropy,
        "cascade_margin": cascade_margin,
        "entity_prob": entity_prob,
        "bio_prob": bio_prob,
        "cascade_bio": df["pred_bio"].astype(str).values,
        "cascade_etype": df["pred_etype"].astype(str).fillna("None").values,
    })
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_regular_cascade(regular_df: pd.DataFrame, cascade_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on (sentence_id, token_idx) and add derived columns."""
    merged = regular_df.merge(cascade_df, on=["sentence_id", "token_idx"], how="inner")
    if merged.empty:
        raise RuntimeError("No aligned tokens between Exp01 and Exp04 outputs.  "
                           "Were they run with the same data split?")

    merged["true_label"] = merged["true_label"].astype(str)
    merged["cascade_true_label"] = merged["cascade_true_label"].astype(str)
    merged["disagree"] = (
        merged["regular_pred_label"].astype(str) != merged["cascade_pred_label"].astype(str)
    )

    # Derived features used by some strategies
    merged["prob_diff"] = merged["regular_prob"] - merged["cascade_prob"]
    merged["abs_prob_diff"] = merged["prob_diff"].abs()
    merged["max_prob"] = merged[["regular_prob", "cascade_prob"]].max(axis=1)

    # BIO/etype parts for SVM features
    def _split_label(lbl):
        lbl = str(lbl)
        if lbl == "O":
            return "O", "None"
        parts = lbl.split("-", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "None")

    reg_parts = merged["regular_pred_label"].map(_split_label)
    merged["regular_bio"] = reg_parts.map(lambda t: t[0])
    merged["regular_etype"] = reg_parts.map(lambda t: t[1])

    return merged


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def to_seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str):
    y_true, y_pred = [], []
    for sid in sorted(df["sentence_id"].unique()):
        sdf = df[df["sentence_id"] == sid].sort_values("token_idx")
        y_true.append(sdf[true_col].astype(str).tolist())
        y_pred.append(sdf[pred_col].astype(str).tolist())
    return y_true, y_pred


def compute_metrics(df: pd.DataFrame, true_col: str = "true_label", pred_col: str = "fused_pred_label"):
    y_true, y_pred = to_seqeval_lists(df, true_col, pred_col)
    if not y_true:
        return None, None, None
    return (
        float(f1_score(y_true, y_pred)),
        float(precision_score(y_true, y_pred)),
        float(recall_score(y_true, y_pred)),
    )


# ---------------------------------------------------------------------------
# Generic ready-fusion entry point
# ---------------------------------------------------------------------------

def run_ready_fusion(
    *,
    strategy_fn: Callable[[pd.DataFrame], pd.DataFrame],
    experiment_id: str,
    experiment_name: str,
    description: str,
    result_basename: str,
    cascade_source: str = "exp04",
    extra_info: dict | None = None,
) -> dict:
    """
    Load Exp01 + Exp04/05 ready outputs, apply *strategy_fn*, compute metrics, save.

    Parameters
    ----------
    strategy_fn : callable(df) -> df
        Receives the merged DataFrame and must add columns:
        ``fused_pred_label``, ``selected_source``, ``selected_confidence``.
    cascade_source : str
        ``"exp04"`` (default) or ``"exp05"`` — which experiment to load cascade
        predictions from.
    extra_info : dict
        Additional keys to include in the result JSON.
    """
    exp01_xlsx = _resolve_source("exp01", "THESIS_READY_EXP01_XLSX")

    cascade_env_var = "THESIS_READY_EXP05_XLSX" if cascade_source == "exp05" else "THESIS_READY_EXP04_XLSX"
    cascade_xlsx = _resolve_source(cascade_source, cascade_env_var)

    regular_df = load_regular_from_exp01(exp01_xlsx)
    cascade_df = load_cascade_from_exp04(cascade_xlsx)

    merged = merge_regular_cascade(regular_df, cascade_df)

    mismatched_truth = int((merged["true_label"] != merged["cascade_true_label"]).sum())

    # Apply the strategy
    merged = strategy_fn(merged)

    # Validate required output columns
    for col in ("fused_pred_label", "selected_source", "selected_confidence"):
        if col not in merged.columns:
            raise RuntimeError(f"strategy_fn must add column '{col}' to the merged DataFrame")

    f1, precision, recall = compute_metrics(merged)

    disagreement_count = int(merged["disagree"].sum())

    metrics_df = pd.DataFrame([{
        "dataset_name": "ready_results_merge",
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tokens_aligned": len(merged),
        "disagreements": disagreement_count,
        "selected_regular": int(merged["selected_source"].astype(str).str.contains("regular").sum()),
        "selected_cascade": int(merged["selected_source"].astype(str).str.contains("cascade").sum()),
        "agreements": int((~merged["disagree"]).sum()),
        "truth_label_mismatch_between_sources": mismatched_truth,
        "cascade_source": cascade_source,
    }])

    if extra_info:
        for k, v in extra_info.items():
            metrics_df[k] = v

    detailed_cols = [
        "sentence_id", "token_idx", "token", "true_label",
        "regular_pred_label", "regular_prob",
        "cascade_pred_label", "cascade_prob",
        "disagree", "selected_source", "selected_confidence", "fused_pred_label",
    ]
    # Include any extra columns the strategy added
    for col in merged.columns:
        if col not in detailed_cols and col.startswith(("calibrated_", "entropy_", "learned_", "svm_")):
            detailed_cols.append(col)
    detailed_df = merged[[c for c in detailed_cols if c in merged.columns]]

    metrics_file = write_result_excel(
        experiment_id,
        f"{result_basename}_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "regular_from_exp01": regular_df,
            "cascade_from_source": cascade_df,
        },
    )

    result = {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "description": description,
        "mode": "ready",
        "cascade_source": cascade_source,
        "source_exp01_xlsx": str(exp01_xlsx),
        "source_cascade_xlsx": str(cascade_xlsx),
        "metrics_file": str(metrics_file),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "status": "ok",
    }
    if extra_info:
        result.update(extra_info)

    out_path = write_result_json(experiment_id, result_basename, result)
    result["result_file"] = str(out_path)
    return result
