from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from common import write_result_excel, write_result_json


def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _to_seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str) -> tuple[list[list[str]], list[list[str]]]:
    true_lists: list[list[str]] = []
    pred_lists: list[list[str]] = []
    for sentence_id in sorted(df["sentence_id"].unique()):
        sentence_df = df[df["sentence_id"] == sentence_id].sort_values("token_idx")
        true_lists.append(sentence_df[true_col].astype(str).tolist())
        pred_lists.append(sentence_df[pred_col].astype(str).tolist())
    return true_lists, pred_lists


def _resolve_metrics_file(exp_id: str, env_var: str) -> Path:
    explicit = (os.environ.get(env_var) or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points to a missing file: {p}")
        return p

    latest_json = Path("outputs") / exp_id / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(f"Cannot resolve metrics file. Missing: {latest_json}")

    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    metrics_file = payload.get("metrics_file")
    if not metrics_file:
        raise ValueError(f"metrics_file is missing in {latest_json}")

    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(f"metrics_file from {latest_json} not found: {p}")
    return p


def _load_regular_from_exp06(metrics_file: Path) -> pd.DataFrame:
    df = pd.read_excel(metrics_file, sheet_name="detailed_results")
    required = {
        "sentence_id", "token_idx", "token", "true_label", "regular_pred_label", "regular_prob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp06 detailed_results missing required columns: {sorted(missing)}")

    out = df[["sentence_id", "token_idx", "token", "true_label", "regular_pred_label", "regular_prob"]].copy()
    out["sentence_id"] = out["sentence_id"].astype(int)
    out["token_idx"] = out["token_idx"].astype(int)
    out["token"] = out["token"].astype(str)
    out["true_label"] = out["true_label"].astype(str)
    out["regular_pred_label"] = out["regular_pred_label"].astype(str)
    out["regular_prob"] = pd.to_numeric(out["regular_prob"], errors="coerce").fillna(0.0)
    return out


def _load_exp05_predicted(metrics_file: Path) -> pd.DataFrame:
    df = pd.read_excel(metrics_file, sheet_name="detailed_results")
    if "eval_mode" in df.columns:
        df = df[df["eval_mode"].astype(str) == "predicted"].copy()

    required = {
        "sentence_id", "token_idx", "token", "true_bio", "true_etype", "pred_bio", "pred_etype", "entity_prob", "bio_prob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp05 detailed_results missing required columns: {sorted(missing)}")

    out = pd.DataFrame({
        "sentence_id": df["sentence_id"].astype(int),
        "token_idx": df["token_idx"].astype(int),
        "token_exp05": df["token"].astype(str),
        "exp05_true_label": [
            _bio_type_to_label(b, t) for b, t in zip(df["true_bio"], df["true_etype"])
        ],
        "exp05_pred_label": [
            _bio_type_to_label(b, t) for b, t in zip(df["pred_bio"], df["pred_etype"])
        ],
        "exp05_prob": np.where(
            df["pred_bio"].astype(str) == "O",
            1.0 - pd.to_numeric(df["entity_prob"], errors="coerce").fillna(0.0),
            pd.to_numeric(df["entity_prob"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["bio_prob"], errors="coerce").fillna(0.0),
        ),
    })
    return out


def _learn_optimal_alpha(merged: pd.DataFrame) -> float:
    best_alpha = 0.5
    best_f1 = -1.0

    for alpha in np.linspace(0.0, 1.0, 101):
        pred = np.where(
            alpha * merged["regular_prob"].values > (1.0 - alpha) * merged["exp05_prob"].values,
            merged["regular_pred_label"].values,
            merged["exp05_pred_label"].values,
        )
        tmp = merged.copy()
        tmp["fused_pred_label"] = pred
        y_true, y_pred = _to_seqeval_lists(tmp, "true_label", "fused_pred_label")
        cur = float(f1_score(y_true, y_pred)) if y_true else 0.0
        if cur > best_f1:
            best_f1 = cur
            best_alpha = float(alpha)

    return best_alpha


def run_ready_fusion(*, mode: str, experiment_id: str, experiment_name: str, description: str, result_basename: str) -> dict:
    if mode not in {"regular", "learned_weights"}:
        raise ValueError("mode must be one of {'regular', 'learned_weights'}")

    exp06_metrics = _resolve_metrics_file("exp06", "THESIS_READY_EXP06_METRICS_XLSX")
    exp05_metrics = _resolve_metrics_file("exp05", "THESIS_READY_EXP05_METRICS_XLSX")

    regular_df = _load_regular_from_exp06(exp06_metrics)
    exp05_df = _load_exp05_predicted(exp05_metrics)

    merged = regular_df.merge(exp05_df, on=["sentence_id", "token_idx"], how="inner")
    if merged.empty:
        raise RuntimeError("No aligned tokens between Exp06 regular side and Exp05 outputs.")

    merged["true_label"] = merged["true_label"].astype(str)
    merged["exp05_true_label"] = merged["exp05_true_label"].astype(str)
    mismatched_truth = int((merged["true_label"] != merged["exp05_true_label"]).sum())

    merged["disagree"] = merged["regular_pred_label"].astype(str) != merged["exp05_pred_label"].astype(str)

    if mode == "regular":
        use_regular = merged["regular_prob"].values >= merged["exp05_prob"].values
        selected_source = np.where(use_regular, "regular", "exp05")
        selected_conf = np.where(use_regular, merged["regular_prob"].values, merged["exp05_prob"].values)
        fused_pred = np.where(use_regular, merged["regular_pred_label"].values, merged["exp05_pred_label"].values)
        learned_alpha = None
    else:
        learned_alpha = _learn_optimal_alpha(merged)
        use_regular = learned_alpha * merged["regular_prob"].values > (1.0 - learned_alpha) * merged["exp05_prob"].values
        selected_source = np.where(use_regular, "weighted_regular", "weighted_exp05")
        selected_conf = np.where(use_regular, learned_alpha * merged["regular_prob"].values, (1.0 - learned_alpha) * merged["exp05_prob"].values)
        fused_pred = np.where(use_regular, merged["regular_pred_label"].values, merged["exp05_pred_label"].values)

    merged["selected_source"] = selected_source
    merged["selected_confidence"] = selected_conf
    merged["fused_pred_label"] = fused_pred

    y_true, y_pred = _to_seqeval_lists(merged, "true_label", "fused_pred_label")
    precision = float(precision_score(y_true, y_pred)) if y_true else None
    recall = float(recall_score(y_true, y_pred)) if y_true else None
    f1 = float(f1_score(y_true, y_pred)) if y_true else None

    metrics_df = pd.DataFrame([
        {
            "dataset_name": "ready_results_merge",
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tokens_aligned": int(len(merged)),
            "disagreements": int(merged["disagree"].sum()),
            "selected_regular": int((merged["selected_source"].astype(str).str.contains("regular")).sum()),
            "selected_exp05": int((merged["selected_source"].astype(str).str.contains("exp05")).sum()),
            "truth_label_mismatch_between_sources": mismatched_truth,
            "mode": mode,
            "learned_alpha": learned_alpha,
        }
    ])

    detailed_df = merged[
        [
            "sentence_id",
            "token_idx",
            "token",
            "true_label",
            "regular_pred_label",
            "regular_prob",
            "exp05_pred_label",
            "exp05_prob",
            "disagree",
            "selected_source",
            "selected_confidence",
            "fused_pred_label",
        ]
    ]

    metrics_file = write_result_excel(
        experiment_id,
        f"{result_basename}_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "regular_from_exp06": regular_df,
            "exp05_predicted": exp05_df,
        },
    )

    result = {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "description": description,
        "mode": mode,
        "source_metrics_file_exp06": str(exp06_metrics),
        "source_metrics_file_exp05": str(exp05_metrics),
        "metrics_file": str(metrics_file),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "status": "ok",
    }
    if learned_alpha is not None:
        result["learned_weights"] = {
            "alpha_regular": float(learned_alpha),
            "alpha_exp05": float(1.0 - learned_alpha),
        }

    out_path = write_result_json(experiment_id, result_basename, result)
    result["result_file"] = str(out_path)
    return result
