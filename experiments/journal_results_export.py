"""
journal_results_export.py — Tables for IEEE-style papers from cross-comparison outputs.

Designed for:
  * 1–3 training seeds (mean ± SD over seeds in ``cross_comparison_*.xlsx``)
  * 5-fold outer CV for ``06_*_oof`` (``fold_metrics`` sheet per run)
  * Exp04 ``loss_config`` sheet + optional ``outputs/exp04_lambda_grid/`` grid
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

OOF_EXPERIMENT_SUFFIX = "_oof"
JOURNAL_FOCUS_EXP_IDS = (
    "exp01",
    "exp04",
    "exp06_svm_oof",
    "exp06_rf_oof",
)


def _safe_read_excel_sheet(path: Path, sheet: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            return None
        df = pd.read_excel(path, sheet_name=sheet)
        return df if not df.empty else None
    except Exception:
        return None


def collect_loss_config_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per Exp04 (or cascade) metrics workbook with loss_config sheet."""
    if results_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, r in results_df.iterrows():
        exp_id = str(r.get("experiment_id", "")).strip().lower()
        if exp_id not in ("exp04", "exp10_cascade"):
            continue
        mf = str(r.get("metrics_file") or "").strip()
        if not mf or mf in seen:
            continue
        seen.add(mf)
        df = _safe_read_excel_sheet(Path(mf), "loss_config")
        if df is None:
            continue
        row = {c: df.iloc[0][c] for c in df.columns}
        row.update(
            {
                "run_model_name": r.get("model_name"),
                "run_condition_group_short": r.get("condition_group_short") or r.get("condition_group_key"),
                "run_seed": r.get("seed"),
                "metrics_file": mf,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def collect_oof_fold_long(results_df: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (run × outer_fold) for OOF router experiments."""
    if results_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, r in results_df.iterrows():
        exp_id = str(r.get("experiment_id", "")).strip().lower()
        if OOF_EXPERIMENT_SUFFIX not in exp_id:
            continue
        mf = str(r.get("metrics_file") or r.get("result_file") or "").strip()
        if not mf or mf in seen:
            continue
        seen.add(mf)
        fold_df = _safe_read_excel_sheet(Path(mf), "fold_metrics")
        if fold_df is None:
            continue
        for _, fr in fold_df.iterrows():
            rows.append(
                {
                    "model_name": r.get("model_name"),
                    "experiment_id": exp_id,
                    "condition_group_short": r.get("condition_group_short") or r.get("condition_group_key"),
                    "training_seed": r.get("seed"),
                    "outer_fold": fr.get("outer_fold"),
                    "f1": fr.get("f1"),
                    "precision": fr.get("precision"),
                    "recall": fr.get("recall"),
                    "train_sentences": fr.get("train_sentences"),
                    "test_sentences": fr.get("test_sentences"),
                    "metrics_file": mf,
                }
            )
    return pd.DataFrame(rows)


def summarize_oof_for_paper(fold_long: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Table A: mean ± SD of fold F1 (across 5 outer folds), optionally over 1–3 seeds.
    """
    if fold_long.empty:
        return pd.DataFrame()
    gcols = ["model_name", "experiment_id", "condition_group_short"]
    out_rows: list[dict[str, Any]] = []
    for keys, grp in fold_long.groupby(gcols, dropna=False):
        model_name, experiment_id, condition = keys
        f1 = pd.to_numeric(grp["f1"], errors="coerce").dropna()
        out_rows.append(
            {
                "model_name": model_name,
                "experiment_id": experiment_id,
                "condition": condition,
                "n_training_seeds": grp["training_seed"].nunique(),
                "n_outer_folds": len(f1),
                "f1_mean_over_folds": float(f1.mean()) if len(f1) else None,
                "f1_std_over_folds": float(f1.std(ddof=1)) if len(f1) > 1 else (0.0 if len(f1) == 1 else None),
                "f1_min_fold": float(f1.min()) if len(f1) else None,
                "f1_max_fold": float(f1.max()) if len(f1) else None,
            }
        )
    return pd.DataFrame(out_rows)


def paired_fold_method_comparison(fold_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise ΔF1 on matching outer folds (n=5) between journal-focus methods.
    """
    if fold_long.empty:
        return pd.DataFrame()
    focus = fold_long[fold_long["experiment_id"].isin(JOURNAL_FOCUS_EXP_IDS)].copy()
    if focus.empty:
        return pd.DataFrame()

    pivot = focus.pivot_table(
        index=["model_name", "condition_group_short", "training_seed", "outer_fold"],
        columns="experiment_id",
        values="f1",
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()

    pairs = [
        ("exp06_svm_oof", "exp01", "Fusion SVM OOF − Direct NER"),
        ("exp06_svm_oof", "exp04", "Fusion SVM OOF − Cascade"),
        ("exp06_rf_oof", "exp01", "Fusion RF OOF − Direct NER"),
        ("exp04", "exp01", "Cascade − Direct NER"),
    ]
    rows: list[dict[str, Any]] = []
    for col_a, col_b, label in pairs:
        if col_a not in pivot.columns or col_b not in pivot.columns:
            continue
        sub = pivot[[col_a, col_b]].dropna()
        if sub.empty:
            continue
        diffs = sub[col_a].astype(float) - sub[col_b].astype(float)
        rows.append(
            {
                "comparison": label,
                "method_a": col_a,
                "method_b": col_b,
                "n_pairs": int(len(diffs)),
                "delta_f1_mean": float(diffs.mean()),
                "delta_f1_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_journal_main_table(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """Seed-aggregated F1 for focus experiments (split-based Exp01/04 + pooled OOF fusion)."""
    if grouped_df.empty:
        return pd.DataFrame()
    mask = grouped_df["experiment_id"].astype(str).str.lower().isin(JOURNAL_FOCUS_EXP_IDS)
    sub = grouped_df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [
        "model_name",
        "experiment_id",
        "experiment_name",
        "condition_group_short",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "recall_mean",
        "n_seeds",
    ]
    present = [c for c in cols if c in sub.columns]
    out = sub[present].sort_values(["model_name", "condition_group_short", "experiment_id"])
    out = out.rename(
        columns={
            "f1_mean": "entity_f1_mean_across_training_seeds",
            "f1_std": "entity_f1_std_across_training_seeds",
            "n_seeds": "n_training_seeds",
        }
    )
    return out


def load_lambda_grid_selection(project_root: Path | None = None) -> pd.DataFrame:
    """Lambda selection from comparison cache and/or global ``outputs/exp04_lambda_grid/``."""
    root = project_root or Path(__file__).resolve().parents[1]
    out_dir = (os.environ.get("THESIS_CROSS_OUTPUT_DIR") or "").strip()
    if out_dir:
        sel = Path(out_dir) / "exp04_lambda_grid_cache" / "exp04_lambda_selection.json"
        if sel.exists():
            try:
                data = json.loads(sel.read_text(encoding="utf-8"))
                return pd.DataFrame([{"source": str(sel), **data}])
            except Exception:
                pass
    grid_dir = root / "outputs" / "exp04_lambda_grid"
    if not grid_dir.exists():
        return pd.DataFrame()
    json_files = sorted(grid_dir.glob("loss_weight_grid_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    xlsx_files = sorted(grid_dir.glob("loss_weight_grid_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        try:
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            return pd.DataFrame(
                [
                    {
                        "source": str(json_files[0]),
                        "selected_lambda_bio": data.get("selected_lambda_bio"),
                        "selected_lambda_type": data.get("selected_lambda_type"),
                        "selected_validation_pipeline_span_f1": data.get("selected_validation_pipeline_span_f1"),
                        "baseline_10_5_f1": data.get("baseline_10_5_f1"),
                        "selection_metric": (data.get("training_parameters") or {}).get("selection_metric"),
                    }
                ]
            )
        except Exception:
            pass
    if xlsx_files:
        try:
            return pd.read_excel(xlsx_files[0], sheet_name="selection")
        except Exception:
            pass
    return pd.DataFrame()


def journal_documentation_rows() -> list[dict[str, str]]:
    return [
        {"section": "PURPOSE", "item": "profile", "description": "IEEE journal paper (1–3 training seeds; 5-fold OOF fusion)"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_main_table", "description": "Table: Exp01/04/06_*_oof F1 mean±SD over training seeds (Exp07 split)"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_oof_fold_summary", "description": "Table: OOF methods — F1 mean±SD over 5 outer folds (primary CV evidence)"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_oof_folds_long", "description": "Raw fold F1 for paired fold tests (n=5)"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_paired_fold_deltas", "description": "ΔF1 paired by outer_fold between methods"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_loss_config", "description": "λ_bio, λ_type recorded per Exp04 run (see loss_config sheet in cascaded_pipeline_results.xlsx)"},
        {"section": "cross_comparison_*.xlsx", "item": "journal_lambda_grid", "description": "Validation grid selection from experiment_04_loss_weight_grid.py"},
        {"section": "consolidated_error_analysis_*.xlsx", "item": "summary_non_crf_*", "description": "Error types, routing, confusions: Regular vs Cascade vs SVM OOF vs RF OOF"},
        {"section": "consolidated_error_analysis_*.xlsx", "item": "loss_config_summary", "description": "Same λ values pulled from Exp04 metrics workbooks"},
        {"section": "consolidated_error_analysis_*.xlsx", "item": "oof_fold_metrics_long", "description": "Copy of fold-level OOF metrics for appendix figures"},
        {"section": "Methods", "item": "loss weights", "description": "Run experiment_04_loss_weight_grid.py once; set THESIS_EXP04_LAMBDA_BIO/TYPE before cross-comparison"},
        {"section": "Methods", "item": "seeds", "description": "Use --num-seeds 1–3; training_seeds.json in output-dir for reproducibility"},
        {"section": "Guide", "item": "full doc", "description": "thesis_overview.md Part IV"},
    ]
