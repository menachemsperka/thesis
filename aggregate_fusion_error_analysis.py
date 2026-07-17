"""
aggregate_fusion_error_analysis.py — Cross-seed error-analysis aggregation.

Scans the per-run result workbooks produced by the ready-fusion experiments
(``fusion_ready_sources.run_ready_fusion``) and aggregates them ACROSS SEEDS so
they can be reported in a thesis error-analysis chapter.

Each per-run workbook is one (model x split_condition x seed) run and contains
``metrics``, ``per_type_metrics``, ``error_type_summary`` and ``confusion_matrix``
sheets (all carrying model / split_condition / seed columns). This script rolls
them up into a single workbook with:

    * overall_summary        — mean / std / 95% CI of F1/precision/recall per
                               (model, split_condition) across seeds
    * per_seed               — the raw per-seed rows behind the summary
    * per_type_summary       — mean / std of entity-level metrics per entity type
    * error_type_summary     — mean / std of each error category across seeds
    * confusion_matrix_total — token-level confusion matrix summed over all seeds
    * significance           — paired significance tests between models (and
                               between split conditions) on per-seed F1
    * documentation          — explains every sheet

Usage
-----
    python aggregate_fusion_error_analysis.py --exp exp06_svm_ready
    python aggregate_fusion_error_analysis.py --dir outputs/exp06_svm_ready
    python aggregate_fusion_error_analysis.py --all
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is optional
    _scipy_stats = None

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Experiment output folders that use the shared error-analysis result format.
DEFAULT_EXPERIMENT_IDS = [
    "exp01",
    "exp03",
    "exp05_ready",
    "exp06_ready",
    "exp06_normalized_ready",
    "exp06_entropy_ready",
    "exp06_learned_ready",
    "exp06_ensemble_ready",
    "exp06_svm_ready",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _result_files(exp_dir: Path) -> list[Path]:
    """Timestamped result workbooks in *exp_dir* (excludes the *_latest.xlsx copy)."""
    files = sorted(exp_dir.glob("*_results_*.xlsx"))
    return [f for f in files if not f.name.endswith("_latest.xlsx")]


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return None


def _load_runs(exp_dir: Path) -> dict[str, list[pd.DataFrame]]:
    """Collect the relevant sheets from every run workbook in *exp_dir*."""
    collected: dict[str, list[pd.DataFrame]] = {
        "metrics": [],
        "per_type_metrics": [],
        "error_type_summary": [],
        "confusion_matrix": [],
    }
    for path in _result_files(exp_dir):
        metrics = _read_sheet(path, "metrics")
        if metrics is None or metrics.empty:
            continue
        if not {"model", "split_condition", "seed"}.issubset(metrics.columns):
            # Older workbook produced before the run-context columns existed.
            print(f"[skip] {path.name}: missing model/split_condition/seed columns")
            continue
        metrics = metrics.copy()
        metrics["source_file"] = path.name
        metrics["experiment_dir"] = exp_dir.name
        collected["metrics"].append(metrics)

        for sheet in ("per_type_metrics", "error_type_summary", "confusion_matrix"):
            df = _read_sheet(path, sheet)
            if df is not None and not df.empty:
                df = df.copy()
                df["source_file"] = path.name
                df["experiment_dir"] = exp_dir.name
                collected[sheet].append(df)
    return collected


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _ci95(values: np.ndarray) -> float:
    """Half-width of the 95% confidence interval of the mean (t-based)."""
    n = len(values)
    if n < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    sem = sd / math.sqrt(n)
    if _scipy_stats is not None:
        t = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    else:
        t = 1.96
    return t * sem


def _summarise(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "ci95_halfwidth": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_halfwidth": _ci95(arr),
    }


def _overall_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean/std/CI of F1/precision/recall per (experiment, model, split_condition)."""
    group_cols = ["experiment_dir", "model", "split_condition"]
    per_seed = metrics[group_cols + ["seed", "f1", "precision", "recall"]].copy()
    per_seed = per_seed.sort_values(group_cols + ["seed"])

    rows: list[dict] = []
    for keys, gdf in metrics.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(gdf["seed"].nunique())
        for metric in ("f1", "precision", "recall"):
            s = _summarise(gdf[metric])
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
            row[f"{metric}_ci95"] = s["ci95_halfwidth"]
            row[f"{metric}_min"] = s["min"]
            row[f"{metric}_max"] = s["max"]
        rows.append(row)
    return pd.DataFrame(rows), per_seed


def _per_type_summary(per_type: pd.DataFrame) -> pd.DataFrame:
    if per_type.empty:
        return pd.DataFrame()
    group_cols = ["experiment_dir", "model", "split_condition", "entity_type"]
    rows: list[dict] = []
    for keys, gdf in per_type.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(gdf["source_file"].nunique())
        for metric in ("precision", "recall", "f1"):
            if metric in gdf.columns:
                s = _summarise(gdf[metric])
                row[f"{metric}_mean"] = s["mean"]
                row[f"{metric}_std"] = s["std"]
        if "support" in gdf.columns:
            row["support_mean"] = _summarise(gdf["support"])["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def _error_type_summary(error_types: pd.DataFrame) -> pd.DataFrame:
    if error_types.empty:
        return pd.DataFrame()
    group_cols = ["experiment_dir", "model", "split_condition", "error_type"]
    rows: list[dict] = []
    for keys, gdf in error_types.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(gdf["source_file"].nunique())
        row["count_mean"] = _summarise(gdf["count"])["mean"]
        row["count_std"] = _summarise(gdf["count"])["std"]
        if "pct_of_tokens" in gdf.columns:
            row["pct_of_tokens_mean"] = _summarise(gdf["pct_of_tokens"])["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def _confusion_total(confusions: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum the token-level confusion grids across all runs."""
    if not confusions:
        return pd.DataFrame()
    long_frames: list[pd.DataFrame] = []
    for df in confusions:
        d = df.copy()
        # First column holds the true tag label (name may vary), drop bookkeeping cols.
        drop = [c for c in ("source_file", "experiment_dir") if c in d.columns]
        d = d.drop(columns=drop)
        true_col = d.columns[0]
        d = d.rename(columns={true_col: "true"})
        melted = d.melt(id_vars="true", var_name="pred", value_name="count")
        long_frames.append(melted)
    combined = pd.concat(long_frames, ignore_index=True)
    combined["count"] = pd.to_numeric(combined["count"], errors="coerce").fillna(0)
    total = combined.groupby(["true", "pred"])["count"].sum().reset_index()
    grid = total.pivot(index="true", columns="pred", values="count").fillna(0)
    return grid.reset_index()


def _significance(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Paired significance tests on per-seed F1.

    Compares (a) models within the same split_condition, and (b) split_conditions
    within the same model, pairing observations by seed.
    """
    rows: list[dict] = []

    def _pair_test(a: pd.Series, b: pd.Series, label_a: str, label_b: str,
                   grouping: str, context: str) -> None:
        merged = pd.merge(
            a.rename("f1_a"), b.rename("f1_b"),
            left_index=True, right_index=True, how="inner",
        )
        n = len(merged)
        if n < 2:
            return
        diff = merged["f1_a"] - merged["f1_b"]
        row = {
            "comparison_type": grouping,
            "context": context,
            "group_a": label_a,
            "group_b": label_b,
            "n_paired_seeds": n,
            "mean_f1_a": float(merged["f1_a"].mean()),
            "mean_f1_b": float(merged["f1_b"].mean()),
            "mean_diff_a_minus_b": float(diff.mean()),
            "ttest_p": float("nan"),
            "wilcoxon_p": float("nan"),
        }
        if _scipy_stats is not None:
            try:
                row["ttest_p"] = float(_scipy_stats.ttest_rel(merged["f1_a"], merged["f1_b"]).pvalue)
            except Exception:
                pass
            try:
                if diff.abs().sum() > 0:
                    row["wilcoxon_p"] = float(_scipy_stats.wilcoxon(merged["f1_a"], merged["f1_b"]).pvalue)
            except Exception:
                pass
        rows.append(row)

    # (a) models within each (experiment, split_condition)
    for (exp, split), gdf in per_seed.groupby(["experiment_dir", "split_condition"]):
        models = sorted(gdf["model"].unique())
        series = {m: gdf[gdf["model"] == m].set_index("seed")["f1"] for m in models}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                _pair_test(series[models[i]], series[models[j]], models[i], models[j],
                           "model_vs_model", f"{exp} | {split}")

    # (b) split conditions within each (experiment, model)
    for (exp, model), gdf in per_seed.groupby(["experiment_dir", "model"]):
        conds = sorted(gdf["split_condition"].unique())
        series = {c: gdf[gdf["split_condition"] == c].set_index("seed")["f1"] for c in conds}
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                _pair_test(series[conds[i]], series[conds[j]], conds[i], conds[j],
                           "split_vs_split", f"{exp} | {model}")

    return pd.DataFrame(rows)


def _documentation(n_files: int, exp_dirs: list[str]) -> pd.DataFrame:
    have_scipy = _scipy_stats is not None
    docs = [
        ("ABOUT", "generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("ABOUT", "run_workbooks_aggregated", str(n_files)),
        ("ABOUT", "experiment_dirs", ", ".join(exp_dirs)),
        ("ABOUT", "scope",
         "Aggregates per-run fusion result workbooks ACROSS SEEDS. Each source run = "
         "one model x one split_condition x one seed."),
        ("ABOUT", "significance_backend",
         "scipy (paired t-test + Wilcoxon)" if have_scipy
         else "scipy NOT installed — significance p-values are blank; run 'pip install scipy'"),
        ("SHEET", "overall_summary",
         "Mean / std / 95% CI / min / max of entity-level F1, precision, recall per "
         "(experiment, model, split_condition) across seeds. n_seeds = runs aggregated."),
        ("SHEET", "per_seed",
         "The raw per-seed F1/precision/recall rows behind overall_summary. Use these "
         "for your own plots or additional tests."),
        ("SHEET", "per_type_summary",
         "Mean / std of entity-level precision/recall/F1 per entity type across seeds, "
         "with mean support (entity count)."),
        ("SHEET", "error_type_summary",
         "Mean / std across seeds of each token-level error category count and its "
         "share of tokens (correct, false_positive, false_negative, type_error, boundary_error)."),
        ("SHEET", "confusion_matrix_total",
         "Token-level confusion matrix SUMMED over every aggregated run. "
         "Rows = true BIO tag, columns = predicted fused tag."),
        ("SHEET", "significance",
         "Paired significance tests on per-seed F1. comparison_type = model_vs_model "
         "(same split_condition) or split_vs_split (same model). Observations are paired "
         "by seed. ttest_p = paired t-test; wilcoxon_p = Wilcoxon signed-rank. "
         "p < 0.05 = statistically significant difference."),
    ]
    return pd.DataFrame(docs, columns=["section", "item", "description"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate(exp_dirs: list[Path], out_path: Path) -> Path:
    all_runs: dict[str, list[pd.DataFrame]] = {
        "metrics": [], "per_type_metrics": [], "error_type_summary": [], "confusion_matrix": [],
    }
    for d in exp_dirs:
        runs = _load_runs(d)
        for key, frames in runs.items():
            all_runs[key].extend(frames)

    if not all_runs["metrics"]:
        raise SystemExit(
            "No aggregatable run workbooks found. Re-run the fusion experiments so the "
            "result files include model/split_condition/seed columns."
        )

    metrics = pd.concat(all_runs["metrics"], ignore_index=True)
    per_type = (pd.concat(all_runs["per_type_metrics"], ignore_index=True)
                if all_runs["per_type_metrics"] else pd.DataFrame())
    error_types = (pd.concat(all_runs["error_type_summary"], ignore_index=True)
                   if all_runs["error_type_summary"] else pd.DataFrame())

    overall_df, per_seed_df = _overall_summary(metrics)
    per_type_df = _per_type_summary(per_type)
    error_type_df = _error_type_summary(error_types)
    confusion_df = _confusion_total(all_runs["confusion_matrix"])
    significance_df = _significance(per_seed_df)
    documentation_df = _documentation(
        n_files=int(metrics["source_file"].nunique()),
        exp_dirs=[d.name for d in exp_dirs],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        documentation_df.to_excel(writer, sheet_name="documentation", index=False)
        overall_df.to_excel(writer, sheet_name="overall_summary", index=False)
        per_seed_df.to_excel(writer, sheet_name="per_seed", index=False)
        if not per_type_df.empty:
            per_type_df.to_excel(writer, sheet_name="per_type_summary", index=False)
        if not error_type_df.empty:
            error_type_df.to_excel(writer, sheet_name="error_type_summary", index=False)
        if not confusion_df.empty:
            confusion_df.to_excel(writer, sheet_name="confusion_matrix_total", index=False)
        if not significance_df.empty:
            significance_df.to_excel(writer, sheet_name="significance", index=False)

    print(f"[aggregated] {int(metrics['source_file'].nunique())} run(s) -> {out_path}")
    return out_path


def _resolve_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    if args.dir:
        for d in args.dir:
            p = Path(d)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            dirs.append(p)
    if args.exp:
        for e in args.exp:
            dirs.append(OUTPUTS_DIR / e)
    if args.all or (not args.dir and not args.exp):
        dirs = [OUTPUTS_DIR / e for e in DEFAULT_EXPERIMENT_IDS]
    # Keep only existing dirs that contain result files.
    existing = [d for d in dirs if d.exists() and _result_files(d)]
    if not existing:
        raise SystemExit(f"No result workbooks found in: {[str(d) for d in dirs]}")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fusion error analysis across seeds.")
    parser.add_argument("--exp", nargs="*", help="Experiment output folder name(s), e.g. exp06_svm_ready")
    parser.add_argument("--dir", nargs="*", help="Explicit output directory path(s)")
    parser.add_argument("--all", action="store_true", help="Aggregate all known ready-fusion experiments")
    parser.add_argument("--out", help="Output xlsx path (default: outputs/aggregated_error_analysis_<ts>.xlsx)")
    args = parser.parse_args()

    dirs = _resolve_dirs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    elif len(dirs) == 1:
        out_path = dirs[0] / f"aggregated_error_analysis_{timestamp}.xlsx"
    else:
        out_path = OUTPUTS_DIR / f"aggregated_error_analysis_{timestamp}.xlsx"

    aggregate(dirs, out_path)


if __name__ == "__main__":
    main()
