"""
Seed-paired significance tests for the sentence-split strategies.

Reads an existing cross_comparison_*.xlsx (the `all_runs` sheet) and compares the
label-aware greedy and multilabel-stratified splits against the baseline simple
random split, pairing runs by training seed within each model x experiment cell.

Usage:
    python analyze_split_significance.py <cross_comparison.xlsx> [output.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon

BASELINE_CONDITION = "Baseline (simple random split)"
VARIANT_CONDITIONS = [
    "Label-aware greedy",
    "Multilabel stratified (paper-style)",
]
ALPHA = 0.05
# F1 values are stored at full float precision, so an arm that never varied across
# seeds still shows a standard deviation around 1e-16 rather than exactly zero.
CONSTANT_SD_TOL = 1e-9


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def _benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, m * pvals[idx] / (rank + 1))
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def _paired_row(
    baseline: pd.Series, variant: pd.Series, label: dict[str, Any]
) -> dict[str, Any]:
    """Run the paired t-test and Wilcoxon signed-rank test on one matched set."""
    diffs = (variant - baseline).astype(float)
    n = len(diffs)
    delta_mean = float(diffs.mean())
    delta_sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    baseline_sd = float(baseline.std(ddof=1)) if n > 1 else 0.0
    variant_sd = float(variant.std(ddof=1)) if n > 1 else 0.0

    # With reused base artifacts an arm can be identical across every seed. The
    # seeds are then not independent replicates, so a test on them would report a
    # p-value driven by the absence of variance rather than by a real effect.
    if n < 2:
        reason = "fewer than two seed pairs"
    elif delta_sd <= CONSTANT_SD_TOL:
        reason = "per-seed differences are constant"
    elif baseline_sd <= CONSTANT_SD_TOL or variant_sd <= CONSTANT_SD_TOL:
        reason = "one arm is identical across all seeds (base artifacts reused)"
    else:
        reason = ""
    degenerate = bool(reason)

    t_stat = t_p = w_stat = w_p = ci_low = ci_high = dz = None
    if not degenerate:
        tr = ttest_rel(variant.astype(float), baseline.astype(float))
        t_stat, t_p = float(tr.statistic), float(tr.pvalue)
        se = delta_sd / np.sqrt(n)
        crit = float(student_t.ppf(1 - ALPHA / 2, n - 1))
        ci_low, ci_high = delta_mean - crit * se, delta_mean + crit * se
        dz = delta_mean / delta_sd
        try:
            wr = wilcoxon(diffs)
            w_stat, w_p = float(wr.statistic), float(wr.pvalue)
        except ValueError:
            pass

    row = dict(label)
    row.update(
        {
            "n_pairs": n,
            "baseline_f1_mean": float(baseline.mean()),
            "baseline_f1_sd": baseline_sd,
            "variant_f1_mean": float(variant.mean()),
            "variant_f1_sd": variant_sd,
            "delta_f1_mean": delta_mean,
            "delta_f1_sd": delta_sd,
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "cohens_dz": dz,
            "n_pairs_variant_better": int((diffs > 0).sum()),
            "t_statistic": t_stat,
            "t_p_value": t_p,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_value": w_p,
            "degenerate_no_seed_variability": degenerate,
            "degenerate_reason": reason,
        }
    )
    return row


def _add_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg corrections over the testable rows of a family."""
    for col in ("t_p_holm", "t_p_bh", "wilcoxon_p_holm", "wilcoxon_p_bh"):
        df[col] = np.nan
    for src, holm_col, bh_col in (
        ("t_p_value", "t_p_holm", "t_p_bh"),
        ("wilcoxon_p_value", "wilcoxon_p_holm", "wilcoxon_p_bh"),
    ):
        mask = df[src].notna() & ~df["degenerate_no_seed_variability"]
        if not mask.any():
            continue
        pvals = df.loc[mask, src].astype(float).tolist()
        df.loc[mask, holm_col] = _holm(pvals)
        df.loc[mask, bh_col] = _benjamini_hochberg(pvals)
    df["significant_raw_p05"] = df["t_p_value"].notna() & (df["t_p_value"] < ALPHA)
    df["significant_holm_p05"] = df["t_p_holm"].notna() & (df["t_p_holm"] < ALPHA)
    return df


def per_cell_tests(runs: pd.DataFrame) -> pd.DataFrame:
    """One test per model x experiment x variant, paired across the 20 seeds."""
    rows: list[dict[str, Any]] = []
    for (model, exp_id), cell in runs.groupby(["model_name", "experiment_id"]):
        wide = cell.pivot_table(
            index="seed", columns="condition_group_short", values="f1", aggfunc="first"
        )
        if BASELINE_CONDITION not in wide.columns:
            continue
        exp_name = str(cell["experiment_name"].iloc[0])
        for variant in VARIANT_CONDITIONS:
            if variant not in wide.columns:
                continue
            pair = wide[[BASELINE_CONDITION, variant]].dropna()
            if pair.empty:
                continue
            rows.append(
                _paired_row(
                    pair[BASELINE_CONDITION],
                    pair[variant],
                    {
                        "model_name": model,
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "comparison": f"{variant} vs {BASELINE_CONDITION}",
                        "variant": variant,
                    },
                )
            )
    return _add_corrections(pd.DataFrame(rows))


def drop_constant_cells(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Remove model x experiment x condition cells that never varied across seeds.

    Pooling would otherwise repeat one identical difference 20 times and read it
    as 20 independent confirmations of the effect.
    """
    sd = runs.groupby(["model_name", "experiment_id", "condition_group_short"])["f1"].transform(
        lambda s: s.std(ddof=1)
    )
    constant_cells = runs.loc[sd <= CONSTANT_SD_TOL, ["model_name", "experiment_id"]].drop_duplicates()
    if constant_cells.empty:
        return runs
    merged = runs.merge(constant_cells.assign(_drop=True), on=["model_name", "experiment_id"], how="left")
    return merged[merged["_drop"].isna()].drop(columns="_drop")


def pooled_tests(runs: pd.DataFrame, by_model: bool) -> pd.DataFrame:
    """Pool across experiments, pairing on (experiment, seed) instead of seed alone."""
    runs = drop_constant_cells(runs)
    rows: list[dict[str, Any]] = []
    groups = runs.groupby("model_name") if by_model else [("all models", runs)]
    for model, cell in groups:
        index_cols = ["experiment_id", "seed"] if by_model else ["model_name", "experiment_id", "seed"]
        wide = cell.pivot_table(
            index=index_cols, columns="condition_group_short", values="f1", aggfunc="first"
        )
        if BASELINE_CONDITION not in wide.columns:
            continue
        for variant in VARIANT_CONDITIONS:
            if variant not in wide.columns:
                continue
            pair = wide[[BASELINE_CONDITION, variant]].dropna()
            if pair.empty:
                continue
            rows.append(
                _paired_row(
                    pair[BASELINE_CONDITION],
                    pair[variant],
                    {
                        "model_name": model,
                        "experiment_id": "pooled (all experiments)",
                        "experiment_name": "pooled (all experiments)",
                        "comparison": f"{variant} vs {BASELINE_CONDITION}",
                        "variant": variant,
                    },
                )
            )
    return _add_corrections(pd.DataFrame(rows))


def variant_head_to_head(runs: pd.DataFrame) -> pd.DataFrame:
    """Label-aware greedy against multilabel stratified, paired across seeds."""
    a, b = VARIANT_CONDITIONS
    rows: list[dict[str, Any]] = []
    for (model, exp_id), cell in runs.groupby(["model_name", "experiment_id"]):
        wide = cell.pivot_table(
            index="seed", columns="condition_group_short", values="f1", aggfunc="first"
        )
        if a not in wide.columns or b not in wide.columns:
            continue
        pair = wide[[b, a]].dropna()
        if pair.empty:
            continue
        rows.append(
            _paired_row(
                pair[b],
                pair[a],
                {
                    "model_name": model,
                    "experiment_id": exp_id,
                    "experiment_name": str(cell["experiment_name"].iloc[0]),
                    "comparison": f"{a} vs {b}",
                    "variant": a,
                },
            )
        )
    return _add_corrections(pd.DataFrame(rows))


def method_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("source", "all_runs sheet of the cross-comparison workbook"),
            ("pairing", "runs matched on training seed within each model x experiment cell"),
            ("baseline", BASELINE_CONDITION),
            ("tests", "two-sided paired t-test and Wilcoxon signed-rank on per-seed F1 differences"),
            ("effect_size", "Cohen's dz = mean(delta) / sd(delta)"),
            ("ci", "95% CI on the mean delta from the t distribution with n-1 df"),
            ("corrections", "Holm-Bonferroni and Benjamini-Hochberg applied within each sheet"),
            (
                "degenerate_no_seed_variability",
                "TRUE when an arm has no variance across seeds (reused base artifacts), so the "
                "seeds are not independent replicates; p-values are omitted and the row is "
                "excluded from the corrections. See degenerate_reason.",
            ),
            ("sheet split_paired_tests", "one test per model x experiment x split variant"),
            (
                "sheet split_pooled_by_model",
                "pooled across experiments, paired on (experiment, seed); experiments with a "
                "zero-variance arm are excluded from the pool",
            ),
            ("sheet split_pooled_overall", "pooled across models and experiments, same exclusion"),
            ("sheet split_variant_head_to_head", "label-aware greedy vs multilabel stratified"),
        ],
        columns=["field", "description"],
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    dst = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else src.with_name(f"{src.stem}_split_significance.xlsx")
    )

    runs = pd.read_excel(src, sheet_name="all_runs")
    runs = runs[runs["f1"].notna()].copy()
    runs["f1"] = pd.to_numeric(runs["f1"], errors="coerce")

    per_cell = per_cell_tests(runs)
    by_model = pooled_tests(runs, by_model=True)
    overall = pooled_tests(runs, by_model=False)
    head_to_head = variant_head_to_head(runs)

    with pd.ExcelWriter(dst, engine="openpyxl") as writer:
        per_cell.to_excel(writer, sheet_name="split_paired_tests", index=False)
        by_model.to_excel(writer, sheet_name="split_pooled_by_model", index=False)
        overall.to_excel(writer, sheet_name="split_pooled_overall", index=False)
        head_to_head.to_excel(writer, sheet_name="split_variant_head_to_head", index=False)
        method_notes().to_excel(writer, sheet_name="methodology", index=False)

    testable = per_cell[~per_cell["degenerate_no_seed_variability"]]
    print(f"Wrote {dst}")
    print(
        f"  split_paired_tests: {len(per_cell)} comparisons "
        f"({len(testable)} testable, "
        f"{int(testable['significant_holm_p05'].sum())} significant after Holm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
