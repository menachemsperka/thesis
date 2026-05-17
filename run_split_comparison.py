"""
run_split_comparison.py — Run experiments 01–06 across all experiment-07 split variants
========================================================================================

Purpose
-------
This script evaluates the impact of sentence-split strategy on downstream
experiments (01 Regular NER, 03 AUC-2T, 04 Cascaded Pipeline,
05 Step-3 Consistency, 06 Fusion) by running each experiment multiple times
(with different seeds) per split variant saved by experiment 07.

By default, all available experiment-07 variants are used (3 variants),
and each (experiment × variant) combination is run with 5 seeds so that
results can be reported as mean ± std.

Outputs
-------
* ``outputs/cross_comparison/cross_comparison_<timestamp>.xlsx``
    with sheets: all_runs, summary_pivot, variant_summary, deltas,
    experiment_details, and documentation.
* ``outputs/cross_comparison/cross_comparison_latest.xlsx`` (copy).
* ``outputs/cross_comparison/cross_comparison_<timestamp>.json`` and
  ``cross_comparison_latest.json``.

Usage
-----
::

    # Reuse saved experiment-07 splits if valid (recommended default):
    python run_split_comparison.py --exp07-source auto

    # Force experiment 07 rerun before comparison:
    python run_split_comparison.py --exp07-source rerun

    # Require saved splits only (fail if missing/incomplete):
    python run_split_comparison.py --exp07-source saved

    # Run subset of experiments:
    python run_split_comparison.py --exp07-source auto --experiments 03,04,05,06

    # Change seed count:
    python run_split_comparison.py --num-seeds 3

Environment Variables
---------------------
``THESIS_EXP07_SOURCE``
    Split artifact policy: ``auto`` (default), ``saved``, or ``rerun``.
``THESIS_SPLIT_COMPARISON_EXPERIMENTS``
    Comma-separated list of experiment IDs to include (default: ``01,03,04,05,06``).
``THESIS_SPLIT_COMPARISON_NUM_SEEDS``
    Number of seeds per (experiment × variant) run (default: ``5``).
``THESIS_MODEL_NAME``
    Override the model path/ID.
``THESIS_DEBUG``
    Set to ``1`` for verbose output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "exp07" / "splits"
COMPARISON_DIR = OUTPUTS_DIR / "cross_comparison"

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _get_exp07_variants() -> list[str]:
    """Lazily import ALL_VARIANTS from experiment 07."""
    from experiment_07_sentence_split_strategy import ALL_VARIANTS
    return ALL_VARIANTS

# ---------------------------------------------------------------------------
# Experiment imports (deferred to avoid import-time side-effects)
# ---------------------------------------------------------------------------
_EXPERIMENT_MODULES: dict[str, Any] = {}


def _import_experiment(exp_id: str):
    """Lazily import an experiment module and return its ``run`` function."""
    if exp_id in _EXPERIMENT_MODULES:
        return _EXPERIMENT_MODULES[exp_id]

    module_map = {
        "01": "experiment_01_regular_ner",
        "03": "experiment_03_auc_2t",
        "04": "experiment_04_auc_cascaded_pipeline",
        "05": "experiment_05_auc_cascaded_pipeline_step3_consistency",
        "06": "experiment_06_fusion_regular_and_cascaded",
        "07": "experiment_07_sentence_split_strategy",
    }
    module_name = module_map.get(exp_id)
    if module_name is None:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    import importlib
    mod = importlib.import_module(module_name)
    _EXPERIMENT_MODULES[exp_id] = mod
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(message: str) -> None:
    """Print a timestamped progress line for long-running comparisons."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _load_split_meta() -> dict:
    meta_path = SPLITS_DIR / "split_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Split meta not found at {meta_path}. "
            "Run experiment 07 first to generate the splits."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_artifacts_ready() -> tuple[bool, str]:
    """Validate that saved exp07 split artifacts exist and are complete."""
    meta_path = SPLITS_DIR / "split_meta.json"
    if not meta_path.exists():
        return False, f"Missing file: {meta_path}"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Cannot parse split_meta.json: {exc}"

    variants = meta.get("variants")
    if not isinstance(variants, list) or not variants:
        return False, "split_meta.json has no variants list"

    for vm in variants:
        train_file = vm.get("train_file")
        eval_file = vm.get("eval_file")
        if not train_file or not eval_file:
            return False, f"Variant entry missing train/eval file: {vm}"
        train_path = SPLITS_DIR / str(train_file)
        eval_path = SPLITS_DIR / str(eval_file)
        if not train_path.exists() or not eval_path.exists():
            return False, f"Missing split files for variant {vm.get('variant')}: {train_path.name}, {eval_path.name}"

    return True, "ok"


def _prepare_exp07_splits(exp07_source: str) -> dict[str, Any]:
    """Prepare exp07 split files based on source policy.

    exp07_source:
      - "saved": require existing saved split artifacts.
      - "rerun": run experiment 07 now to regenerate split artifacts.
      - "auto": use saved artifacts if valid, otherwise rerun exp07.
    """
    mode = (exp07_source or "auto").strip().lower()
    if mode not in {"saved", "rerun", "auto"}:
        raise ValueError(f"Invalid exp07 source mode: {exp07_source}")

    ready, reason = _split_artifacts_ready()

    if mode == "saved":
        if not ready:
            raise FileNotFoundError(
                "Requested saved exp07 artifacts, but they are not ready: "
                f"{reason}. Run with --exp07-source rerun once."
            )
        _log("Using existing saved exp07 split artifacts.")
        return {"source": "saved", "reran_exp07": False}

    if mode == "auto" and ready:
        _log("Saved exp07 split artifacts are valid; reusing them.")
        return {"source": "saved", "reran_exp07": False}

    _log("Running experiment 07 to generate split artifacts...")
    t0 = time.time()
    mod07 = _import_experiment("07")
    payload07 = mod07.run()
    elapsed = time.time() - t0

    ready_after, reason_after = _split_artifacts_ready()
    if not ready_after:
        raise RuntimeError(
            "Experiment 07 completed but split artifacts are still incomplete: "
            f"{reason_after}"
        )

    _log(f"Experiment 07 completed in {elapsed:.1f}s; split artifacts are ready.")
    return {
        "source": "rerun",
        "reran_exp07": True,
        "exp07_result_file": payload07.get("result_file"),
        "exp07_metrics_file": payload07.get("metrics_file"),
    }


def _set_presplit_env(train_json: Path, eval_json: Path) -> None:
    """Set environment variables to direct experiments to pre-split data."""
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(train_json)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(eval_json)


def _clear_presplit_env() -> None:
    """Remove pre-split environment variables so the next run uses defaults."""
    os.environ.pop("THESIS_PRESPLIT_TRAIN_JSON", None)
    os.environ.pop("THESIS_PRESPLIT_EVAL_JSON", None)


def _extract_metrics(payload: dict) -> dict:
    """Extract the key metrics from an experiment's result payload."""
    return {
        "f1": payload.get("f1"),
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "status": payload.get("status", "ok"),
    }


EXP_NAMES = {
    "01": "Regular NER",
    "03": "AUC-2T",
    "04": "AUC Cascaded Pipeline",
    "05": "AUC Cascaded Step-3 Consistency",
    "06": "Fusion (Regular + Cascaded)",
}


def _to_float(v: Any) -> float | None:
    """Safely cast to float."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(
    experiment_ids: list[str] | None = None,
    exp07_source: str = "auto",
    num_seeds: int = 5,
) -> dict:
    """Run each experiment with ALL exp07 split variants and compare.

    Parameters
    ----------
    experiment_ids : list[str] | None
        Which experiments to include (default ``["01", "03", "04", "05", "06"]``).
    num_seeds : int
        Number of seeds per (experiment x variant) run (default: 5).

    Returns
    -------
    dict
        Full result payload including per-seed metrics, aggregated stats,
        and delta analysis for every exp07 split variant.
    """
    if experiment_ids is None:
        raw = (os.environ.get("THESIS_SPLIT_COMPARISON_EXPERIMENTS") or "01,03,04,05,06").strip()
        experiment_ids = [x.strip() for x in raw.split(",") if x.strip()]

    base_seed = 42

    _log(f"Preparing split artifacts with exp07 source mode: {exp07_source}")
    prep_info = _prepare_exp07_splits(exp07_source)

    meta = _load_split_meta()
    variants_meta = meta.get("variants", [])
    # Filter to only the variants that still exist in the current codebase
    exp07_variants = _get_exp07_variants()
    variants_meta = [vm for vm in variants_meta if vm["variant"] in exp07_variants]
    if not variants_meta:
        raise RuntimeError(
            "split_meta.json has no 'variants' list. "
            "Re-run experiment 07 to generate per-variant split files."
        )

    # Build conditions: one per exp07 variant
    conditions: list[tuple[str, str, Path, Path]] = []  # (condition_name, label, train_path, eval_path)
    baseline_condition: str | None = None
    for vm in variants_meta:
        variant_key = vm["variant"]
        label = vm["label"]
        train_path = SPLITS_DIR / vm["train_file"]
        eval_path = SPLITS_DIR / vm["eval_file"]
        if not train_path.exists() or not eval_path.exists():
            print(f"  WARNING: split files missing for {label}, skipping")
            continue
        conditions.append((variant_key, label, train_path, eval_path))
        if variant_key == meta.get("baseline_variant"):
            baseline_condition = variant_key

    if not conditions:
        raise FileNotFoundError("No valid split files found in " + str(SPLITS_DIR))

    rows: list[dict] = []  # per-seed rows

    total_runs = len(experiment_ids) * len(conditions) * num_seeds
    run_counter = 0
    comparison_start = time.time()

    _log(
        f"Starting comparison: {len(experiment_ids)} experiments x "
        f"{len(conditions)} variants x {num_seeds} seeds = {total_runs} runs"
    )

    for exp_id in experiment_ids:
        exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
        print(f"\n{'='*60}")
        print(f"  Experiment {exp_id}: {exp_name}  ({len(conditions)} variants x {num_seeds} seeds)")
        print(f"{'='*60}")

        for variant_key, variant_label, train_path, eval_path in conditions:
            for seed_idx in range(num_seeds):
                seed = base_seed + seed_idx
                run_counter += 1
                run_t0 = time.time()
                _log(
                    f"Run {run_counter}/{total_runs} | exp{exp_id} | {variant_label} | "
                    f"seed={seed} | train={train_path.name} eval={eval_path.name}"
                )
                _set_presplit_env(train_path, eval_path)
                os.environ["THESIS_SPLIT_SEED"] = str(seed)
                try:
                    mod = _import_experiment(exp_id)
                    payload = mod.run()
                    metrics = _extract_metrics(payload)
                except Exception as exc:
                    traceback.print_exc()
                    metrics = {"f1": None, "precision": None, "recall": None, "status": f"error: {exc}"}
                    payload = {}
                finally:
                    _clear_presplit_env()
                    os.environ.pop("THESIS_SPLIT_SEED", None)

                row = {
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "data_source": "exp07",
                    "condition_key": f"exp07_{variant_key}",
                    "condition_label": f"[Exp07] {variant_label}",
                    "condition_short": variant_label,
                    "condition_description": next(
                        (vm["description"] for vm in variants_meta if vm["variant"] == variant_key),
                        variant_key,
                    ),
                    "is_baseline": variant_key == baseline_condition,
                    "seed": seed,
                    "f1": metrics.get("f1"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "status": metrics.get("status"),
                    "result_file": payload.get("result_file", ""),
                    "metrics_file": payload.get("metrics_file", ""),
                    "elapsed_seconds": round(time.time() - run_t0, 1),
                }
                rows.append(row)

                f1_str = f"{metrics['f1']:.4f}" if metrics.get("f1") is not None else "N/A"
                run_elapsed = time.time() - run_t0
                _log(f"Completed run {run_counter}/{total_runs} | F1={f1_str} | elapsed={run_elapsed:.1f}s")

    total_elapsed = time.time() - comparison_start
    results_df = pd.DataFrame(rows)

    # ==================================================================
    # Post-processing: build analytical sheets
    # ==================================================================

    # ── 1. Summary pivot: one row per experiment, columns = variant mean F1
    pivot_rows: list[dict] = []
    for exp_id in experiment_ids:
        exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
        exp_results = results_df[results_df["experiment_id"] == f"exp{exp_id}"]
        pivot_row: dict[str, Any] = {"experiment_id": f"exp{exp_id}", "experiment_name": exp_name}
        for variant_key, variant_label, _, _ in conditions:
            cond_results = exp_results[exp_results["condition_key"] == f"exp07_{variant_key}"]
            f1s = pd.to_numeric(cond_results["f1"], errors="coerce").dropna()
            if not f1s.empty:
                mean = float(f1s.mean())
                std = float(f1s.std()) if len(f1s) > 1 else 0.0
                pivot_row[variant_label] = mean
                pivot_row[f"{variant_label}_std"] = std
        pivot_rows.append(pivot_row)
    pivot_df = pd.DataFrame(pivot_rows) if pivot_rows else pd.DataFrame()

    # ── 2. Exp07 deltas: each variant vs baseline (using seed-aggregated means)
    delta_rows: list[dict] = []
    for exp_id in experiment_ids:
        exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
        exp_results = results_df[results_df["experiment_id"] == f"exp{exp_id}"]
        # Baseline aggregated metrics
        b_results = exp_results[exp_results["condition_key"] == f"exp07_{baseline_condition}"]
        b_f1s = pd.to_numeric(b_results["f1"], errors="coerce").dropna()
        b_precs = pd.to_numeric(b_results["precision"], errors="coerce").dropna()
        b_recs = pd.to_numeric(b_results["recall"], errors="coerce").dropna()
        if b_f1s.empty:
            continue
        b_f1_mean = float(b_f1s.mean())
        b_prec_mean = float(b_precs.mean()) if not b_precs.empty else None
        b_rec_mean = float(b_recs.mean()) if not b_recs.empty else None
        for variant_key, variant_label, _, _ in conditions:
            if variant_key == baseline_condition:
                continue
            a_results = exp_results[exp_results["condition_key"] == f"exp07_{variant_key}"]
            a_f1s = pd.to_numeric(a_results["f1"], errors="coerce").dropna()
            a_precs = pd.to_numeric(a_results["precision"], errors="coerce").dropna()
            a_recs = pd.to_numeric(a_results["recall"], errors="coerce").dropna()
            if a_f1s.empty:
                continue
            a_f1_mean = float(a_f1s.mean())
            a_prec_mean = float(a_precs.mean()) if not a_precs.empty else None
            a_rec_mean = float(a_recs.mean()) if not a_recs.empty else None
            delta_rows.append({
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
                "variant": variant_label,
                "baseline_f1": b_f1_mean,
                "variant_f1": a_f1_mean,
                "delta_f1": a_f1_mean - b_f1_mean,
                "baseline_precision": b_prec_mean,
                "variant_precision": a_prec_mean,
                "delta_precision": (a_prec_mean - b_prec_mean) if a_prec_mean is not None and b_prec_mean is not None else None,
                "baseline_recall": b_rec_mean,
                "variant_recall": a_rec_mean,
                "delta_recall": (a_rec_mean - b_rec_mean) if a_rec_mean is not None and b_rec_mean is not None else None,
            })
    deltas_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame()

    # ── 3. Variant summary across all experiments and seeds
    variant_summary_rows: list[dict] = []
    for variant_key, variant_label, _, _ in conditions:
        cond_results = results_df[results_df["condition_key"] == f"exp07_{variant_key}"]
        f1s = pd.to_numeric(cond_results["f1"], errors="coerce").dropna()
        variant_summary_rows.append({
            "data_source": "exp07",
            "condition": variant_label,
            "description": next(
                (vm["description"] for vm in variants_meta if vm["variant"] == variant_key),
                variant_key,
            ),
            "num_runs": len(cond_results),
            "f1_mean": float(f1s.mean()) if not f1s.empty else None,
            "f1_std": float(f1s.std()) if len(f1s) > 1 else None,
            "f1_min": float(f1s.min()) if not f1s.empty else None,
            "f1_max": float(f1s.max()) if not f1s.empty else None,
        })
    variant_summary_df = (
        pd.DataFrame(variant_summary_rows)
        .sort_values(by="f1_mean", ascending=False, ignore_index=True)
        if variant_summary_rows else pd.DataFrame()
    )

    # ── 4. Documentation sheet
    from split_io import build_thesis_documentation_df
    variant_labels_str = "; ".join(vm["label"] for vm in variants_meta)
    doc_df = build_thesis_documentation_df(
        "cross_comparison",
        "Split Comparison: Experiment-07 Split Strategies x Experiments 01-06",
        extra_rows=[
            {"Section": "Design", "Key": "Conditions",
             "Value": f"All {len(conditions)} exp07 variants: {variant_labels_str}"},
            {"Section": "Design", "Key": "Seeds per run",
             "Value": str(num_seeds)},
            {"Section": "Design", "Key": "Comparison",
             "Value": "Each experiment is run N seeds per split variant; deltas computed on mean F1 vs baseline"},
            {"Section": "Design", "Key": "Best Variant (from Exp07)",
             "Value": meta.get("best_variant_label", "N/A")},
            {"Section": "Design", "Key": "Best Variant F1 Mean (Exp07)",
             "Value": f"{meta.get('best_variant_f1_mean', 'N/A')}"},
            {"Section": "Design", "Key": "Downstream Experiments",
             "Value": "; ".join(f"exp{e}: {EXP_NAMES.get(e, e)}" for e in experiment_ids)},
            {"Section": "Design", "Key": "Total Runs",
             "Value": str(total_runs)},
            {"Section": "Interpretation", "Key": "Positive delta_f1",
             "Value": "The split variant improved that experiment's mean F1 vs baseline"},
            {"Section": "Interpretation", "Key": "Negative delta_f1",
             "Value": "The baseline split was better for that experiment"},
            {"Section": "Sheets", "Key": "all_runs",
             "Value": "Per-seed, per-experiment, per-variant F1/precision/recall"},
            {"Section": "Sheets", "Key": "summary_pivot",
             "Value": "Mean F1 pivot table: one row per experiment, one column per variant (with std)"},
            {"Section": "Sheets", "Key": "variant_summary",
             "Value": "Per-variant aggregate statistics (mean/std/min/max F1) across all experiments and seeds"},
            {"Section": "Sheets", "Key": "deltas",
             "Value": "Paired delta (variant mean - baseline mean) for each experiment x variant"},
            {"Section": "Sheets", "Key": "experiment_details",
             "Value": "Extended detail including file paths and descriptions"},
            {"Section": "Sheets", "Key": "documentation",
             "Value": "This sheet - describes every column and how to cite the results"},
        ],
    )

    # ==================================================================
    # Write outputs to cross_comparison/
    # ==================================================================
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()

    # ── Excel
    xlsx_path = COMPARISON_DIR / f"cross_comparison_{ts}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not pivot_df.empty:
            pivot_df.to_excel(writer, sheet_name="summary_pivot", index=False)
        results_df.to_excel(writer, sheet_name="all_runs", index=False)
        if not deltas_df.empty:
            deltas_df.to_excel(writer, sheet_name="deltas", index=False)
        if not variant_summary_df.empty:
            variant_summary_df.to_excel(writer, sheet_name="variant_summary", index=False)
        details_df = results_df[[
            "experiment_id", "experiment_name",
            "data_source", "condition_label", "condition_description",
            "seed", "f1", "precision", "recall", "status",
            "result_file", "metrics_file", "elapsed_seconds",
        ]].copy()
        details_df.to_excel(writer, sheet_name="experiment_details", index=False)
        doc_df.to_excel(writer, sheet_name="documentation", index=False)

    latest_xlsx = COMPARISON_DIR / "cross_comparison_latest.xlsx"
    if latest_xlsx.exists():
        latest_xlsx.unlink()
    shutil.copy2(xlsx_path, latest_xlsx)

    # ── JSON
    payload_out: dict[str, Any] = {
        "name": "Split Comparison: Experiment 07 Variants x Experiments 01-06",
        "description": (
            f"Runs experiments {', '.join(experiment_ids)} with "
            f"{len(conditions)} split variants x {num_seeds} seeds. "
            f"Total runs: {total_runs}."
        ),
        "experiments": experiment_ids,
        "exp07_preparation": prep_info,
        "exp07_meta": meta,
        "num_variants": len(conditions),
        "num_experiments": len(experiment_ids),
        "num_seeds": num_seeds,
        "total_runs": len(rows),
        "elapsed_seconds": round(total_elapsed, 1),
        "results": results_df.to_dict(orient="records"),
        "summary_pivot": pivot_df.to_dict(orient="records") if not pivot_df.empty else [],
        "variant_summary": variant_summary_df.to_dict(orient="records") if not variant_summary_df.empty else [],
        "deltas": deltas_df.to_dict(orient="records") if not deltas_df.empty else [],
        "xlsx": str(xlsx_path),
        "xlsx_latest": str(latest_xlsx),
        "status": "ok",
    }
    json_path = COMPARISON_DIR / f"cross_comparison_{ts}.json"
    json_path.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    latest_json = COMPARISON_DIR / "cross_comparison_latest.json"
    latest_json.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    _log(f"Comparison complete in {total_elapsed:.1f}s")
    return payload_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp01-06 across exp07 split variants with multiple seeds.")
    parser.add_argument(
        "--exp07-source",
        choices=["auto", "saved", "rerun"],
        default=(os.environ.get("THESIS_EXP07_SOURCE") or "auto").strip().lower(),
        help="How to prepare exp07 split artifacts: auto (default), saved, or rerun.",
    )
    parser.add_argument(
        "--experiments",
        default=(os.environ.get("THESIS_SPLIT_COMPARISON_EXPERIMENTS") or "01,03,04,05,06").strip(),
        help="Comma-separated experiment IDs to run, e.g. 01,03,04,05,06",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=int((os.environ.get("THESIS_SPLIT_COMPARISON_NUM_SEEDS") or "5").strip()),
        help="Number of seeds per (experiment x variant) run (default: 5)",
    )
    args = parser.parse_args()

    experiment_ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    result = run_comparison(
        experiment_ids=experiment_ids,
        exp07_source=args.exp07_source,
        num_seeds=args.num_seeds,
    )
    print("\n" + "=" * 70)
    print("  SPLIT COMPARISON RESULTS  (all exp07 variants x experiments)")
    print("=" * 70)

    # Print pivot table (mean F1 per variant)
    pivot = result.get("summary_pivot", [])
    if pivot:
        # Get variant columns (exclude _std, experiment_id, experiment_name)
        variant_cols = [k for k in pivot[0].keys()
                        if k not in ("experiment_id", "experiment_name") and not k.endswith("_std")]
        col_width = 12
        header = f"{'Experiment':<35s}" + "".join(f"{c[:col_width]:>{col_width+1}s}" for c in variant_cols)
        print(f"\n  Mean F1 ({args.num_seeds} seeds):\n")
        print(f"{header}")
        print("-" * len(header))
        for row in pivot:
            line = f"  {row['experiment_name']:<33s}"
            for c in variant_cols:
                val = row.get(c)
                std = row.get(f"{c}_std")
                if val is not None:
                    std_str = f"\u00b1{std:.3f}" if std is not None else ""
                    line += f"  {val:>{col_width-6}.4f}{std_str}"
                else:
                    line += f"  {'N/A':>{col_width}s}"
            print(line)

    # Print variant ranking
    vsummary = result.get("variant_summary", [])
    if vsummary:
        print(f"\n{'Variant':<35s} {'Mean F1':>10s} {'Std':>8s} {'Min F1':>10s} {'Max F1':>10s} {'Runs':>6s}")
        print("-" * 81)
        for vs in vsummary:
            mean_str = f"{vs['f1_mean']:.4f}" if vs.get("f1_mean") is not None else "N/A"
            std_str = f"{vs['f1_std']:.4f}" if vs.get("f1_std") is not None else "N/A"
            min_str = f"{vs['f1_min']:.4f}" if vs.get("f1_min") is not None else "N/A"
            max_str = f"{vs['f1_max']:.4f}" if vs.get("f1_max") is not None else "N/A"
            runs_str = str(vs.get("num_runs", "?"))
            print(f"  {vs['condition']:<33s} {mean_str:>10s} {std_str:>8s} {min_str:>10s} {max_str:>10s} {runs_str:>6s}")

    # Print deltas vs baseline
    deltas = result.get("deltas", [])
    if deltas:
        print(f"\n{'Experiment':<30s} {'Variant':<30s} {'Delta F1':>10s}")
        print("-" * 72)
        for d in deltas:
            delta = f"{d['delta_f1']:+.4f}" if d.get("delta_f1") is not None else "N/A"
            print(f"  {d['experiment_name']:<28s} {d['variant']:<28s} {delta:>10s}")

    print(f"\n  Total runs: {result.get('total_runs', '?')}")
    print(f"  Seeds:      {args.num_seeds}")
    print(f"  Elapsed:    {float(result.get('elapsed_seconds', 0.0)):.1f}s")
    print(f"  Excel: {result.get('xlsx_latest', 'N/A')}")
