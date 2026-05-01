"""run_experiments_with_exp08_data.py — Run Experiments 03-06 with Exp08 Augmented Data
======================================================================================

This script runs experiments 03 (AUC-2T), 04 (AUC Cascaded Pipeline),
05 (AUC Cascaded + Step3 Consistency), and 06 (Fusion) using the training
datasets produced by experiment 08 (LLM mask-filling augmentation).

It runs each experiment twice:
  1. **Baseline**: original training data from exp08 (no augmentation).
  2. **Augmented**: augmented training data from exp08 (original + LLM-generated).

Both conditions use the **same eval set** (never modified).

Auto-preparation
----------------
If exp08 split files are missing, this script automatically runs
``experiments/experiment_08_llm_augmentation.py`` first.

You can also force rerunning experiment 08 even when split files already
exist (see usage below).

Usage
-----
::

    python run_experiments_with_exp08_data.py

    # Force rerun experiment 08 first (even if split files exist)
    python run_experiments_with_exp08_data.py --force-exp08

    # Run only specific experiments (comma-separated)
    set THESIS_EXP08_RUN_EXPERIMENTS=03,04
    python run_experiments_with_exp08_data.py

    # Force rerun experiment 08 via environment variable
    set THESIS_EXP08_FORCE_RERUN=1
    python run_experiments_with_exp08_data.py

Outputs
-------
* ``outputs/exp08_comparison/*.xlsx`` — Excel with all results, deltas, pivot
* ``outputs/exp08_comparison/*.json`` — machine-readable results
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "exp08" / "splits"

# Ensure experiments/ is importable
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


# ── Experiment registry ───────────────────────────────────────────────────

EXP_NAMES = {
    "03": "AUC-2T",
    "04": "AUC Cascaded Pipeline",
    "05": "AUC Cascaded Step-3 Consistency",
    "06": "Fusion (Regular + Cascaded)",
}

EXP_SCRIPTS = {
    "03": "experiment_03_auc_2t",
    "04": "experiment_04_auc_cascaded_pipeline",
    "05": "experiment_05_auc_cascaded_pipeline_step3_consistency",
    "06": "experiment_06_fusion_regular_and_cascaded",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


def _import_experiment(exp_id: str):
    """Dynamically import an experiment module."""
    module_name = EXP_SCRIPTS.get(exp_id)
    if module_name is None:
        raise ValueError(f"Unknown experiment ID: {exp_id}")
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _set_presplit_env(train_json: Path, eval_json: Path) -> None:
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(train_json)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(eval_json)


def _clear_presplit_env() -> None:
    os.environ.pop("THESIS_PRESPLIT_TRAIN_JSON", None)
    os.environ.pop("THESIS_PRESPLIT_EVAL_JSON", None)


def _extract_metrics(payload: dict) -> dict:
    return {
        "f1": payload.get("f1"),
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "status": payload.get("status", "ok"),
    }


def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


# ── Validation ────────────────────────────────────────────────────────────

def _validate_splits() -> dict:
    """Check that exp08 split files exist and load metadata."""
    meta_path = SPLITS_DIR / "split_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Exp08 split metadata not found: {meta_path}\n"
            "Run experiment 08 first:\n"
            "  python experiments/experiment_08_llm_augmentation.py"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    required_files = [
        "baseline_train.json", "baseline_eval.json",
        "augmented_train.json", "augmented_eval.json",
    ]
    for f in required_files:
        p = SPLITS_DIR / f
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    return meta


def _split_files_ready() -> tuple[bool, str]:
    """Return whether exp08 split artifacts are present and valid."""
    if not SPLITS_DIR.exists():
        return False, f"split directory does not exist: {SPLITS_DIR}"

    required_files = [
        "baseline_train.json", "baseline_eval.json",
        "augmented_train.json", "augmented_eval.json", "split_meta.json",
    ]
    missing = [name for name in required_files if not (SPLITS_DIR / name).exists()]
    if missing:
        return False, f"missing files: {', '.join(missing)}"

    try:
        _ = json.loads((SPLITS_DIR / "split_meta.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"split_meta.json unreadable: {exc}"

    return True, "ok"


def _prepare_exp08_splits(force_rerun: bool = False) -> dict:
    """Ensure exp08 split files exist, optionally forcing a fresh exp08 run."""
    ready, reason = _split_files_ready()

    if not force_rerun and ready:
        _log(f"Using existing exp08 splits ({SPLITS_DIR})")
        return {"source": "saved", "reran_exp08": False, "reason": "already_ready"}

    if force_rerun:
        _log("Force mode enabled: rerunning experiment 08 before comparison")
    else:
        _log(f"Exp08 split artifacts not ready: {reason}")
        _log("Running experiment 08 to generate augmentation datasets...")

    t0 = time.time()
    mod08 = importlib.import_module("experiment_08_llm_augmentation")
    mod08 = importlib.reload(mod08)
    payload08 = mod08.run()
    elapsed = time.time() - t0

    ready_after, reason_after = _split_files_ready()
    if not ready_after:
        raise RuntimeError(
            "Experiment 08 completed but split artifacts are still incomplete: "
            f"{reason_after}"
        )

    _log(f"Experiment 08 completed in {elapsed:.1f}s; split artifacts are ready")
    return {
        "source": "rerun",
        "reran_exp08": True,
        "reason": "forced" if force_rerun else "missing_or_invalid",
        "exp08_result_file": payload08.get("result_file"),
        "exp08_metrics_file": payload08.get("metrics_file"),
    }


# ── Conditions ────────────────────────────────────────────────────────────

CONDITIONS = [
    {
        "key": "exp08_baseline",
        "label": "Exp08 Baseline (no augmentation)",
        "train_file": "baseline_train.json",
        "eval_file": "baseline_eval.json",
    },
    {
        "key": "exp08_augmented",
        "label": "Exp08 Augmented (LLM mask-fill)",
        "train_file": "augmented_train.json",
        "eval_file": "augmented_eval.json",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────

def run(force_exp08_rerun: bool = False) -> dict:
    from experiments.common import configure_network_environment, configure_model_environment
    configure_network_environment()
    configure_model_environment()

    raw_ids = (os.environ.get("THESIS_EXP08_RUN_EXPERIMENTS") or "03,04,05,06").strip()
    experiment_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]

    env_force = (os.environ.get("THESIS_EXP08_FORCE_RERUN") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    effective_force = force_exp08_rerun or env_force

    prep_info = _prepare_exp08_splits(force_rerun=effective_force)
    meta = _validate_splits()

    print("=" * 70)
    print("Run Experiments 03-06 with Exp08 Augmented Data")
    print(f"  Split seed: {meta.get('seed')}")
    print(f"  Baseline train: {meta.get('baseline_train_sentences')} sentences")
    print(f"  Augmented train: {meta.get('augmented_train_sentences')} sentences "
          f"(+{meta.get('generated_sentences')} generated)")
    print(f"  Eval: {meta.get('eval_sentences')} sentences (untouched)")
    print(f"  Experiments: {', '.join(experiment_ids)}")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Exp08 split source: {prep_info.get('source')}"
          f" ({prep_info.get('reason', 'n/a')})")
    print("=" * 70)

    rows: list[dict] = []
    total_runs = len(experiment_ids) * len(CONDITIONS)
    run_counter = 0
    start_time = time.time()

    for exp_id in experiment_ids:
        exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
        print(f"\n--- Experiment {exp_id}: {exp_name} ---")

        for cond in CONDITIONS:
            run_counter += 1
            train_path = SPLITS_DIR / cond["train_file"]
            eval_path = SPLITS_DIR / cond["eval_file"]

            _log(f"Run {run_counter}/{total_runs} | exp{exp_id} | {cond['label']}")
            _set_presplit_env(train_path, eval_path)

            t0 = time.time()
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

            elapsed = time.time() - t0
            f1_str = _fmt(metrics.get("f1"))
            _log(f"  F1={f1_str} ({elapsed:.1f}s)")

            rows.append({
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
                "condition": cond["key"],
                "condition_label": cond["label"],
                "train_file": cond["train_file"],
                "eval_file": cond["eval_file"],
                "f1": _to_float(metrics.get("f1")),
                "precision": _to_float(metrics.get("precision")),
                "recall": _to_float(metrics.get("recall")),
                "status": metrics.get("status"),
                "result_file": payload.get("result_file", ""),
                "metrics_file": payload.get("metrics_file", ""),
                "elapsed_seconds": round(elapsed, 1),
            })

    total_elapsed = time.time() - start_time
    results_df = pd.DataFrame(rows)

    # ── Delta analysis ────────────────────────────────────────────────
    delta_rows: list[dict] = []
    for exp_id in experiment_ids:
        baseline_row = results_df[
            (results_df["experiment_id"] == f"exp{exp_id}") &
            (results_df["condition"] == "exp08_baseline")
        ]
        augmented_row = results_df[
            (results_df["experiment_id"] == f"exp{exp_id}") &
            (results_df["condition"] == "exp08_augmented")
        ]
        if baseline_row.empty or augmented_row.empty:
            continue
        b = baseline_row.iloc[0]
        a = augmented_row.iloc[0]

        def _delta(metric: str):
            bv, av = _to_float(b.get(metric)), _to_float(a.get(metric))
            return (av - bv) if av is not None and bv is not None else None

        delta_rows.append({
            "experiment_id": f"exp{exp_id}",
            "experiment_name": EXP_NAMES.get(exp_id, f"Experiment {exp_id}"),
            "baseline_f1": _to_float(b.get("f1")),
            "augmented_f1": _to_float(a.get("f1")),
            "delta_f1": _delta("f1"),
            "baseline_precision": _to_float(b.get("precision")),
            "augmented_precision": _to_float(a.get("precision")),
            "delta_precision": _delta("precision"),
            "baseline_recall": _to_float(b.get("recall")),
            "augmented_recall": _to_float(a.get("recall")),
            "delta_recall": _delta("recall"),
        })

    deltas_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame()

    # ── Pivot table: experiment × condition ───────────────────────────
    pivot_rows: list[dict] = []
    for exp_id in experiment_ids:
        exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
        exp_results = results_df[results_df["experiment_id"] == f"exp{exp_id}"]
        pivot_row: dict[str, Any] = {
            "experiment_id": f"exp{exp_id}",
            "experiment_name": exp_name,
        }
        for _, r in exp_results.iterrows():
            pivot_row[f"{r['condition_label']} F1"] = r.get("f1")
        if not deltas_df.empty:
            exp_delta = deltas_df[deltas_df["experiment_id"] == f"exp{exp_id}"]
            if not exp_delta.empty:
                pivot_row["Delta F1"] = exp_delta.iloc[0].get("delta_f1")
        pivot_rows.append(pivot_row)
    pivot_df = pd.DataFrame(pivot_rows)

    # ── Save outputs ──────────────────────────────────────────────────
    out_dir = OUTPUTS_DIR / "exp08_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Excel
    excel_path = out_dir / f"exp08_comparison_{ts}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pivot_df.to_excel(writer, sheet_name="summary", index=False)
        results_df.to_excel(writer, sheet_name="all_runs", index=False)
        if not deltas_df.empty:
            deltas_df.to_excel(writer, sheet_name="deltas", index=False)

    # Latest copy
    latest_excel = out_dir / "exp08_comparison_latest.xlsx"
    try:
        if latest_excel.exists():
            latest_excel.unlink()
        import shutil
        shutil.copy2(excel_path, latest_excel)
    except PermissionError:
        pass

    # JSON
    result_payload = {
        "description": (
            "Runs experiments 03-06 with both baseline and augmented training "
            "data from experiment 08 (LLM mask-filling augmentation)."
        ),
        "exp08_meta": meta,
        "exp08_preparation": prep_info,
        "experiments": experiment_ids,
        "conditions": [c["label"] for c in CONDITIONS],
        "results": results_df.to_dict(orient="records"),
        "deltas": deltas_df.to_dict(orient="records") if not deltas_df.empty else [],
        "pivot": pivot_df.to_dict(orient="records"),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "excel_file": str(excel_path),
    }
    json_path = out_dir / f"exp08_comparison_{ts}.json"
    json_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    latest_json = out_dir / "exp08_comparison_latest.json"
    try:
        latest_json.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except PermissionError:
        pass

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    header = f"{'Experiment':<40s} {'Baseline F1':>12s} {'Augmented F1':>13s} {'Delta F1':>10s}"
    print(header)
    print("-" * len(header))
    for dr in delta_rows:
        print(
            f"{dr['experiment_name']:<40s} "
            f"{_fmt(dr['baseline_f1']):>12s} "
            f"{_fmt(dr['augmented_f1']):>13s} "
            f"{_fmt(dr['delta_f1']):>10s}"
        )
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print(f"Excel: {excel_path}")
    print(f"JSON: {json_path}")

    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run experiments 03-06 with baseline and augmented datasets from exp08. "
            "If split artifacts are missing, exp08 is run automatically."
        )
    )
    parser.add_argument(
        "--force-exp08",
        action="store_true",
        help="Force rerun experiment_08_llm_augmentation.py before running experiments 03-06.",
    )
    args = parser.parse_args()
    run(force_exp08_rerun=args.force_exp08)
