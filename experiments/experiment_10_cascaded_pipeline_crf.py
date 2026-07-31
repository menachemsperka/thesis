"""
experiment_10_cascaded_pipeline_crf.py — Cascaded NER + CRF head (Experiment 10_cascade)
========================================================================================

This is a **thin wrapper** around ``core/cascaded_crf_runtime.py``:

1. Sets ``THESIS_STEP3_BI_TYPE_RECONCILE=1`` so B/I entity types are reconciled after decode.
2. Runs the runtime as a subprocess (same pattern as ``experiment_04_auc_cascaded_pipeline.py``).
3. Moves ``cascaded_pipeline_crf_results.xlsx`` into ``outputs/exp10_cascade/``.
4. Calls ``cleanup_training_artifacts()`` to free disk space.

Students should study the **runtime** file for training loops; this file only handles IO and env.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import shutil

import pandas as pd

from common import configure_model_environment, get_experiment_output_dir, is_debug_enabled, now_timestamp, write_result_json
from model_cleanup import cleanup_training_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"


def _augment_cascaded_excel(excel_path: Path) -> None:
    if not excel_path.exists():
        return
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    detailed_df = all_sheets.get("detailed_results")
    if detailed_df is None or detailed_df.empty:
        return
    token_columns = [
        "eval_mode",
        "sentence_id",
        "token_idx",
        "token",
        "true_bio",
        "pred_bio",
        "true_etype",
        "pred_etype",
        "e_true",
        "e_pred",
        "b_true",
        "b_pred",
    ]
    present_token_columns = [column for column in token_columns if column in detailed_df.columns]
    if present_token_columns:
        all_sheets["token_level"] = detailed_df[present_token_columns].copy()
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in all_sheets.items():
            sheet_df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


def _extract_final_f1(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    df = pd.read_excel(metrics_path, sheet_name="metrics")
    final_rows = df[(df["epoch"].astype(str) == "final_optimised") & (df["eval_mode"] == "predicted")]
    if final_rows.empty:
        return None
    return float(final_rows.iloc[-1]["pipeline_span_f1"])


def run() -> dict:
    model_name, is_local_model = configure_model_environment()
    seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 42
    debug = is_debug_enabled()
    run_kwargs = {
        "cwd": str(CORE_DIR),
        "check": False,
        "env": {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "THESIS_STEP3_BI_TYPE_RECONCILE": "1",
        },
    }
    if not debug:
        run_kwargs.update(
            {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
        )

    completed = subprocess.run(
        [sys.executable, str(CORE_DIR / "cascaded_crf_runtime.py")],
        **run_kwargs,
    )
    if completed.returncode != 0:
        stderr_tail = ""
        stdout_tail = ""
        if isinstance(completed.stderr, str):
            stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-40:])
        if isinstance(completed.stdout, str):
            stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-40:])
        details = stderr_tail or stdout_tail or "No subprocess output captured."
        raise RuntimeError(
            f"Cascaded CRF pipeline failed. Exit code: {completed.returncode}. Last output:\n{details}"
        )

    metrics_path = CORE_DIR / "cascaded_pipeline_crf_results.xlsx"
    f1 = _extract_final_f1(metrics_path)
    exp_dir = get_experiment_output_dir("exp10_cascade")
    timestamp = now_timestamp()
    archived_metrics_path = exp_dir / f"cascaded_pipeline_crf_results_{timestamp}.xlsx"
    if metrics_path.exists():
        shutil.move(str(metrics_path), str(archived_metrics_path))
        _augment_cascaded_excel(archived_metrics_path)
    else:
        archived_metrics_path = metrics_path

    cleanup_training_artifacts()

    result = {
        "experiment_id": "exp10_cascade",
        "name": "Cascaded Pipeline with CRF + Step-3 Consistency",
        "description": (
            "Three-step cascaded NER with a joint full-tag CRF head (Viterbi decode) "
            "and B/I entity-type consistency reconciliation."
        ),
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": {
            "model_name": model_name,
            "model_local_only": is_local_model,
            "split_seed": split_seed,
            "crf_head": "full_bio_tag_linear_chain_crf",
            "step3_bi_consistency": True,
            "config_source": "core/cascaded_crf_runtime.py",
        },
        "metrics_file": str(archived_metrics_path),
        "f1": f1,
        "status": "ok",
    }
    out_path = write_result_json("exp10_cascade", "cascaded_pipeline_crf", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    payload = run()
    f1 = payload.get("f1")
    print(f"[exp10_cascade] F1={f1:.4f}" if f1 is not None else "[exp10_cascade] F1=N/A")
