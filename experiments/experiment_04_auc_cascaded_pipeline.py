from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import shutil

import pandas as pd

from common import configure_model_environment, get_experiment_output_dir, is_debug_enabled, now_timestamp, write_result_json, write_split_runs_excel


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

    if "true_bio" in detailed_df.columns and "pred_bio" in detailed_df.columns:
        bio_df = detailed_df[["true_bio", "pred_bio"]].dropna()
        if not bio_df.empty:
            all_sheets["confusion_bio"] = pd.crosstab(bio_df["true_bio"], bio_df["pred_bio"]).reset_index()

    if "true_etype" in detailed_df.columns and "pred_etype" in detailed_df.columns:
        type_df = detailed_df[["true_etype", "pred_etype"]].dropna()
        type_df = type_df[type_df["true_etype"].astype(str) != "O"]
        if not type_df.empty:
            all_sheets["confusion_entity_type"] = pd.crosstab(type_df["true_etype"], type_df["pred_etype"]).reset_index()

    mismatch_filters = []
    if "true_bio" in detailed_df.columns and "pred_bio" in detailed_df.columns:
        mismatch_filters.append(detailed_df["true_bio"].astype(str) != detailed_df["pred_bio"].astype(str))
    if "true_etype" in detailed_df.columns and "pred_etype" in detailed_df.columns:
        mismatch_filters.append(detailed_df["true_etype"].astype(str) != detailed_df["pred_etype"].astype(str))
    if mismatch_filters:
        mismatch_mask = mismatch_filters[0]
        for extra_mask in mismatch_filters[1:]:
            mismatch_mask = mismatch_mask | extra_mask
        all_sheets["token_errors"] = detailed_df[mismatch_mask].copy()

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
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
    }
    if not debug:
        run_kwargs.update({
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        })

    completed = subprocess.run(
        [sys.executable, str(CORE_DIR / "auc_cascaded_pipeline.py")],
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
            "Cascaded pipeline failed. "
            f"Exit code: {completed.returncode}. "
            f"Last output lines:\n{details}"
        )

    metrics_path = CORE_DIR / "cascaded_pipeline_results.xlsx"
    f1 = _extract_final_f1(metrics_path)
    exp_dir = get_experiment_output_dir("exp04")
    timestamp = now_timestamp()
    archived_metrics_path = exp_dir / f"cascaded_pipeline_results_{timestamp}.xlsx"
    if metrics_path.exists():
        shutil.move(str(metrics_path), str(archived_metrics_path))
        _augment_cascaded_excel(archived_metrics_path)
    else:
        archived_metrics_path = metrics_path

    result = {
        "experiment_id": "exp04",
        "name": "AUC Cascaded Pipeline",
        "description": "Three-step cascaded NER with threshold tuning and span-level pipeline F1.",
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": {
            "model_name": model_name,
            "model_local_only": is_local_model,
            "train_fraction": 0.7,
            "validation_fraction": 0.3,
            "split_seed": split_seed,
            "split_strategy": "statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)",
            "config_source": "core/auc_cascaded_pipeline.py: TRAINING_CONFIG + LOSS_CONFIG",
        },
        "metrics_file": str(archived_metrics_path),
        "f1": f1,
        "status": "ok",
    }
    out_path = write_result_json("exp04", "auc_cascaded_pipeline", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    existing_seed = os.environ.get("THESIS_SPLIT_SEED")
    if existing_seed is not None:
        payload = run()
        print(
            f"[exp04] Pipeline Span F1={payload['f1']:.4f} | {payload['description']}"
            if payload["f1"] is not None
            else f"[exp04] F1=N/A | {payload['description']}"
        )
    else:
        num_runs = int((os.environ.get("THESIS_DIRECT_SPLIT_RUNS") or "5").strip() or "5")
        base_seed = int((os.environ.get("THESIS_DIRECT_BASE_SEED") or "42").strip() or "42")
        split_rows = []
        for run_idx in range(1, num_runs + 1):
            split_seed = base_seed + (run_idx - 1)
            os.environ["THESIS_SPLIT_SEED"] = str(split_seed)
            payload = run()
            split = payload.get("training_parameters", {})
            split_rows.append(
                {
                    "run_index": run_idx,
                    "split_seed": split_seed,
                    "f1": payload.get("f1"),
                    "train_fraction": split.get("train_fraction"),
                    "validation_fraction": split.get("validation_fraction"),
                    "split_strategy": split.get("split_strategy"),
                    "metrics_file": payload.get("metrics_file"),
                    "result_file": payload.get("result_file"),
                    "status": payload.get("status"),
                }
            )
            print(f"[exp04] run {run_idx}/{num_runs} seed={split_seed} F1={payload.get('f1')}")

        if existing_seed is None:
            os.environ.pop("THESIS_SPLIT_SEED", None)

        runs_df = pd.DataFrame(split_rows)
        ok_f1 = [float(v) for v in runs_df["f1"].dropna().tolist()]
        summary_df = pd.DataFrame(
            [
                {
                    "runs": num_runs,
                    "base_seed": base_seed,
                    "f1_mean": (sum(ok_f1) / len(ok_f1)) if ok_f1 else None,
                    "f1_best": max(ok_f1) if ok_f1 else None,
                    "f1_worst": min(ok_f1) if ok_f1 else None,
                }
            ]
        )
        out = write_split_runs_excel("exp04", "split_runs", runs_df, summary_df=summary_df)
        print(f"[exp04] Saved split summary: {out}")
