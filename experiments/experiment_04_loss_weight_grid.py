"""
Validation-grid search for cascaded loss weights (lambda_bio, lambda_type).

Runs Exp04 training once per grid cell on the same train/validation split,
selects the pair that maximizes validation pipeline span F1 (predicted mode,
after threshold tuning). The test set is not used.

Default grid: lambda_bio, lambda_type in {1, 5, 10} (9 configurations).

Usage (single split seed for selection):
    set THESIS_SPLIT_SEED=42
    python experiments/experiment_04_loss_weight_grid.py

Optional:
    set THESIS_EXP04_LAMBDA_GRID_VALUES=1,5,10
    set THESIS_EXP04_FAST=1
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from common import configure_model_environment, get_experiment_output_dir, now_timestamp, write_result_json
from experiment_04_auc_cascaded_pipeline import run as run_exp04


EXPERIMENT_ID = "exp04_lambda_grid"


def _parse_lambda_grid_values() -> list[float]:
    raw = (os.environ.get("THESIS_EXP04_LAMBDA_GRID_VALUES") or "1,5,10").strip()
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values or [1.0, 5.0, 10.0]


def _select_best(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r.get("validation_pipeline_span_f1") is not None]
    if not ok:
        return None

    def sort_key(r: dict) -> tuple:
        f1 = float(r["validation_pipeline_span_f1"])
        bio = float(r["lambda_bio"])
        typ = float(r["lambda_type"])
        return (f1, -bio, -typ)

    return max(ok, key=sort_key)


def _load_progress(cache_dir: Path | None) -> dict[str, dict]:
    if cache_dir is None:
        return {}
    path = cache_dir / "exp04_lambda_grid_progress.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        done = raw.get("completed_pairs") if isinstance(raw, dict) else None
        if isinstance(done, dict):
            return done
    except Exception:
        pass
    return {}


def _save_progress(cache_dir: Path | None, completed: dict[str, dict]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "exp04_lambda_grid_progress.json"
    path.write_text(
        json.dumps({"completed_pairs": completed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_grid(
    *,
    cache_dir: Path | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
) -> dict:
    model_name, is_local_model = configure_model_environment()
    split_seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(split_seed_raw)
    except ValueError:
        split_seed = 42
    os.environ["THESIS_SPLIT_SEED"] = str(split_seed)

    grid_values = _parse_lambda_grid_values()
    completed = {} if force_rebuild else (_load_progress(cache_dir) if resume else {})
    grid_rows: list[dict] = list(completed.values()) if completed else []

    total = len(grid_values) ** 2
    print(
        f"[{EXPERIMENT_ID}] Validation loss-weight grid: "
        f"{grid_values} x {grid_values} = {total} runs | split_seed={split_seed} | model={model_name}"
        + (f" | cache={cache_dir}" if cache_dir else "")
    )
    if resume and completed:
        print(f"[{EXPERIMENT_ID}] Resuming grid: {len(completed)}/{total} cells already done.")

    for lambda_bio in grid_values:
        for lambda_type in grid_values:
            key = f"{float(lambda_bio):g}_{float(lambda_type):g}"
            if resume and key in completed:
                print(f"[{EXPERIMENT_ID}] Skip cached λ_bio={lambda_bio} λ_type={lambda_type}")
                continue
            os.environ["THESIS_EXP04_LAMBDA_BIO"] = str(lambda_bio)
            os.environ["THESIS_EXP04_LAMBDA_TYPE"] = str(lambda_type)
            print(f"[{EXPERIMENT_ID}] Training lambda_bio={lambda_bio} lambda_type={lambda_type} …")
            payload = run_exp04()
            f1 = payload.get("f1")
            row = {
                "split_seed": split_seed,
                "lambda_bio": lambda_bio,
                "lambda_type": lambda_type,
                "validation_pipeline_span_f1": f1,
                "model": model_name,
                "metrics_file": payload.get("metrics_file"),
                "result_file": payload.get("result_file"),
                "status": payload.get("status"),
            }
            grid_rows.append(row)
            completed[key] = row
            _save_progress(cache_dir, completed)
            print(
                f"[{EXPERIMENT_ID}]   validation pipeline span F1={f1}"
                if f1 is not None
                else f"[{EXPERIMENT_ID}]   F1=N/A"
            )

    runs_df = pd.DataFrame(grid_rows)
    pivot = runs_df.pivot(
        index="lambda_bio",
        columns="lambda_type",
        values="validation_pipeline_span_f1",
    )
    best = _select_best(grid_rows)
    selection_metric = "validation pipeline span F1 (predicted, final_optimised, after threshold sweep)"

    summary = {
        "split_seed": split_seed,
        "model": model_name,
        "model_local": is_local_model,
        "grid_values": grid_values,
        "selection_metric": selection_metric,
        "selected_lambda_bio": best["lambda_bio"] if best else None,
        "selected_lambda_type": best["lambda_type"] if best else None,
        "selected_validation_pipeline_span_f1": best["validation_pipeline_span_f1"] if best else None,
        "baseline_10_5_f1": None,
    }
    baseline_rows = runs_df[
        (runs_df["lambda_bio"] == 10.0) & (runs_df["lambda_type"] == 5.0)
    ]
    if not baseline_rows.empty and baseline_rows.iloc[0]["validation_pipeline_span_f1"] is not None:
        summary["baseline_10_5_f1"] = float(baseline_rows.iloc[0]["validation_pipeline_span_f1"])

    exp_dir = cache_dir if cache_dir is not None else get_experiment_output_dir(EXPERIMENT_ID)
    exp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_timestamp()
    excel_path = exp_dir / f"loss_weight_grid_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="grid_runs", index=False)
        pivot.to_excel(writer, sheet_name="heatmap_f1")
        pd.DataFrame([summary]).to_excel(writer, sheet_name="selection", index=False)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "name": "Exp04 validation loss-weight grid",
        "description": (
            "Coarse validation grid for lambda_bio and lambda_type in the cascaded pipeline. "
            "Selects weights maximizing validation pipeline span F1."
        ),
        "model": model_name,
        "training_parameters": {
            "split_seed": split_seed,
            "train_fraction": 0.7,
            "validation_fraction": 0.3,
            "lambda_grid_values": grid_values,
            "selection_metric": selection_metric,
        },
        "selected_lambda_bio": summary["selected_lambda_bio"],
        "selected_lambda_type": summary["selected_lambda_type"],
        "selected_validation_pipeline_span_f1": summary["selected_validation_pipeline_span_f1"],
        "baseline_lambda_bio": 10.0,
        "baseline_lambda_type": 5.0,
        "baseline_10_5_f1": summary["baseline_10_5_f1"],
        "grid_excel": str(excel_path),
        "grid_runs": grid_rows,
        "status": "ok" if best else "no_valid_runs",
    }
    out_path = write_result_json(EXPERIMENT_ID, "loss_weight_grid", result)
    result["result_file"] = str(out_path)

    if best:
        print(
            f"[{EXPERIMENT_ID}] Selected lambda_bio={best['lambda_bio']} "
            f"lambda_type={best['lambda_type']} "
            f"(validation pipeline span F1={best['validation_pipeline_span_f1']})"
        )
        print(
            f"[{EXPERIMENT_ID}] Set for downstream Exp04 runs:\n"
            f"  set THESIS_EXP04_LAMBDA_BIO={best['lambda_bio']}\n"
            f"  set THESIS_EXP04_LAMBDA_TYPE={best['lambda_type']}"
        )
    print(f"[{EXPERIMENT_ID}] Grid results: {excel_path}")

    return result


if __name__ == "__main__":
    run_grid()
