from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG = False
RUNS_PER_EXPERIMENT = 5
BASE_SPLIT_SEED = 42


EXPERIMENTS = [
    {
        "id": "exp01",
        "title": "1) Regular NER with DictaBERT",
        "description": "Baseline token classification using DictaBERT.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_01_regular_ner.py",
    },
    {
        "id": "exp02",
        "title": "2) Imbalance + LLM Generation + Duplication",
        "description": "Compares original, generated, and duplicated training sets.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_02_imbalance_llm_duplication.py",
    },
    {
        "id": "exp03",
        "title": "3) AUC-2T",
        "description": "Single AUC-2T run on full training split.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_03_auc_2t.py",
    },
    {
        "id": "exp04",
        "title": "4) AUC Cascaded Pipeline",
        "description": "Three-step cascaded pipeline with span-level evaluation.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_04_auc_cascaded_pipeline.py",
    },
    {
        "id": "exp05",
        "title": "5) AUC Cascaded Pipeline + Step3 B/I Type Consistency",
        "description": "Experiment 4 with Step 3 rule: adjacent B and I tokens must share the entity type chosen by higher probability.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_05_auc_cascaded_pipeline_step3_consistency.py",
    },
    {
        "id": "exp06",
        "title": "6) Fusion: Regular NER + AUC Cascaded Pipeline",
        "description": "Fuses experiment 1 and experiment 4 predictions; when they disagree, selects the label with the higher probability.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_06_fusion_regular_and_cascaded.py",
    },
    {
        "id": "exp08",
        "title": "8) LLM Mask-Filling Augmentation for Rare Labels",
        "description": "Augments training data by generating new sentences for rare entity types using DictaBERT fill-mask, then compares baseline vs augmented NER performance.",
        "split": "Train 70% / Validation 30%",
        "script": "experiment_08_llm_augmentation.py",
    },
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_latest_payload(exp_id: str) -> dict | None:
    exp_dir = OUTPUTS_DIR / exp_id
    latest_path = exp_dir / "latest.json"
    payload = _read_json(latest_path)
    if payload:
        return payload

    if not exp_dir.exists():
        return None

    candidates = sorted(
        [p for p in exp_dir.glob("*.json") if p.name != "latest.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        payload = _read_json(candidate)
        if payload:
            return payload
    return None


def _extract_f1_from_text(text: str | None):
    if not text:
        return None
    match = re.search(r"F1\s*=\s*([0-9]*\.?[0-9]+)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_f1(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


def _format_training_params(params) -> str:
    if not isinstance(params, dict) or not params:
        return "N/A"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _write_split_summary_excel(summary: list[dict], output_path: Path) -> None:
    split_rows = []
    experiment_rows = []

    for exp in summary:
        experiment_rows.append(
            {
                "experiment_id": exp.get("id"),
                "title": exp.get("title"),
                "status": exp.get("status"),
                "split_method": exp.get("split"),
                "runs_per_experiment": exp.get("runs_per_experiment"),
                "f1_mean": _to_float(exp.get("f1_mean")),
                "f1_best": _to_float(exp.get("f1_best")),
                "f1_worst": _to_float(exp.get("f1_worst")),
                "error": exp.get("error"),
            }
        )

        for run in exp.get("runs", []):
            split_rows.append(
                {
                    "experiment_id": exp.get("id"),
                    "experiment_title": exp.get("title"),
                    "run_index": run.get("run_index"),
                    "split_seed": run.get("split_seed"),
                    "status": run.get("status"),
                    "f1": _to_float(run.get("f1")),
                    "precision": _to_float(run.get("precision")),
                    "recall": _to_float(run.get("recall")),
                    "split_train_fraction": run.get("split_train_fraction"),
                    "split_validation_fraction": run.get("split_validation_fraction"),
                    "split_train_sentences": run.get("split_train_sentences"),
                    "split_validation_sentences": run.get("split_validation_sentences"),
                    "metrics_file": run.get("metrics_file"),
                    "result_file": run.get("result_file"),
                    "error": run.get("error"),
                }
            )

    split_df = pd.DataFrame(split_rows)
    exp_df = pd.DataFrame(experiment_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        split_df.to_excel(writer, sheet_name="per_split", index=False)
        exp_df.to_excel(writer, sheet_name="per_experiment", index=False)


def _write_per_experiment_split_excels(summary: list[dict]) -> list[Path]:
    created_paths: list[Path] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp in summary:
        exp_id = exp.get("id")
        if not exp_id:
            continue
        exp_dir = OUTPUTS_DIR / str(exp_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for run in exp.get("runs", []):
            rows.append(
                {
                    "experiment_id": exp_id,
                    "experiment_title": exp.get("title"),
                    "run_index": run.get("run_index"),
                    "split_seed_requested": run.get("split_seed"),
                    "split_seed_used": run.get("split_seed_used"),
                    "seed_match": bool(run.get("split_seed") == run.get("split_seed_used")) if run.get("split_seed_used") is not None else None,
                    "status": run.get("status"),
                    "f1": _to_float(run.get("f1")),
                    "precision": _to_float(run.get("precision")),
                    "recall": _to_float(run.get("recall")),
                    "split_train_fraction": run.get("split_train_fraction"),
                    "split_validation_fraction": run.get("split_validation_fraction"),
                    "split_train_sentences": run.get("split_train_sentences"),
                    "split_validation_sentences": run.get("split_validation_sentences"),
                    "metrics_file": run.get("metrics_file"),
                    "result_file": run.get("result_file"),
                    "error": run.get("error"),
                }
            )

        per_split_df = pd.DataFrame(rows)
        agg_row = {
            "experiment_id": exp_id,
            "experiment_title": exp.get("title"),
            "runs_per_experiment": exp.get("runs_per_experiment"),
            "f1_mean": _to_float(exp.get("f1_mean")),
            "f1_best": _to_float(exp.get("f1_best")),
            "f1_worst": _to_float(exp.get("f1_worst")),
            "status": exp.get("status"),
            "split_method": exp.get("split"),
            "error": exp.get("error"),
        }
        agg_df = pd.DataFrame([agg_row])

        timestamped = exp_dir / f"split_runs_{stamp}.xlsx"
        latest = exp_dir / "split_runs_latest.xlsx"

        with pd.ExcelWriter(timestamped, engine="openpyxl") as writer:
            per_split_df.to_excel(writer, sheet_name="per_split", index=False)
            agg_df.to_excel(writer, sheet_name="summary", index=False)

        with pd.ExcelWriter(latest, engine="openpyxl") as writer:
            per_split_df.to_excel(writer, sheet_name="per_split", index=False)
            agg_df.to_excel(writer, sheet_name="summary", index=False)

        created_paths.extend([timestamped, latest])

    return created_paths


def run() -> None:
    debug = DEBUG or os.environ.get("THESIS_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    from experiments.common import configure_network_environment
    configure_network_environment()

    print("=" * 78)
    print("Thesis Experiments Runner")
    print(f"Runs each experiment {RUNS_PER_EXPERIMENT} times with statistical stratified sentence splits and summarizes F1")
    print("=" * 78)

    for exp in EXPERIMENTS:
        script_path = EXPERIMENTS_DIR / exp["script"]

        print(f"\n--- Running {exp['title']} ---", flush=True)
        print(f"Description: {exp['description']}", flush=True)
        print(f"Split: {exp['split']}", flush=True)

        run_rows = []
        for run_idx in range(1, RUNS_PER_EXPERIMENT + 1):
            split_seed = BASE_SPLIT_SEED + (run_idx - 1)
            return_code = 0
            error_text = None
            completed = None

            print(f"  Run {run_idx}/{RUNS_PER_EXPERIMENT} (split_seed={split_seed})", flush=True)
            try:
                run_kwargs = {
                    "cwd": str(EXPERIMENTS_DIR),
                    "check": True,
                    "env": {
                        **os.environ,
                        "THESIS_DEBUG": "1" if debug else "0",
                        "THESIS_SPLIT_SEED": str(split_seed),
                        "THESIS_SPLIT_RUN_INDEX": str(run_idx),
                    },
                }
                if not debug:
                    run_kwargs.update({"capture_output": True, "text": True})
                completed = subprocess.run([sys.executable, str(script_path)], **run_kwargs)
            except subprocess.CalledProcessError as exc:
                return_code = exc.returncode
                error_text = f"Process failed with code {exc.returncode}"
                completed = exc
            except Exception as exc:
                return_code = 1
                error_text = str(exc)

            payload = _read_latest_payload(exp["id"])
            f1 = payload.get("f1") if payload else None
            training_parameters = payload.get("training_parameters") if payload else None
            payload_split_seed = None
            if isinstance(training_parameters, dict):
                payload_split_seed = training_parameters.get("split_seed")
            if f1 is None:
                f1 = _extract_f1_from_text(getattr(completed, "stdout", None))
            if f1 is None:
                f1 = _extract_f1_from_text(getattr(completed, "stderr", None))

            run_rows.append(
                {
                    "run_index": run_idx,
                    "split_seed": split_seed,
                    "split_seed_used": payload_split_seed,
                    "f1": f1,
                    "precision": payload.get("precision") if payload else None,
                    "recall": payload.get("recall") if payload else None,
                    "split_train_fraction": (payload.get("split") or {}).get("train_fraction") if payload else None,
                    "split_validation_fraction": (payload.get("split") or {}).get("validation_fraction") if payload else None,
                    "split_train_sentences": (payload.get("split") or {}).get("train_sentences") if payload else None,
                    "split_validation_sentences": (payload.get("split") or {}).get("validation_sentences") if payload else None,
                    "metrics_file": payload.get("metrics_file") if payload else None,
                    "result_file": payload.get("result_file") if payload else None,
                    "status": "ok" if return_code == 0 else "failed",
                    "error": error_text,
                    "training_parameters": training_parameters,
                }
            )

            if return_code == 0:
                print(f"    F1: {_format_f1(f1)}", flush=True)
                if payload_split_seed is not None and str(payload_split_seed) != str(split_seed):
                    print(
                        f"    Warning: requested split seed {split_seed}, but payload reported {payload_split_seed}",
                        flush=True,
                    )
            else:
                print(f"    Failed: {error_text}", flush=True)

        ok_f1 = [row["f1"] for row in run_rows if row["status"] == "ok" and row["f1"] is not None]
        mean_f1 = (sum(ok_f1) / len(ok_f1)) if ok_f1 else None
        best_f1 = max(ok_f1) if ok_f1 else None
        worst_f1 = min(ok_f1) if ok_f1 else None
        status = "ok" if all(row["status"] == "ok" for row in run_rows) else "failed"
        training_parameters = next((row.get("training_parameters") for row in run_rows if row.get("training_parameters")), None)

        print(f"Completed {exp['title']} runs", flush=True)
        print(f"Mean F1: {_format_f1(mean_f1)} | Best: {_format_f1(best_f1)} | Worst: {_format_f1(worst_f1)}", flush=True)
        print(f"Training params: {_format_training_params(training_parameters)}", flush=True)

        summary.append(
            {
                "id": exp["id"],
                "title": exp["title"],
                "description": exp["description"],
                "split": "Statistical stratified sentence split preserving non-O label distribution (train=70%, val=30%), repeated for 5 seeds",
                "runs_per_experiment": RUNS_PER_EXPERIMENT,
                "split_seeds": [row["split_seed"] for row in run_rows],
                "runs": run_rows,
                "training_parameters": training_parameters,
                "f1_mean": mean_f1,
                "f1_best": best_f1,
                "f1_worst": worst_f1,
                "status": status,
                "error": None if status == "ok" else "One or more runs failed",
            }
        )

    print("\n" + "=" * 78)
    print("F1 Results Summary")
    print("=" * 78)
    for row in summary:
        status = "✅" if row["status"] == "ok" else "❌"
        print(f"{status} {row['title']}")
        print(f"   {row['description']}")
        print(f"   Split: {row['split']}")
        print(f"   Training params: {_format_training_params(row.get('training_parameters'))}")
        print(f"   F1 mean: {_format_f1(row.get('f1_mean'))} | best: {_format_f1(row.get('f1_best'))} | worst: {_format_f1(row.get('f1_worst'))}")
        if row["error"]:
            print(f"   Note: {row['error']}")

    summary_path = OUTPUTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    split_excel_path = OUTPUTS_DIR / "summary_splits.xlsx"
    _write_split_summary_excel(summary, split_excel_path)
    per_exp_excel_paths = _write_per_experiment_split_excels(summary)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved split-level Excel to: {split_excel_path}")
    if per_exp_excel_paths:
        print("Saved per-experiment split Excel files:")
        for path in per_exp_excel_paths:
            print(f" - {path}")


if __name__ == "__main__":
    run()
