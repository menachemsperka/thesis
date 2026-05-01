from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from common import (
    configure_model_environment,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
    write_split_runs_excel,
)


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from NERtraining import PrepDataSetNERTraining, prepare_eval_results  # type: ignore


def _build_token_level_df(detailed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sentence_idx, row in detailed_df.reset_index(drop=True).iterrows():
        tokens = str(row.get("sentence", "")).split()
        true_labels = str(row.get("true_labels", "")).split()
        pred_labels = str(row.get("predicted_labels", "")).split()
        for token_idx, (token, true_label, pred_label) in enumerate(
            zip(tokens, true_labels, pred_labels),
            start=1,
        ):
            rows.append(
                {
                    "dataset_name": row.get("dataset_name"),
                    "sentence_id": sentence_idx + 1,
                    "token_id": token_idx,
                    "token": token,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "is_correct": true_label == pred_label,
                }
            )
    return pd.DataFrame(rows)


def _build_extra_sheets(token_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if token_df.empty:
        return {}

    confusion_df = pd.crosstab(token_df["true_label"], token_df["predicted_label"]).reset_index()
    report_dict = classification_report(
        token_df["true_label"],
        token_df["predicted_label"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    mismatches_df = token_df[token_df["is_correct"] == False].copy()

    per_dataset_accuracy = (
        token_df.groupby("dataset_name", as_index=False)["is_correct"].mean().rename(columns={"is_correct": "token_accuracy"})
    )

    return {
        "token_level": token_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
        "token_errors": mismatches_df,
        "per_dataset_accuracy": per_dataset_accuracy,
    }


def _single_run(csv_name: str) -> dict:
    dataset_path = resolve_dataset(csv_name)
    with suppress_output_if_needed():
        worker = PrepDataSetNERTraining()
        data = worker.load_and_prepare_data(str(dataset_path))
        trainer, eval_results, label_list, ds_eval = worker.run_training_steps(data)
        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        df_sentences, _, _, _, _ = prepare_eval_results(ds_eval, trainer, processor, label_list)

    detailed_df = df_sentences.rename(
        columns={
            "Sentence": "sentence",
            "True Labels": "true_labels",
            "Predicted Labels": "predicted_labels",
        }
    )
    detailed_df.insert(0, "dataset_name", csv_name)

    return {
        "dataset_name": csv_name,
        "dataset_path": str(dataset_path),
        "f1": eval_results.get("eval_overall_f1"),
        "precision": eval_results.get("eval_overall_precision"),
        "recall": eval_results.get("eval_overall_recall"),
        "detailed_df": detailed_df,
    }


def run() -> dict:
    model_name, is_local_model = configure_model_environment()
    seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 42
    training_parameters = {
        "model_name": model_name,
        "model_local_only": is_local_model,
        "train_fraction": 0.7,
        "validation_fraction": 0.3,
        "split_seed": split_seed,
        "split_strategy": "statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)",
        "variants": ["ner_dataset.csv", "ner_training_generated.csv", "ner_training_duplicated.csv"],
        "framework": "transformers.Trainer (default training args)",
    }
    variants = [
        "ner_dataset.csv",
        "ner_training_generated.csv",
        "ner_training_duplicated.csv",
    ]
    runs = []
    all_detailed_frames = []
    for variant in variants:
        run_result = _single_run(variant)
        for metric in ("f1", "precision", "recall"):
            value = run_result.get(metric)
            run_result[metric] = float(value) if value is not None else None
        detailed_df = run_result.pop("detailed_df")
        all_detailed_frames.append(detailed_df)
        runs.append(run_result)

    best = max((r for r in runs if r["f1"] is not None), key=lambda r: r["f1"], default=None)
    metrics_df = pd.DataFrame(runs)
    detailed_df = pd.concat(all_detailed_frames, ignore_index=True) if all_detailed_frames else pd.DataFrame()
    token_df = _build_token_level_df(detailed_df)
    extra_sheets = _build_extra_sheets(token_df)
    metrics_file = write_result_excel(
        "exp02",
        "imbalance_llm_duplication_results",
        metrics_df,
        detailed_df,
        extra_sheets=extra_sheets,
    )

    result = {
        "experiment_id": "exp02",
        "name": "Imbalance Handling + LLM Generation + Duplication",
        "description": "Compares baseline, generated-sentence, and duplicated-sentence training datasets.",
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": training_parameters,
        "metrics_file": str(metrics_file),
        "runs": runs,
        "best_variant": best["dataset_name"] if best else None,
        "f1": best["f1"] if best else None,
        "status": "ok",
    }
    out_path = write_result_json("exp02", "imbalance_llm_duplication", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    existing_seed = os.environ.get("THESIS_SPLIT_SEED")
    if existing_seed is not None:
        payload = run()
        if payload["f1"] is not None:
            print(
                f"[exp02] Best F1={payload['f1']:.4f} ({payload['best_variant']}) | "
                f"{payload['description']}"
            )
        else:
            print(f"[exp02] F1=N/A | {payload['description']}")
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
                    "best_variant": payload.get("best_variant"),
                    "train_fraction": split.get("train_fraction"),
                    "validation_fraction": split.get("validation_fraction"),
                    "split_strategy": split.get("split_strategy"),
                    "metrics_file": payload.get("metrics_file"),
                    "result_file": payload.get("result_file"),
                    "status": payload.get("status"),
                }
            )
            print(f"[exp02] run {run_idx}/{num_runs} seed={split_seed} best_F1={payload.get('f1')}")

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
        out = write_split_runs_excel("exp02", "split_runs", runs_df, summary_df=summary_df)
        print(f"[exp02] Saved split summary: {out}")
