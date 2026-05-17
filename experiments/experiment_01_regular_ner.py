from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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
                    "sentence_id": sentence_idx + 1,
                    "token_id": token_idx,
                    "token": token,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "is_correct": true_label == pred_label,
                }
            )
    return pd.DataFrame(rows)


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits)
    exp_vals = np.exp(stable)
    return exp_vals / np.sum(exp_vals)


def _safe_label_name(label_idx: int, label_list: list[str]) -> str | None:
    if label_idx == -100:
        return None
    if 0 <= int(label_idx) < len(label_list):
        return label_list[int(label_idx)]
    return None


def _build_token_predictions(eval_ds, trainer, tokenizer, label_list: list[str]) -> pd.DataFrame:
    """Build per-token predictions with probabilities, entropy, and margin for downstream fusion."""
    preds, _, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds, axis=2)

    rows = []
    for sentence_idx, item in enumerate(eval_ds, start=1):
        input_ids = item["input_ids"]
        true_ids = item["labels"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_id = 0
        for token, true_id, pred_id, token_logits in zip(
            tokens, true_ids, pred_ids[sentence_idx - 1], preds[sentence_idx - 1]
        ):
            true_label = _safe_label_name(int(true_id), label_list)
            pred_label = _safe_label_name(int(pred_id), label_list)
            if int(true_id) == -100 or true_label is None or pred_label is None or str(token).startswith("["):
                continue

            token_id += 1
            probs = _softmax(np.asarray(token_logits, dtype=np.float64))
            sorted_probs = np.sort(probs)[::-1]

            rows.append(
                {
                    "sentence_id": sentence_idx,
                    "token_idx": token_id,
                    "token": str(token),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "prob": float(np.max(probs)),
                    "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
                    "margin": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0,
                }
            )

    return pd.DataFrame(rows)


def _build_extra_sheets(token_df: pd.DataFrame, global_metrics: dict | None) -> dict[str, pd.DataFrame]:
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

    extra = {
        "token_level": token_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
        "token_errors": mismatches_df,
    }
    if global_metrics:
        extra["global_metrics"] = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in global_metrics.items()]
        )
    return extra


def run() -> dict:
    dataset_override = (os.environ.get("THESIS_NER_CSV") or "").strip()
    dataset_path = Path(dataset_override) if dataset_override else resolve_dataset("ner_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    model_name, is_local_model = configure_model_environment()
    seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 42

    # Support pre-computed splits from experiment 07 / 08
    presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
    presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
    use_presplit = (
        presplit_train and presplit_eval
        and Path(presplit_train).exists() and Path(presplit_eval).exists()
    )

    if use_presplit:
        split_strategy = f"pre-split from experiment 07/08 ({Path(presplit_train).stem})"
    else:
        split_strategy = "statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)"

    training_parameters = {
        "model_name": model_name,
        "model_local_only": is_local_model,
        "train_fraction": 0.7,
        "validation_fraction": 0.3,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "framework": "transformers.Trainer (default training args)",
    }
    with suppress_output_if_needed():
        worker = PrepDataSetNERTraining()
        data = worker.load_and_prepare_data(str(dataset_path))
        print(f"Using dataset: {dataset_path}")
        print(f"Dataset exists: {dataset_path.exists()}")

        # Load the dataset and count sentences
        sentence_count = len(data['id'].unique()) if 'id' in data.columns else 'Unknown'
        print(f"Loaded dataset with {sentence_count} unique sentences.")

        if use_presplit:
            from split_io import load_split
            train_sentences = load_split(Path(presplit_train))
            eval_sentences = load_split(Path(presplit_eval))
            trainer, eval_results, label_list, ds_eval = worker.run_training_with_presplit(
                data, train_sentences, eval_sentences
            )
        else:
            trainer, eval_results, label_list, ds_eval = worker.run_training_steps(data)
        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        df_sentences, _, _, _, global_metrics = prepare_eval_results(ds_eval, trainer, processor, label_list)

    f1 = eval_results.get("eval_overall_f1")
    precision = eval_results.get("eval_overall_precision")
    recall = eval_results.get("eval_overall_recall")
    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "f1": float(f1) if f1 is not None else None,
                "precision": float(precision) if precision is not None else None,
                "recall": float(recall) if recall is not None else None,
            }
        ]
    )
    detailed_df = df_sentences.rename(
        columns={
            "Sentence": "sentence",
            "True Labels": "true_labels",
            "Predicted Labels": "predicted_labels",
        }
    )
    token_df = _build_token_level_df(detailed_df)
    token_predictions_df = _build_token_predictions(ds_eval, trainer, processor, label_list)
    extra_sheets = _build_extra_sheets(token_df, global_metrics)
    extra_sheets["token_predictions"] = token_predictions_df
    metrics_file = write_result_excel(
        "exp01",
        "regular_ner_results",
        metrics_df,
        detailed_df,
        extra_sheets=extra_sheets,
    )

    result = {
        "experiment_id": "exp01",
        "name": "Regular NER with DictaBERT",
        "description": "Baseline token classification using DictaBERT on the original dataset.",
        "dataset": str(dataset_path),
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": training_parameters,
        "metrics_file": str(metrics_file),
        "f1": float(f1) if f1 is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "status": "ok",
    }
    out_path = write_result_json("exp01", "regular_ner", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    existing_seed = os.environ.get("THESIS_SPLIT_SEED")
    if existing_seed is not None:
        payload = run()
        print(
            f"[exp01] F1={payload['f1']:.4f} | {payload['description']}"
            if payload["f1"] is not None
            else f"[exp01] F1=N/A | {payload['description']}"
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
                    "precision": payload.get("precision"),
                    "recall": payload.get("recall"),
                    "train_fraction": split.get("train_fraction"),
                    "validation_fraction": split.get("validation_fraction"),
                    "split_strategy": split.get("split_strategy"),
                    "metrics_file": payload.get("metrics_file"),
                    "result_file": payload.get("result_file"),
                    "status": payload.get("status"),
                }
            )
            print(f"[exp01] run {run_idx}/{num_runs} seed={split_seed} F1={payload.get('f1')}")

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
        out = write_split_runs_excel("exp01", "split_runs", runs_df, summary_df=summary_df)
        print(f"[exp01] Saved split summary: {out}")
