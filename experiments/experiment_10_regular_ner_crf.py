"""
experiment_10_regular_ner_crf.py — Regular NER with BERT + CRF (Experiment 10_regular)
======================================================================================

Student reading order
---------------------
1. ``experiments/experiment_10_README.md`` — motivation and diagrams.
2. ``core/crf_layer.py`` — forward score, partition function, Viterbi.
3. ``core/bert_crf_training.py`` — model + ``CRFTrainer``.
4. This file ``run()`` — dataset → train → Excel/JSON → ``cleanup_training_artifacts()``.

Compared to Experiment 01
-------------------------
Exp01 uses ``PrepDataSetNERTraining.run_training_steps`` and a softmax head in
``AutoModelForTokenClassification``. This experiment keeps the **same splits and metrics** but swaps
the head for ``BertCRFForTokenClassification`` and Viterbi decoding.

Outputs
-------
* ``outputs/exp10_regular/regular_ner_crf_results_<timestamp>.xlsx``
* ``outputs/exp10_regular/latest.json`` — pointer for fusion experiments.
"""

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
from error_analysis import (
    annotate_error_types,
    build_error_analysis_sheets,
    model_display_name,
)
from model_cleanup import cleanup_training_artifacts


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import th_functions as tf  # type: ignore
from bert_crf_training import (  # type: ignore
    prepare_eval_results_crf,
    setup_bert_crf_token_classification,
    train_and_evaluate_bert_crf,
)
from NERtraining import PrepDataSetNERTraining  # type: ignore


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
    import torch

    model = trainer.model
    model.eval()
    rows = []
    for sentence_idx, item in enumerate(eval_ds, start=1):
        input_ids = torch.tensor([item["input_ids"]])
        attention_mask = torch.tensor([item.get("attention_mask", [1] * len(item["input_ids"]))])
        true_ids = item["labels"]
        tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emissions = outputs.logits[0].cpu().numpy()
            decoded = model.crf.viterbi_decode(outputs.logits, attention_mask)[0]
        token_id = 0
        for tok_idx, (token, true_id, pred_id) in enumerate(zip(tokens, true_ids, decoded)):
            true_label = _safe_label_name(int(true_id), label_list)
            pred_label = _safe_label_name(int(pred_id), label_list)
            if int(true_id) == -100 or true_label is None or pred_label is None or str(token).startswith("["):
                continue
            if tok_idx >= len(emissions):
                continue
            token_id += 1
            probs = _softmax(np.asarray(emissions[tok_idx], dtype=np.float64))
            sorted_probs = np.sort(probs)[::-1]
            rows.append(
                {
                    "sentence_id": sentence_idx,
                    "token_idx": token_id,
                    "token": str(token),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "prob": float(probs[int(pred_id)]),
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

    presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
    presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
    use_presplit = (
        presplit_train and presplit_eval and Path(presplit_train).exists() and Path(presplit_eval).exists()
    )

    if use_presplit:
        split_strategy = f"pre-split from experiment 07/08 ({Path(presplit_train).stem})"
    else:
        split_strategy = (
            "statistical stratified sentence split preserving non-O label distribution "
            "(with best-effort train coverage)"
        )

    training_parameters = {
        "model_name": model_name,
        "model_local_only": is_local_model,
        "train_fraction": 0.7,
        "validation_fraction": 0.3,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "framework": "BERT + LinearChainCRF (Viterbi decode)",
        "o_bias_init": 6.0,
    }

    local_only = is_local_model
    with suppress_output_if_needed():
        worker = PrepDataSetNERTraining()
        data = worker.load_and_prepare_data(str(dataset_path))
        print(f"Using dataset: {dataset_path}")

        if use_presplit:
            from split_io import load_split

            train_sentences = load_split(Path(presplit_train))
            eval_sentences = load_split(Path(presplit_eval))
            model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list = (
                setup_bert_crf_token_classification(
                    data,
                    train_sentences,
                    eval_sentences,
                    eval_sentences,
                    model_name,
                    local_files_only=local_only,
                )
            )
        else:
            sentences = tf.train_data_fit(data)
            train_data, test_data = tf.split_list(sentences, split_ratio=0.7)
            model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list = (
                setup_bert_crf_token_classification(
                    data,
                    train_data,
                    test_data,
                    test_data,
                    model_name,
                    local_files_only=local_only,
                )
            )

        trainer, eval_results = train_and_evaluate_bert_crf(
            model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"
        )
        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        df_sentences, _, _, _, global_metrics = prepare_eval_results_crf(ds_eval, trainer, processor, label_list)

    cleanup_training_artifacts()

    f1 = eval_results.get("eval_overall_f1")
    precision = eval_results.get("eval_overall_precision")
    recall = eval_results.get("eval_overall_recall")
    model_display = model_display_name(model_name)
    split_condition = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "model": model_display,
                "split_condition": split_condition,
                "seed": split_seed,
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

    if token_predictions_df is not None and not token_predictions_df.empty:
        token_predictions_df = annotate_error_types(token_predictions_df, "true_label", "pred_label")
        ea_sheets = build_error_analysis_sheets(
            token_predictions_df,
            experiment_name="Regular NER with BERT-CRF (Exp10)",
            model=model_display,
            split_condition=split_condition,
            seed=str(split_seed),
            true_col="true_label",
            pred_col="pred_label",
            sentence_col="sentence_id",
            token_idx_col="token_idx",
            token_col="token",
            confidence_col="prob",
            extra_sheet_docs=[
                ("token_predictions", "Per-token CRF/Viterbi predictions with emission-based confidence."),
                ("token_level", "Per-token true vs predicted label with is_correct flag."),
                ("classification_report", "Token-level (sklearn) per-label precision/recall/F1."),
                ("token_errors", "All misclassified tokens."),
                ("global_metrics", "Aggregate evaluation metrics reported by the trainer."),
            ],
        )
        merged_sheets = dict(ea_sheets)
        for name, sheet in extra_sheets.items():
            merged_sheets.setdefault(name, sheet)
        extra_sheets = merged_sheets
    extra_sheets["token_predictions"] = token_predictions_df

    metrics_file = write_result_excel(
        "exp10_regular",
        "regular_ner_crf_results",
        metrics_df,
        detailed_df,
        extra_sheets=extra_sheets,
    )

    result = {
        "experiment_id": "exp10_regular",
        "name": "Regular NER with BERT-CRF",
        "description": "Token classification with a CRF layer and Viterbi decoding (Souza et al. 2019; Lample et al. 2016).",
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
    out_path = write_result_json("exp10_regular", "regular_ner_crf", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    payload = run()
    f1 = payload.get("f1")
    print(f"[exp10_regular] F1={f1:.4f}" if f1 is not None else "[exp10_regular] F1=N/A")
