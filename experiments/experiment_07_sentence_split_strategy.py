from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from common import resolve_dataset, write_result_excel, write_result_json


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import th_functions as tf  # type: ignore
from NERtraining import PrepDataSetNERTraining  # type: ignore


def _resolve_seed(default_seed: int = 42) -> int:
    raw = (os.environ.get("THESIS_SPLIT_SEED") or str(default_seed)).strip()
    try:
        return int(raw)
    except ValueError:
        return default_seed


def _non_o_label_counts(sentences: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        labels = sentence.get("labels", []) if isinstance(sentence, dict) else []
        for label in labels:
            key = str(label)
            if key == "O":
                continue
            counts[key] = counts.get(key, 0) + 1
    return counts


def _non_o_labels_in_sentence(sentence: dict) -> set[str]:
    labels = sentence.get("labels", []) if isinstance(sentence, dict) else []
    return {str(label) for label in labels if str(label) != "O"}


def _sentence_presence_counts(sentences: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        for label in _non_o_labels_in_sentence(sentence):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _label_distribution_report(train_sentences: list[dict], eval_sentences: list[dict]) -> pd.DataFrame:
    train_token_counts = _non_o_label_counts(train_sentences)
    eval_token_counts = _non_o_label_counts(eval_sentences)
    full_token_counts = _non_o_label_counts(train_sentences + eval_sentences)

    train_sentence_counts = _sentence_presence_counts(train_sentences)
    eval_sentence_counts = _sentence_presence_counts(eval_sentences)
    full_sentence_counts = _sentence_presence_counts(train_sentences + eval_sentences)

    labels = sorted(full_token_counts.keys())
    if not labels:
        return pd.DataFrame(
            [
                {
                    "label": "<none>",
                    "full_token_count": 0,
                    "train_token_count": 0,
                    "eval_token_count": 0,
                    "train_token_share_of_full": None,
                    "eval_token_share_of_full": None,
                    "full_sentence_count": 0,
                    "train_sentence_count": 0,
                    "eval_sentence_count": 0,
                    "in_train": False,
                    "in_eval": False,
                    "rare_label_q1": False,
                }
            ]
        )

    full_counts_series = pd.Series([full_token_counts[label] for label in labels])
    q1_threshold = float(full_counts_series.quantile(0.25))

    rows = []
    for label in labels:
        full_tokens = int(full_token_counts.get(label, 0))
        train_tokens = int(train_token_counts.get(label, 0))
        eval_tokens = int(eval_token_counts.get(label, 0))

        full_sents = int(full_sentence_counts.get(label, 0))
        train_sents = int(train_sentence_counts.get(label, 0))
        eval_sents = int(eval_sentence_counts.get(label, 0))

        rows.append(
            {
                "label": label,
                "full_token_count": full_tokens,
                "train_token_count": train_tokens,
                "eval_token_count": eval_tokens,
                "train_token_share_of_full": (train_tokens / full_tokens) if full_tokens else None,
                "eval_token_share_of_full": (eval_tokens / full_tokens) if full_tokens else None,
                "full_sentence_count": full_sents,
                "train_sentence_count": train_sents,
                "eval_sentence_count": eval_sents,
                "in_train": train_tokens > 0,
                "in_eval": eval_tokens > 0,
                "rare_label_q1": full_tokens <= q1_threshold,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["full_token_count", "label"],
        ascending=[True, True],
        ignore_index=True,
    )


def run() -> dict:
    dataset_override = (os.environ.get("THESIS_NER_CSV") or "").strip()
    dataset_path = Path(dataset_override) if dataset_override else resolve_dataset("ner_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    split_seed = _resolve_seed(default_seed=42)
    split_ratio = 0.7

    worker = PrepDataSetNERTraining()
    data = worker.load_and_prepare_data(str(dataset_path))
    sentences = tf.train_data_fit(data)

    train_sentences, eval_sentences = tf.split_list(
        sentences,
        split_ratio=split_ratio,
        seed=split_seed,
        ensure_label_coverage=True,
    )

    label_df = _label_distribution_report(train_sentences, eval_sentences)
    train_only_labels = sorted(label_df[(label_df["in_train"] == True) & (label_df["in_eval"] == False)]["label"].tolist())
    eval_only_labels = sorted(label_df[(label_df["in_train"] == False) & (label_df["in_eval"] == True)]["label"].tolist())

    rare_df = label_df[label_df["rare_label_q1"] == True].copy()
    rare_in_train = int((rare_df["in_train"] == True).sum()) if not rare_df.empty else 0
    rare_total = int(len(rare_df))

    summary_df = pd.DataFrame(
        [
            {
                "dataset": str(dataset_path),
                "split_seed": split_seed,
                "split_ratio_train": split_ratio,
                "split_ratio_eval": 1 - split_ratio,
                "source_sentences": len(sentences),
                "train_sentences": len(train_sentences),
                "eval_sentences": len(eval_sentences),
                "actual_train_fraction": (len(train_sentences) / len(sentences)) if sentences else None,
                "actual_eval_fraction": (len(eval_sentences) / len(sentences)) if sentences else None,
                "unique_non_o_labels": int(label_df[label_df["label"] != "<none>"].shape[0]),
                "labels_missing_in_train": int((label_df["in_train"] == False).sum()),
                "labels_missing_in_eval": int((label_df["in_eval"] == False).sum()),
                "rare_labels_q1_count": rare_total,
                "rare_labels_q1_covered_in_train": rare_in_train,
                "rare_labels_q1_coverage_train": (rare_in_train / rare_total) if rare_total else None,
            }
        ]
    )

    metrics_file = write_result_excel(
        "exp07",
        "sentence_split_strategy",
        summary_df,
        label_df,
        extra_sheets={
            "rare_labels_q1": rare_df,
            "train_only_labels": pd.DataFrame({"label": train_only_labels}),
            "eval_only_labels": pd.DataFrame({"label": eval_only_labels}),
        },
    )

    result = {
        "experiment_id": "exp07",
        "name": "Sentence-Level 70/30 Statistical Stratified Split Audit",
        "description": (
            "Audits the seeded sentence-level split used in training: 70% train / 30% eval, "
            "with non-O label distribution balancing and best-effort coverage of rare labels in train."
        ),
        "dataset": str(dataset_path),
        "split_parameters": {
            "split_seed": split_seed,
            "train_fraction": split_ratio,
            "eval_fraction": 1 - split_ratio,
            "split_function": "core.th_functions.split_list",
            "ensure_label_coverage": True,
        },
        "summary": summary_df.to_dict(orient="records")[0],
        "metrics_file": str(metrics_file),
        "status": "ok",
    }

    out_path = write_result_json("exp07", "sentence_split_strategy", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    payload = run()
    summary = payload.get("summary", {})
    print(
        "[exp07] "
        f"seed={summary.get('split_seed')} "
        f"train={summary.get('train_sentences')}/{summary.get('source_sentences')} "
        f"eval={summary.get('eval_sentences')}/{summary.get('source_sentences')} "
        f"rare_train_coverage={summary.get('rare_labels_q1_coverage_train')}"
    )
