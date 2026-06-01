"""experiment_07_sentence_split_strategy.py — Sentence Split Strategy Comparison
=================================================================================

This experiment evaluates how different train/eval **sentence-split strategies**
affect NER model performance on a Hebrew NER corpus using DictaBERT.

Motivation
----------
In NER datasets with many rare entity types, a naive random split can leave the
training set without examples of some labels, hurting F1.  This experiment
compares three strategies for allocating sentences to train vs eval:

1. **Baseline** — simple random shuffle.
2. **Label-aware greedy** — minimise squared deviation from target label counts.
3. **Multilabel stratified** — iterative stratification (Sechidis et al., 2011).

Each strategy is trained with 5 random seeds and metrics are aggregated as
mean ± std.  After all variants are evaluated, the baseline and best-variant
train/eval sentence lists are **saved to JSON** in ``outputs/exp07/splits/``
so that experiments 03–06 can re-use the same split for a fair comparison.

Outputs
-------
* ``outputs/exp07/*.xlsx`` — Excel workbook with score tables, label-frequency
  analysis, and a documentation sheet for academic citation.
* ``outputs/exp07/*.json`` — machine-readable results.
* ``outputs/exp07/*.csv``  — per-seed / metric-stats / thesis-summary CSVs.
* ``outputs/exp07/splits/`` — saved baseline & best splits as JSON.

Usage
-----
::

    python experiments/experiment_07_sentence_split_strategy.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    configure_model_environment,
    get_experiment_output_dir,
    now_timestamp,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
)
from split_io import save_split, get_splits_dir, build_thesis_documentation_df


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import th_functions as tf  # type: ignore
from NERtraining import PrepDataSetNERTraining  # type: ignore

try:
    from transformers import logging as transformers_logging
except Exception:  # pragma: no cover - optional defensive import
    transformers_logging = None


BEFORE_VARIANT = "before_exp01_baseline"
AFTER_VARIANT = "after_label_aware_split"
VARIANT_MULTILABEL_STRATIFIED = "after_multilabel_stratified"
VARIANT_MULTILABEL_ITERATIVE_PAPER = "after_multilabel_iterative_paper"

BEFORE_DESCRIPTION = "Regular NER with DictaBERT"
AFTER_DESCRIPTION = "Statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)"

ALL_VARIANTS = [
    BEFORE_VARIANT,
    AFTER_VARIANT,
    VARIANT_MULTILABEL_STRATIFIED,
    VARIANT_MULTILABEL_ITERATIVE_PAPER,
]

VARIANT_DESCRIPTIONS = {
    BEFORE_VARIANT: BEFORE_DESCRIPTION,
    AFTER_VARIANT: AFTER_DESCRIPTION,
    VARIANT_MULTILABEL_STRATIFIED: "Iterative multilabel stratification (Sechidis et al., 2011): distributes each label proportionally across train/eval",
    VARIANT_MULTILABEL_ITERATIVE_PAPER: "Paper-style iterative stratification: rarest-label-first with tie-breaks by per-label need, then fold capacity, then random",
}

THESIS_LABELS = {
    BEFORE_VARIANT: "Baseline (simple random split)",
    AFTER_VARIANT: "Label-aware greedy",
    VARIANT_MULTILABEL_STRATIFIED: "Multilabel stratified",
    VARIANT_MULTILABEL_ITERATIVE_PAPER: "Multilabel stratified (paper-style)",
}


def _resolve_seed(default_seed: int = 42) -> int:
    raw = (os.environ.get("THESIS_SPLIT_SEED") or str(default_seed)).strip()
    try:
        return int(raw)
    except ValueError:
        return default_seed


def _resolve_num_seeds(default_num_seeds: int = 5) -> int:
    raw = (os.environ.get("THESIS_EXP07_NUM_SEEDS") or str(default_num_seeds)).strip()
    try:
        value = int(raw)
        return max(2, value)
    except ValueError:
        return default_num_seeds


def _configure_quiet_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if transformers_logging is not None:
        transformers_logging.set_verbosity_error()


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


def _sentence_non_o_label_counts(sentence: dict) -> dict[str, int]:
    labels = sentence.get("labels", []) if isinstance(sentence, dict) else []
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        if key == "O":
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _enforce_sentence_ratio(
    train_sentences: list[dict],
    eval_sentences: list[dict],
    split_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Rebalance sentence counts to exactly match the requested split ratio.

    Some label-aware allocators can drift from the target ratio while trying to
    satisfy label constraints. This post-step preserves their allocation as much
    as possible while enforcing deterministic train/eval sizes.
    """
    total = len(train_sentences) + len(eval_sentences)
    if total <= 1:
        return train_sentences, eval_sentences

    target_train = max(1, min(total - 1, int(total * split_ratio)))
    current_train = len(train_sentences)
    if current_train == target_train:
        return train_sentences, eval_sentences

    rng = random.Random(seed)

    def _move_candidates(items: list[dict]) -> list[int]:
        # Move sentences with fewer non-O tokens first to minimize label impact.
        scored: list[tuple[int, int, float, int]] = []
        for idx, sent in enumerate(items):
            non_o_counts = _sentence_non_o_label_counts(sent)
            non_o_token_count = sum(non_o_counts.values())
            unique_non_o_labels = len(non_o_counts)
            scored.append((non_o_token_count, unique_non_o_labels, rng.random(), idx))
        scored.sort()
        return [idx for _, _, _, idx in scored]

    train = list(train_sentences)
    eval_ = list(eval_sentences)

    if len(train) > target_train:
        need_to_move = len(train) - target_train
        for idx in sorted(_move_candidates(train)[:need_to_move], reverse=True):
            eval_.append(train.pop(idx))
    else:
        need_to_move = target_train - len(train)
        for idx in sorted(_move_candidates(eval_)[:need_to_move], reverse=True):
            train.append(eval_.pop(idx))

    return train, eval_


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


def _simple_random_split(sentences: list[dict], split_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Baseline: shuffle sentences uniformly at random and split at *split_ratio*.

    This is the control condition — no label-awareness whatsoever.
    """
    items = list(sentences)
    if not items:
        return [], []
    if len(items) == 1:
        return items, []

    rng = random.Random(seed)
    rng.shuffle(items)

    split_index = int(len(items) * split_ratio)
    split_index = max(1, min(len(items) - 1, split_index))
    return items[:split_index], items[split_index:]


def _label_aware_split(sentences: list[dict], split_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Wrapper for tf.split_list with label coverage."""
    train, eval_ = tf.split_list(sentences, split_ratio=split_ratio, seed=seed, ensure_label_coverage=True)
    return _enforce_sentence_ratio(train, eval_, split_ratio, seed)


def _multilabel_stratified_split(sentences: list[dict], split_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Iterative multilabel stratification (Sechidis et al., 2011).

    Treats each sentence as a multilabel instance where labels are the unique
    non-O entity types present.  The algorithm processes labels from rarest to
    most common and assigns each sentence to the fold (train or eval) that has
    the greatest remaining need for the sentence's labels.  This produces
    train/eval splits where every label's proportion closely mirrors the
    full-dataset proportion.
    """
    items = list(sentences)
    if not items:
        return [], []
    if len(items) == 1:
        return items, []

    rng = random.Random(seed)

    # Build label sets per sentence (unique non-O labels)
    label_sets: list[frozenset[str]] = []
    for item in items:
        labels = item.get("labels", []) if isinstance(item, dict) else []
        non_o = frozenset(str(lb) for lb in labels if str(lb) != "O")
        label_sets.append(non_o)

    all_labels = sorted(set().union(*label_sets)) if label_sets else []
    if not all_labels:
        return _simple_random_split(items, split_ratio, seed)

    proportions = [split_ratio, 1.0 - split_ratio]  # [train, eval]

    # Per-label: indices of sentences containing it, and desired count per fold
    label_to_indices: dict[str, list[int]] = {lb: [] for lb in all_labels}
    for i, ls in enumerate(label_sets):
        for lb in ls:
            label_to_indices[lb].append(i)

    desired: dict[str, list[float]] = {}
    for lb in all_labels:
        n = len(label_to_indices[lb])
        desired[lb] = [n * p for p in proportions]

    # Assignments: -1 = unassigned, 0 = train, 1 = eval
    assignments = [-1] * len(items)
    current: dict[str, list[int]] = {lb: [0, 0] for lb in all_labels}
    processed: set[str] = set()

    # Iterative stratification: process labels rarest-first
    while len(processed) < len(all_labels):
        # Find unprocessed label with fewest unassigned examples
        min_label: str | None = None
        min_unassigned = len(items) + 1
        for lb in all_labels:
            if lb in processed:
                continue
            unassigned = sum(1 for i in label_to_indices[lb] if assignments[i] == -1)
            if unassigned < min_unassigned:
                min_unassigned = unassigned
                min_label = lb

        if min_label is None:
            break

        # Assign unassigned examples that carry this label
        for idx in label_to_indices[min_label]:
            if assignments[idx] != -1:
                continue
            # Compute each fold's total need across ALL labels of this sentence
            needs = [0.0, 0.0]
            for fold in range(2):
                for lb in label_sets[idx]:
                    needs[fold] += desired[lb][fold] - current[lb][fold]
                needs[fold] += rng.random() * 1e-6  # tie-break
            best_fold = 0 if needs[0] >= needs[1] else 1
            assignments[idx] = best_fold
            for lb in label_sets[idx]:
                current[lb][best_fold] += 1

        processed.add(min_label)

    # Assign remaining sentences (no non-O labels) proportionally
    train_target = max(1, min(len(items) - 1, int(len(items) * split_ratio)))
    current_train = sum(1 for a in assignments if a == 0)
    unassigned = [i for i in range(len(items)) if assignments[i] == -1]
    rng.shuffle(unassigned)
    for idx in unassigned:
        if current_train < train_target:
            assignments[idx] = 0
            current_train += 1
        else:
            assignments[idx] = 1

    train = [items[i] for i in range(len(items)) if assignments[i] == 0]
    eval_ = [items[i] for i in range(len(items)) if assignments[i] == 1]
    return _enforce_sentence_ratio(train, eval_, split_ratio, seed)


def _multilabel_iterative_paper_split(sentences: list[dict], split_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Paper-style iterative stratification (Sechidis et al., 2011 inspired).

    Differences vs the existing multilabel stratified implementation:
    1. Priority label is chosen among labels with fewest remaining unassigned
       examples (random tie-break).
    2. Example assignment tie-breaks first by per-label remaining target,
       then by subset total remaining capacity, then random.
    """
    items = list(sentences)
    n_items = len(items)
    if n_items == 0:
        return [], []
    if n_items == 1:
        return items, []

    rng = random.Random(seed)

    # Build label sets per sentence (unique non-O labels)
    label_sets: list[frozenset[str]] = []
    for item in items:
        labels = item.get("labels", []) if isinstance(item, dict) else []
        non_o = frozenset(str(lb) for lb in labels if str(lb) != "O")
        label_sets.append(non_o)

    all_labels = sorted(set().union(*label_sets)) if label_sets else []
    if not all_labels:
        return _simple_random_split(items, split_ratio, seed)

    # Exact fold capacity targets (train/eval)
    train_target = max(1, min(n_items - 1, int(n_items * split_ratio)))
    eval_target = n_items - train_target
    remaining_total = [float(train_target), float(eval_target)]

    # Per-label desired counts per fold
    label_to_indices: dict[str, list[int]] = {lb: [] for lb in all_labels}
    for i, ls in enumerate(label_sets):
        for lb in ls:
            label_to_indices[lb].append(i)

    remaining_label_target: dict[str, list[float]] = {}
    for lb in all_labels:
        n_lb = len(label_to_indices[lb])
        remaining_label_target[lb] = [n_lb * split_ratio, n_lb * (1.0 - split_ratio)]

    assignments = [-1] * n_items  # -1 unassigned, 0 train, 1 eval

    while True:
        unassigned_idx = [i for i, a in enumerate(assignments) if a == -1]
        if not unassigned_idx:
            break

        # Remaining counts per label among unassigned examples
        rem_counts: dict[str, int] = {}
        for i in unassigned_idx:
            for lb in label_sets[i]:
                rem_counts[lb] = rem_counts.get(lb, 0) + 1

        positive_labels = [lb for lb, cnt in rem_counts.items() if cnt > 0]
        if not positive_labels:
            # Remaining are O-only sentences; place by remaining fold capacity.
            for i in unassigned_idx:
                if remaining_total[0] > remaining_total[1]:
                    chosen_fold = 0
                elif remaining_total[1] > remaining_total[0]:
                    chosen_fold = 1
                else:
                    chosen_fold = rng.choice([0, 1])
                assignments[i] = chosen_fold
                remaining_total[chosen_fold] -= 1.0
            break

        # Priority label: fewest remaining examples, random tie-break.
        min_count = min(rem_counts[lb] for lb in positive_labels)
        tied = [lb for lb in positive_labels if rem_counts[lb] == min_count]
        priority_label = rng.choice(tied)

        # Process all currently-unassigned examples containing priority label.
        candidate_indices = [i for i in label_to_indices[priority_label] if assignments[i] == -1]
        rng.shuffle(candidate_indices)
        for i in candidate_indices:
            # Tie-break 1: maximize remaining target for priority label
            label_needs = remaining_label_target[priority_label]
            max_label_need = max(label_needs)
            best_folds = [f for f in (0, 1) if label_needs[f] == max_label_need]

            # Tie-break 2: among best folds, maximize remaining total capacity
            if len(best_folds) > 1:
                max_total_need = max(remaining_total[f] for f in best_folds)
                best_folds = [f for f in best_folds if remaining_total[f] == max_total_need]

            # Tie-break 3: random
            chosen_fold = best_folds[0] if len(best_folds) == 1 else rng.choice(best_folds)

            assignments[i] = chosen_fold
            remaining_total[chosen_fold] -= 1.0
            for lb in label_sets[i]:
                remaining_label_target[lb][chosen_fold] -= 1.0

    train = [items[i] for i in range(n_items) if assignments[i] == 0]
    eval_ = [items[i] for i in range(n_items) if assignments[i] == 1]
    return _enforce_sentence_ratio(train, eval_, split_ratio, seed)


SPLIT_FNS = {
    BEFORE_VARIANT: _simple_random_split,
    AFTER_VARIANT: _label_aware_split,
    VARIANT_MULTILABEL_STRATIFIED: _multilabel_stratified_split,
    VARIANT_MULTILABEL_ITERATIVE_PAPER: _multilabel_iterative_paper_split,
}


def _build_split_artifacts(
    variant: str,
    train_sentences: list[dict],
    eval_sentences: list[dict],
    split_seed: int,
    split_ratio: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    label_df = _label_distribution_report(train_sentences, eval_sentences)
    train_only_labels = sorted(
        label_df[(label_df["in_train"] == True) & (label_df["in_eval"] == False)]["label"].tolist()
    )
    eval_only_labels = sorted(
        label_df[(label_df["in_train"] == False) & (label_df["in_eval"] == True)]["label"].tolist()
    )

    rare_df = label_df[label_df["rare_label_q1"] == True].copy()
    rare_in_train = int((rare_df["in_train"] == True).sum()) if not rare_df.empty else 0
    rare_total = int(len(rare_df))

    summary = {
        "variant": variant,
        "split_seed": split_seed,
        "split_ratio_train": split_ratio,
        "split_ratio_eval": 1 - split_ratio,
        "source_sentences": len(train_sentences) + len(eval_sentences),
        "train_sentences": len(train_sentences),
        "eval_sentences": len(eval_sentences),
        "actual_train_fraction": (
            len(train_sentences) / (len(train_sentences) + len(eval_sentences))
            if (len(train_sentences) + len(eval_sentences))
            else None
        ),
        "actual_eval_fraction": (
            len(eval_sentences) / (len(train_sentences) + len(eval_sentences))
            if (len(train_sentences) + len(eval_sentences))
            else None
        ),
        "unique_non_o_labels": int(label_df[label_df["label"] != "<none>"].shape[0]),
        "labels_missing_in_train": int((label_df["in_train"] == False).sum()),
        "labels_missing_in_eval": int((label_df["in_eval"] == False).sum()),
        "rare_labels_q1_count": rare_total,
        "rare_labels_q1_covered_in_train": rare_in_train,
        "rare_labels_q1_coverage_train": (rare_in_train / rare_total) if rare_total else None,
    }
    return summary, label_df, rare_df, train_only_labels, eval_only_labels


def _train_split(
    data: pd.DataFrame,
    train_sentences: list[dict],
    eval_sentences: list[dict],
    model_name: str,
    is_local_model: bool,
) -> dict:
    # Reconstruct the expected output model directory to check for cached trained model
    exp_id = os.environ.get("THESIS_CURRENT_EXP_ID", "exp07")
    model_id = os.environ.get("THESIS_MODEL_NAME", model_name)
    condition_key = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
    seed = os.environ.get("THESIS_SPLIT_SEED", "42")
    model_short = model_id.replace("/", "_").replace("\\", "_").split("_")[-1]
    save_name = f"{exp_id}_{model_short}_{condition_key}_seed{seed}"
    
    model_save_base = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "outputs", "trained_models"
    )
    model_save_path = os.path.join(model_save_base, save_name)
    
    old_epochs = os.environ.get("THESIS_NUM_EPOCHS")
    
    if os.path.exists(os.path.join(model_save_path, "model.safetensors")):
        print(f"[Cache Skip] Model already fully trained at {model_save_path}. Evaluating saved weights.")
        model_name = model_save_path
        is_local_model = True
        os.environ["THESIS_NUM_EPOCHS"] = "0.0"

    try:
        model, tokenizer, data_collator, ds_train, ds_eval, _, label_list = tf.setup_token_classification(
            data=data,
            train_data=train_sentences,
            test_data=eval_sentences,
            eval_data=eval_sentences,
            model_name=model_name,
            local_files_only=is_local_model,
        )
        _, eval_results = tf.train_and_evaluate_model(
            model,
            ds_train,
            ds_eval,
            data_collator,
            tokenizer,
            label_list,
            metric_name="seqeval",
        )
    finally:
        # Restore old epochs env variable
        if old_epochs is not None:
            os.environ["THESIS_NUM_EPOCHS"] = old_epochs
        else:
            os.environ.pop("THESIS_NUM_EPOCHS", None)

    return {
        "f1": eval_results.get("eval_overall_f1"),
        "precision": eval_results.get("eval_overall_precision"),
        "recall": eval_results.get("eval_overall_recall"),
        "accuracy": eval_results.get("eval_overall_accuracy"),
        "loss": eval_results.get("eval_loss"),
    }


def _fmt_metric(value: float | None) -> str:
    return f"{float(value):.4f}" if value is not None else "N/A"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
        if pd.isna(numeric):
            return None
        return numeric
    except Exception:
        return None


def _build_metric_stats(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["f1", "precision", "recall", "accuracy", "loss"]
    variants_in_data = list(per_seed_df["variant"].unique())
    ordered_variants = [v for v in ALL_VARIANTS if v in variants_in_data]

    rows = []
    for variant in ordered_variants:
        subset = per_seed_df[per_seed_df["variant"] == variant]
        desc = VARIANT_DESCRIPTIONS.get(variant, variant)
        row = {
            "variant": variant,
            "variant_description": desc,
            "seeds": int(subset["split_seed"].nunique()),
        }
        for metric in metric_names:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
        rows.append(row)

    baseline_row = rows[0] if rows else None
    for row in rows[1:]:
        delta = {
            "variant": f"delta_{row['variant']}_minus_baseline",
            "variant_description": f"Delta: {THESIS_LABELS.get(row['variant'], row['variant'])} - baseline",
            "seeds": min(int(baseline_row["seeds"]), int(row["seeds"])) if baseline_row else 0,
        }
        for metric in metric_names:
            b = _safe_float(baseline_row.get(f"{metric}_mean")) if baseline_row else None
            a = _safe_float(row.get(f"{metric}_mean"))
            delta[f"{metric}_mean"] = (a - b) if a is not None and b is not None else None
            delta[f"{metric}_std"] = None
        rows.append(delta)

    return pd.DataFrame(rows)


def _build_thesis_summary_table(metric_stats_df: pd.DataFrame) -> pd.DataFrame:
    non_delta = metric_stats_df[~metric_stats_df["variant"].str.startswith("delta_")].copy()
    non_delta["Condition"] = non_delta["variant"].map(THESIS_LABELS).fillna(non_delta["variant"])
    non_delta["F1 (mean±std)"] = non_delta.apply(
        lambda r: f"{_fmt_metric(r.get('f1_mean'))}±{_fmt_metric(r.get('f1_std'))}", axis=1,
    )
    non_delta["Precision (mean±std)"] = non_delta.apply(
        lambda r: f"{_fmt_metric(r.get('precision_mean'))}±{_fmt_metric(r.get('precision_std'))}", axis=1,
    )
    non_delta["Recall (mean±std)"] = non_delta.apply(
        lambda r: f"{_fmt_metric(r.get('recall_mean'))}±{_fmt_metric(r.get('recall_std'))}", axis=1,
    )
    non_delta["Accuracy (mean±std)"] = non_delta.apply(
        lambda r: f"{_fmt_metric(r.get('accuracy_mean'))}±{_fmt_metric(r.get('accuracy_std'))}", axis=1,
    )

    thesis_df = non_delta[
        ["Condition", "seeds", "F1 (mean±std)", "Precision (mean±std)", "Recall (mean±std)", "Accuracy (mean±std)"]
    ].copy()

    baseline_stats = non_delta[non_delta["variant"] == BEFORE_VARIANT]
    if not baseline_stats.empty:
        bs = baseline_stats.iloc[0]
        for _, vs in non_delta[non_delta["variant"] != BEFORE_VARIANT].iterrows():
            delta_row = {
                "Condition": f"Delta ({THESIS_LABELS.get(vs['variant'], vs['variant'])} - baseline)",
                "seeds": min(int(bs["seeds"]), int(vs["seeds"])),
                "F1 (mean±std)": _fmt_metric(_safe_float(vs.get("f1_mean")) - _safe_float(bs.get("f1_mean"))),
                "Precision (mean±std)": _fmt_metric(
                    _safe_float(vs.get("precision_mean")) - _safe_float(bs.get("precision_mean"))
                ),
                "Recall (mean±std)": _fmt_metric(
                    _safe_float(vs.get("recall_mean")) - _safe_float(bs.get("recall_mean"))
                ),
                "Accuracy (mean±std)": _fmt_metric(
                    _safe_float(vs.get("accuracy_mean")) - _safe_float(bs.get("accuracy_mean"))
                ),
            }
            thesis_df = pd.concat([thesis_df, pd.DataFrame([delta_row])], ignore_index=True)
    return thesis_df


def _build_excel_score_tables(
    per_seed_df: pd.DataFrame,
    metric_stats_df: pd.DataFrame,
    thesis_summary_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    per_seed_scores_df = per_seed_df[
        ["variant", "split_seed", "f1", "precision", "recall", "accuracy", "loss"]
    ].copy()
    per_seed_scores_df["Condition"] = per_seed_scores_df["variant"].map(THESIS_LABELS).fillna(per_seed_scores_df["variant"])
    per_seed_scores_df = per_seed_scores_df[
        ["Condition", "variant", "split_seed", "f1", "precision", "recall", "accuracy", "loss"]
    ]

    non_delta_stats = metric_stats_df[~metric_stats_df["variant"].str.startswith("delta_")].copy()
    non_delta_stats["Condition"] = non_delta_stats["variant"].map(THESIS_LABELS).fillna(non_delta_stats["variant"])
    non_delta_stats["f1_ci95"] = non_delta_stats.apply(
        lambda r: (1.96 * float(r["f1_std"]) / (float(r["seeds"]) ** 0.5))
        if _safe_float(r.get("f1_std")) is not None and float(r.get("seeds", 0)) > 0
        else None,
        axis=1,
    )

    score_summary_numeric_df = non_delta_stats[
        [
            "Condition",
            "variant",
            "seeds",
            "f1_mean",
            "f1_std",
            "f1_ci95",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "accuracy_mean",
            "accuracy_std",
            "loss_mean",
            "loss_std",
        ]
    ].copy()

    score_ranking_f1_df = score_summary_numeric_df.sort_values(by=["f1_mean", "recall_mean"], ascending=[False, False]).copy()
    score_ranking_f1_df.insert(0, "rank_by_f1", range(1, len(score_ranking_f1_df) + 1))

    delta_rows = metric_stats_df[metric_stats_df["variant"].str.startswith("delta_")].copy()
    if not delta_rows.empty:
        delta_rows["Condition"] = delta_rows["variant_description"].fillna(delta_rows["variant"])
        score_deltas_df = delta_rows[
            [
                "Condition",
                "variant",
                "seeds",
                "f1_mean",
                "precision_mean",
                "recall_mean",
                "accuracy_mean",
                "loss_mean",
            ]
        ].copy()
    else:
        score_deltas_df = pd.DataFrame(
            columns=[
                "Condition",
                "variant",
                "seeds",
                "f1_mean",
                "precision_mean",
                "recall_mean",
                "accuracy_mean",
                "loss_mean",
            ]
        )

    return {
        "score_overview": thesis_summary_df.copy(),
        "per_seed_scores": per_seed_scores_df,
        "score_summary_numeric": score_summary_numeric_df,
        "score_ranking_f1": score_ranking_f1_df,
        "score_deltas_vs_baseline": score_deltas_df,
    }


def _build_training_label_count_table(
    original_train_sentences: list[dict],
    adjusted_train_sentences: list[dict],
) -> pd.DataFrame:
    original_instance = _non_o_label_counts(original_train_sentences)
    adjusted_instance = _non_o_label_counts(adjusted_train_sentences)
    original_distinct = _sentence_presence_counts(original_train_sentences)
    adjusted_distinct = _sentence_presence_counts(adjusted_train_sentences)

    labels = sorted(set(original_instance) | set(adjusted_instance) | set(original_distinct) | set(adjusted_distinct))
    if not labels:
        return pd.DataFrame(
            [
                {
                    "Label": "<none>",
                    "Before - Entity Token Count": 0,
                    "Before - Entity Token %": 0.0,
                    "After - Entity Token Count": 0,
                    "After - Entity Token %": 0.0,
                    "Delta - Entity Token Count": 0,
                    "Delta - Entity Token %": 0.0,
                    "Before - Sentence Count": 0,
                    "Before - Sentence %": 0.0,
                    "After - Sentence Count": 0,
                    "After - Sentence %": 0.0,
                    "Delta - Sentence Count": 0,
                    "Delta - Sentence %": 0.0,
                    "Before - In Train": False,
                    "After - In Train": False,
                    "original_instance_count": 0,
                    "original_distinct_sentence_count": 0,
                    "after_instance_count": 0,
                    "after_distinct_sentence_count": 0,
                }
            ]
        )

    total_original_instance = sum(original_instance.values())
    total_adjusted_instance = sum(adjusted_instance.values())
    total_original_sentences = max(1, len(original_train_sentences))
    total_adjusted_sentences = max(1, len(adjusted_train_sentences))

    rows = []
    for label in labels:
        o_i = int(original_instance.get(label, 0))
        a_i = int(adjusted_instance.get(label, 0))
        o_d = int(original_distinct.get(label, 0))
        a_d = int(adjusted_distinct.get(label, 0))

        o_i_pct = (100.0 * o_i / total_original_instance) if total_original_instance > 0 else 0.0
        a_i_pct = (100.0 * a_i / total_adjusted_instance) if total_adjusted_instance > 0 else 0.0
        o_d_pct = 100.0 * o_d / total_original_sentences
        a_d_pct = 100.0 * a_d / total_adjusted_sentences

        rows.append(
            {
                "Label": label,
                "Before - Entity Token Count": o_i,
                "Before - Entity Token %": round(o_i_pct, 4),
                "After - Entity Token Count": a_i,
                "After - Entity Token %": round(a_i_pct, 4),
                "Delta - Entity Token Count": a_i - o_i,
                "Delta - Entity Token %": round(a_i_pct - o_i_pct, 4),
                "Before - Sentence Count": o_d,
                "Before - Sentence %": round(o_d_pct, 4),
                "After - Sentence Count": a_d,
                "After - Sentence %": round(a_d_pct, 4),
                "Delta - Sentence Count": a_d - o_d,
                "Delta - Sentence %": round(a_d_pct - o_d_pct, 4),
                "Before - In Train": o_i > 0,
                "After - In Train": a_i > 0,
                "original_instance_count": o_i,
                "original_distinct_sentence_count": o_d,
                "after_instance_count": a_i,
                "after_distinct_sentence_count": a_d,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["Delta - Entity Token Count", "After - Entity Token Count", "Label"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _write_csv_outputs(
    exp_dir: Path,
    per_seed_df: pd.DataFrame,
    metric_stats_df: pd.DataFrame,
    thesis_df: pd.DataFrame,
    label_count_table_df: pd.DataFrame,
) -> dict:
    ts = now_timestamp()

    def _safe_write_latest(df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            pass  # file open elsewhere — skip latest copy

    per_seed_csv = exp_dir / f"sentence_split_strategy_per_seed_{ts}.csv"
    per_seed_df.to_csv(per_seed_csv, index=False)
    per_seed_latest = exp_dir / "sentence_split_strategy_per_seed_latest.csv"
    _safe_write_latest(per_seed_df, per_seed_latest)

    stats_csv = exp_dir / f"sentence_split_strategy_metric_stats_{ts}.csv"
    metric_stats_df.to_csv(stats_csv, index=False)
    stats_latest = exp_dir / "sentence_split_strategy_metric_stats_latest.csv"
    _safe_write_latest(metric_stats_df, stats_latest)

    thesis_csv = exp_dir / f"sentence_split_strategy_thesis_summary_{ts}.csv"
    thesis_df.to_csv(thesis_csv, index=False)
    thesis_latest = exp_dir / "sentence_split_strategy_thesis_summary_latest.csv"
    _safe_write_latest(thesis_df, thesis_latest)

    label_count_csv = exp_dir / f"sentence_split_strategy_training_label_count_{ts}.csv"
    label_count_table_df.to_csv(label_count_csv, index=False)
    label_count_latest = exp_dir / "sentence_split_strategy_training_label_count_latest.csv"
    _safe_write_latest(label_count_table_df, label_count_latest)

    return {
        "per_seed_csv": str(per_seed_csv),
        "per_seed_csv_latest": str(per_seed_latest),
        "metric_stats_csv": str(stats_csv),
        "metric_stats_csv_latest": str(stats_latest),
        "thesis_summary_csv": str(thesis_csv),
        "thesis_summary_csv_latest": str(thesis_latest),
        "training_label_count_csv": str(label_count_csv),
        "training_label_count_csv_latest": str(label_count_latest),
    }


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = df.astype(object).where(pd.notna(df), None)
    return normalized.to_dict(orient="records")


def _variant_label_table_sheet_name(variant_key: str) -> str:
    if variant_key == BEFORE_VARIANT:
        return "label_table_baseline"
    if variant_key == AFTER_VARIANT:
        return "label_table_label_aware"
    if variant_key == VARIANT_MULTILABEL_STRATIFIED:
        return "label_table_multilabel"
    if variant_key == VARIANT_MULTILABEL_ITERATIVE_PAPER:
        return "label_table_ml_paper"
    return f"label_table_{variant_key}"[:31]


def _build_label_summary_tables(detailed_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build one summary table per split variant from label_distribution_details.

    Each table reports per-label mean counts across seeds and derived percentages.
    """
    if detailed_df.empty:
        return {}

    tables: dict[str, pd.DataFrame] = {}

    for variant_key in ALL_VARIANTS:
        subset = detailed_df[detailed_df["variant"] == variant_key].copy()
        if subset.empty:
            continue

        grouped = (
            subset.groupby("label", as_index=False)[
                ["full_token_count", "train_token_count", "eval_token_count"]
            ]
            .mean()
        )

        full_total = float(grouped["full_token_count"].sum())
        train_total = float(grouped["train_token_count"].sum())
        test_total = float(grouped["eval_token_count"].sum())

        rows: list[dict[str, Any]] = []
        for _, row in grouped.sort_values("label").iterrows():
            label = str(row["label"])
            full_count = float(row["full_token_count"])
            train_count = float(row["train_token_count"])
            test_count = float(row["eval_token_count"])

            train_ratio = (train_count / test_count) if test_count > 0 else None

            rows.append(
                {
                    "Label": label,
                    "Full Count": round(full_count, 1),
                    "Full %": round((100.0 * full_count / full_total), 2) if full_total > 0 else None,
                    "Train Count": round(train_count, 1),
                    "Train %": round((100.0 * train_count / full_count), 2) if full_count > 0 else None,
                    "Test Count": round(test_count, 1),
                    "Test %": round((100.0 * test_count / full_count), 2) if full_count > 0 else None,
                    "Train/Test Ratio": round(train_ratio, 2) if train_ratio is not None else "N/A",
                }
            )

        rows.append(
            {
                "Label": "TOTAL",
                "Full Count": round(full_total, 1),
                "Full %": 100.0 if full_total > 0 else None,
                "Train Count": round(train_total, 1),
                "Train %": round((100.0 * train_total / full_total), 2) if full_total > 0 else None,
                "Test Count": round(test_total, 1),
                "Test %": round((100.0 * test_total / full_total), 2) if full_total > 0 else None,
                "Train/Test Ratio": "",
            }
        )

        table_df = pd.DataFrame(rows)
        sheet_name = _variant_label_table_sheet_name(variant_key)
        tables[sheet_name] = table_df

    return tables


def run() -> dict:
    dataset_override = (os.environ.get("THESIS_NER_CSV") or "").strip()
    dataset_path = Path(dataset_override) if dataset_override else resolve_dataset("ner_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    split_seed = _resolve_seed(default_seed=42)
    num_seeds = _resolve_num_seeds(default_num_seeds=5)
    split_ratio = 0.7
    _configure_quiet_runtime()
    model_name, is_local_model = configure_model_environment()
    exp_dir = get_experiment_output_dir("exp07")

    worker = PrepDataSetNERTraining()
    with suppress_output_if_needed():
        data = worker.load_and_prepare_data(str(dataset_path))
        sentences = tf.train_data_fit(data)

    per_seed_rows: list[dict[str, Any]] = []
    label_distribution_rows: list[pd.DataFrame] = []
    rare_sheet_rows: list[pd.DataFrame] = []
    train_only_rows: list[pd.DataFrame] = []
    eval_only_rows: list[pd.DataFrame] = []
    first_seed_train_sets: dict[str, list[dict]] = {}
    first_seed_eval_sets: dict[str, list[dict]] = {}

    for seed_offset in range(num_seeds):
        current_seed = split_seed + seed_offset
        print(f"  seed {current_seed} ({seed_offset + 1}/{num_seeds})")

        for variant_key in ALL_VARIANTS:
            split_fn = SPLIT_FNS[variant_key]
            train, eval_ = split_fn(sentences, split_ratio, current_seed)

            # Check if cached metrics and a fully trained model are already available to skip loading and evaluation entirely
            cache_dir = exp_dir / "metrics_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_variant_key = variant_key.replace(" ", "_")
            metrics_cache_file = cache_dir / f"metrics_seed{current_seed}_{safe_variant_key}.json"
            
            # Reconstruct model save path to verify model existence
            model_short_repr = model_name.replace("/", "_").replace("\\", "_").split("_")[-1]
            check_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "outputs", 
                "trained_models", 
                f"exp07_{model_short_repr}_{safe_variant_key}_seed{current_seed}"
            )
            has_trained_model = os.path.exists(os.path.join(check_model_path, "model.safetensors"))

            metrics = None
            if metrics_cache_file.exists() and has_trained_model:
                try:
                    import json as _json
                    metrics = _json.loads(metrics_cache_file.read_text(encoding="utf-8"))
                    print(f"    [Cache Skip] Loaded cached metrics for {variant_key} from cache file.")
                except Exception as e:
                    print(f"    [Cache Skip Warning] Failed to load cached metrics: {e}")
                    metrics = None

            if metrics is None:
                with suppress_output_if_needed():
                    # Set environment variables so HuggingFace Trainer uniquely isolates colab checkpoints
                    os.environ["THESIS_CURRENT_EXP_ID"] = "exp07"
                    os.environ["THESIS_CURRENT_CONDITION_KEY"] = safe_variant_key
                    os.environ["THESIS_SPLIT_SEED"] = str(current_seed)
                    os.environ["THESIS_MODEL_NAME"] = model_name

                    metrics = _train_split(
                        data=data,
                        train_sentences=train,
                        eval_sentences=eval_,
                        model_name=model_name,
                        is_local_model=is_local_model,
                    )
                # Save to cache
                try:
                    import json as _json
                    metrics_cache_file.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"    [Cache Save Error] Failed to write cache file: {e}")

            summary, label_df, rare_df_v, train_only, eval_only = _build_split_artifacts(
                variant=variant_key,
                train_sentences=train,
                eval_sentences=eval_,
                split_seed=current_seed,
                split_ratio=split_ratio,
            )

            per_seed_rows.append(
                {**summary, **metrics, "variant_description": VARIANT_DESCRIPTIONS.get(variant_key, variant_key)}
            )
            label_distribution_rows.append(label_df.assign(variant=variant_key, split_seed=current_seed))
            rare_sheet_rows.append(rare_df_v.assign(variant=variant_key, split_seed=current_seed))
            train_only_rows.append(
                pd.DataFrame({"variant": variant_key, "split_seed": current_seed, "label": train_only})
            )
            eval_only_rows.append(
                pd.DataFrame({"variant": variant_key, "split_seed": current_seed, "label": eval_only})
            )

            if seed_offset == 0:
                first_seed_train_sets[variant_key] = list(train)
                first_seed_eval_sets[variant_key] = list(eval_)

    per_seed_df = pd.DataFrame(per_seed_rows)
    metric_stats_df = _build_metric_stats(per_seed_df)
    thesis_summary_df = _build_thesis_summary_table(metric_stats_df)
    excel_score_tables = _build_excel_score_tables(per_seed_df, metric_stats_df, thesis_summary_df)

    detailed_df = pd.concat(label_distribution_rows, ignore_index=True)
    rare_df = pd.concat(rare_sheet_rows, ignore_index=True)
    train_only_df = pd.concat(train_only_rows, ignore_index=True)
    eval_only_df = pd.concat(eval_only_rows, ignore_index=True)
    label_summary_tables = _build_label_summary_tables(detailed_df)

    baseline_train = first_seed_train_sets.get(BEFORE_VARIANT, [])
    best_after_key = AFTER_VARIANT
    best_f1 = -1.0
    for vk in ALL_VARIANTS:
        if vk == BEFORE_VARIANT:
            continue
        subset = per_seed_df[per_seed_df["variant"] == vk]
        mean_f1 = float(pd.to_numeric(subset["f1"], errors="coerce").dropna().mean())
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_after_key = vk
    training_label_count_table_df = _build_training_label_count_table(
        original_train_sentences=baseline_train,
        adjusted_train_sentences=first_seed_train_sets.get(best_after_key, []),
    )

    # ---- Save ALL variant splits to JSON for reuse by exp03–06 ----
    splits_dir = get_splits_dir()
    all_variant_meta: list[dict] = []
    for vk in ALL_VARIANTS:
        safe_name = vk.replace(" ", "_")
        save_split(first_seed_train_sets.get(vk, []), splits_dir / f"{safe_name}_train.json")
        save_split(first_seed_eval_sets.get(vk, []),  splits_dir / f"{safe_name}_eval.json")
        vk_subset = per_seed_df[per_seed_df["variant"] == vk]
        vk_f1 = float(pd.to_numeric(vk_subset["f1"], errors="coerce").dropna().mean()) if not vk_subset.empty else None
        all_variant_meta.append({
            "variant": vk,
            "label": THESIS_LABELS.get(vk, vk),
            "description": VARIANT_DESCRIPTIONS.get(vk, vk),
            "f1_mean": vk_f1,
            "train_file": f"{safe_name}_train.json",
            "eval_file": f"{safe_name}_eval.json",
        })
    split_meta = {
        "baseline_variant": BEFORE_VARIANT,
        "best_variant": best_after_key,
        "best_variant_label": THESIS_LABELS.get(best_after_key, best_after_key),
        "best_variant_f1_mean": best_f1,
        "seed": split_seed,
        "num_seeds": num_seeds,
        "seed_list": [split_seed + i for i in range(num_seeds)],
        "split_ratio": split_ratio,
        "variants": all_variant_meta,
    }
    import json as _json
    (splits_dir / "split_meta.json").write_text(
        _json.dumps(split_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved splits ({len(ALL_VARIANTS)} variants) to {splits_dir}")

    csv_outputs = _write_csv_outputs(
        exp_dir,
        per_seed_df,
        metric_stats_df,
        thesis_summary_df,
        training_label_count_table_df,
    )

    metrics_file = write_result_excel(
        "exp07",
        "sentence_split_strategy",
        excel_score_tables["score_overview"],
        excel_score_tables["per_seed_scores"],
        extra_sheets={
            "metric_stats": metric_stats_df,
            "score_summary_numeric": excel_score_tables["score_summary_numeric"],
            "score_ranking_f1": excel_score_tables["score_ranking_f1"],
            "score_deltas_vs_baseline": excel_score_tables["score_deltas_vs_baseline"],
            "training_label_count": training_label_count_table_df,
            "label_distribution_details": detailed_df,
            "rare_labels_q1": rare_df,
            "train_only_labels": train_only_df,
            "eval_only_labels": eval_only_df,
            **label_summary_tables,
            "documentation": build_thesis_documentation_df(
                "exp07",
                "Sentence Split Strategy Comparison",
                extra_rows=[
                    {"Section": "Experiment", "Key": "Variants", "Value": "; ".join(
                        f"{THESIS_LABELS.get(v, v)}: {VARIANT_DESCRIPTIONS.get(v, v)}" for v in ALL_VARIANTS
                    )},
                    {"Section": "Experiment", "Key": "Best Variant", "Value": THESIS_LABELS.get(best_after_key, best_after_key)},
                    {"Section": "Experiment", "Key": "Best Variant F1 Mean", "Value": f"{best_f1:.4f}"},
                    {"Section": "Saved Splits", "Key": "Location", "Value": str(splits_dir)},
                    {"Section": "Saved Splits", "Key": "Files", "Value": "baseline_train.json, baseline_eval.json, best_train.json, best_eval.json"},
                ],
            ),
        },
    )

    metric_stats_records = _records_for_json(metric_stats_df)
    thesis_summary_records = _records_for_json(thesis_summary_df)

    variant_stats: dict[str, Any] = {}
    for rec in metric_stats_records:
        variant_stats[rec["variant"]] = rec

    result = {
        "experiment_id": "exp07",
        "name": "Sentence Split Strategy Multi-Seed Multi-Strategy Comparison",
        "description": (
            "Runs multi-seed training comparisons across sentence-split strategies, "
            "then reports detailed score summaries and before/after training-label distributions."
        ),
        "dataset": str(dataset_path),
        "model": model_name,
        "model_local": is_local_model,
        "split_parameters": {
            "base_split_seed": split_seed,
            "num_seeds": num_seeds,
            "seed_list": [split_seed + i for i in range(num_seeds)],
            "train_fraction": split_ratio,
            "eval_fraction": 1 - split_ratio,
            "variants": {k: v for k, v in VARIANT_DESCRIPTIONS.items()},
        },
        "metric_stats": metric_stats_records,
        "thesis_summary": thesis_summary_records,
        "training_label_count_table": _records_for_json(training_label_count_table_df),
        "variant_stats": variant_stats,
        "csv_outputs": csv_outputs,
        "metrics_file": str(metrics_file),
        "status": "ok",
    }

    out_path = write_result_json("exp07", "sentence_split_strategy", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    payload = run()
    split_params = payload.get("split_parameters", {})
    vstats = payload.get("variant_stats", {})
    print(
        f"[exp07] seeds={split_params.get('num_seeds')} "
        f"base_seed={split_params.get('base_split_seed')} "
        f"variants={len(ALL_VARIANTS)}"
    )
    print()
    header = f"{'Condition':<40s} {'F1 (mean±std)':>18s} {'Precision':>18s} {'Recall':>18s} {'Accuracy':>18s}"
    print(header)
    print("-" * len(header))
    for variant_key in ALL_VARIANTS:
        stats = vstats.get(variant_key, {})
        label = THESIS_LABELS.get(variant_key, variant_key)[:40]
        print(
            f"{label:<40s} "
            f"{_fmt_metric(stats.get('f1_mean'))}±{_fmt_metric(stats.get('f1_std')):>7s} "
            f"{_fmt_metric(stats.get('precision_mean'))}±{_fmt_metric(stats.get('precision_std')):>7s} "
            f"{_fmt_metric(stats.get('recall_mean'))}±{_fmt_metric(stats.get('recall_std')):>7s} "
            f"{_fmt_metric(stats.get('accuracy_mean'))}±{_fmt_metric(stats.get('accuracy_std')):>7s}"
        )
    print()
    baseline = vstats.get(BEFORE_VARIANT, {})
    print(f"{'Deltas vs baseline:':<40s}")
    for variant_key in ALL_VARIANTS[1:]:
        delta_key = f"delta_{variant_key}_minus_baseline"
        delta = vstats.get(delta_key, {})
        label = THESIS_LABELS.get(variant_key, variant_key)[:40]
        print(
            f"  {label:<38s} "
            f"F1={_fmt_metric(delta.get('f1_mean'))} "
            f"Prec={_fmt_metric(delta.get('precision_mean'))} "
            f"Rec={_fmt_metric(delta.get('recall_mean'))} "
            f"Acc={_fmt_metric(delta.get('accuracy_mean'))}"
        )
