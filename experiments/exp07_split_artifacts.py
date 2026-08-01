"""Regenerate outputs/exp07/splits JSON from ner_dataset.csv (no NER training)."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import resolve_dataset
from split_io import get_splits_dir, save_split

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import th_functions as tf  # type: ignore
from NERtraining import PrepDataSetNERTraining  # type: ignore

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

def regenerate_exp07_splits(
    *,
    dataset_path: Path | None = None,
    split_seed: int = 42,
    split_ratio: float = 0.7,
    clear_existing: bool = True,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    """Build train/eval JSON + split_meta.json from the current dataset CSV."""
    csv_override = (os.environ.get("THESIS_NER_CSV") or "").strip()
    ds = dataset_path or (Path(csv_override) if csv_override else resolve_dataset("ner_dataset.csv"))
    if not ds.exists():
        raise FileNotFoundError(f"Dataset not found: {ds}")

    worker = PrepDataSetNERTraining()
    data = worker.load_and_prepare_data(str(ds))
    sentences = tf.train_data_fit(data)
    from hebrew_text_io import validate_hebrew_sentence_list

    validate_hebrew_sentence_list(sentences, context=f"exp07 source sentences ({ds})")

    splits_dir = get_splits_dir()
    if clear_existing and splits_dir.exists():
        for p in splits_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass

    use_variants = variants or list(ALL_VARIANTS)
    all_variant_meta: list[dict] = []
    for vk in use_variants:
        fn = SPLIT_FNS.get(vk)
        if fn is None:
            raise KeyError(f"Unknown split variant: {vk}")
        train, eval_ = fn(sentences, split_ratio, split_seed)
        safe_name = vk.replace(" ", "_")
        train_file = f"{safe_name}_train.json"
        eval_file = f"{safe_name}_eval.json"
        save_split(train, splits_dir / train_file)
        save_split(eval_, splits_dir / eval_file)
        all_variant_meta.append({
            "variant": vk,
            "label": THESIS_LABELS.get(vk, vk),
            "description": VARIANT_DESCRIPTIONS.get(vk, vk),
            "f1_mean": None,
            "train_file": train_file,
            "eval_file": eval_file,
        })

    split_meta = {
        "baseline_variant": BEFORE_VARIANT,
        "best_variant": AFTER_VARIANT,
        "best_variant_label": THESIS_LABELS.get(AFTER_VARIANT, AFTER_VARIANT),
        "best_variant_f1_mean": None,
        "dataset": str(ds),
        "seed": split_seed,
        "split_ratio": split_ratio,
        "source_sentences": len(sentences),
        "variants": all_variant_meta,
    }
    meta_path = splits_dir / "split_meta.json"
    meta_path.write_text(json.dumps(split_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "splits_dir": str(splits_dir),
        "split_meta": str(meta_path),
        "dataset": str(ds),
        "variants": use_variants,
        "source_sentences": len(sentences),
    }
