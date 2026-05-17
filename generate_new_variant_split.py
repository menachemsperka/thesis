"""
generate_new_variant_split.py — Generate only Split 4 (paper-style iterative
multilabel stratification) and patch split_meta.json so comparison scripts can
pick it up.

Usage:
    python generate_new_variant_split.py

This does NOT retrain anything. It only:
1. Loads the NER dataset.
2. Runs the new _multilabel_iterative_paper_split at seed 42 and 70/30 ratio.
3. Saves the train/eval JSON files into outputs/exp07/splits/.
4. Patches split_meta.json to include the new variant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from split_io import save_split, get_splits_dir
from experiment_07_sentence_split_strategy import (
    _multilabel_iterative_paper_split,
    VARIANT_MULTILABEL_ITERATIVE_PAPER,
    VARIANT_DESCRIPTIONS,
    THESIS_LABELS,
)
import th_functions as tf
from NERtraining import PrepDataSetNERTraining


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "ner_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("Loading dataset...")
    worker = PrepDataSetNERTraining()
    data = worker.load_and_prepare_data(str(dataset_path))
    sentences = tf.train_data_fit(data)
    print(f"  {len(sentences)} sentences loaded.")

    seed = 42
    split_ratio = 0.7
    print(f"Running Split 4 paper-style iterative split (seed={seed}, ratio={split_ratio})...")
    train, eval_ = _multilabel_iterative_paper_split(sentences, split_ratio, seed)
    print(f"  Train: {len(train)} sentences, Eval: {len(eval_)} sentences")

    splits_dir = get_splits_dir()
    safe_name = VARIANT_MULTILABEL_ITERATIVE_PAPER.replace(" ", "_")
    train_file = f"{safe_name}_train.json"
    eval_file = f"{safe_name}_eval.json"

    save_split(train, splits_dir / train_file)
    save_split(eval_, splits_dir / eval_file)
    print(f"  Saved: {splits_dir / train_file}")
    print(f"  Saved: {splits_dir / eval_file}")

    # Patch split_meta.json
    meta_path = splits_dir / "split_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"split_meta.json not found at {meta_path}. "
            "Run experiment 07 at least once first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    variants = meta.get("variants", [])

    # Remove old entry if present (idempotent re-run)
    variants = [v for v in variants if v.get("variant") != VARIANT_MULTILABEL_ITERATIVE_PAPER]

    variants.append({
        "variant": VARIANT_MULTILABEL_ITERATIVE_PAPER,
        "label": THESIS_LABELS[VARIANT_MULTILABEL_ITERATIVE_PAPER],
        "description": VARIANT_DESCRIPTIONS[VARIANT_MULTILABEL_ITERATIVE_PAPER],
        "f1_mean": None,  # not yet trained
        "train_file": train_file,
        "eval_file": eval_file,
    })
    meta["variants"] = variants
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Patched: {meta_path}")
    print("\nDone. You can now run cross-comparison with --resume to evaluate only this new variant.")


if __name__ == "__main__":
    main()
