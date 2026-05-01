"""
Generate only the missing exp07+augmentation variant for
`after_multilabel_stratified` and patch exp07_augmented split metadata.

Usage:
    python generate_missing_exp07aug_multilabel_variant.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CORE_DIR = PROJECT_ROOT / "core"

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from common import configure_model_environment  # type: ignore
from NERtraining import PrepDataSetNERTraining  # type: ignore
from split_io import load_split, save_split  # type: ignore
import experiment_08_llm_augmentation as exp08  # type: ignore


EXP07_SPLITS_DIR = PROJECT_ROOT / "outputs" / "exp07" / "splits"
EXP07_AUG_SPLITS_DIR = PROJECT_ROOT / "outputs" / "exp07_augmented" / "splits"
TARGET_VARIANT = "after_multilabel_stratified"


def main() -> None:
    meta07_path = EXP07_SPLITS_DIR / "split_meta.json"
    meta07_aug_path = EXP07_AUG_SPLITS_DIR / "split_meta.json"
    if not meta07_path.exists():
        raise FileNotFoundError(f"Missing exp07 split meta: {meta07_path}")

    meta07 = json.loads(meta07_path.read_text(encoding="utf-8"))

    # Ensure exp07+aug dir and meta exist
    EXP07_AUG_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    if meta07_aug_path.exists():
        meta07_aug = json.loads(meta07_aug_path.read_text(encoding="utf-8"))
    else:
        meta07_aug = {
            "description": "Exp07 split variants with exp08-style LLM augmentation applied to training data",
            "augmentation_multiplier": None,
            "augmentation_model": "",
            "ner_model_for_resolution": "",
            "source_exp07_meta": str(meta07_path),
            "variants": [],
        }

    existing = {str(v.get("variant", "")): v for v in meta07_aug.get("variants", [])}
    if TARGET_VARIANT in existing:
        print(f"Already present: {TARGET_VARIANT} (nothing to do)")
        return

    vm = next((v for v in meta07.get("variants", []) if v.get("variant") == TARGET_VARIANT), None)
    if vm is None:
        raise RuntimeError(f"Target variant not found in exp07 meta: {TARGET_VARIANT}")

    train_path = EXP07_SPLITS_DIR / str(vm.get("train_file"))
    eval_path = EXP07_SPLITS_DIR / str(vm.get("eval_file"))
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Missing exp07 train/eval split files for target variant")

    # Load base data for augmentation token lookup.
    worker = PrepDataSetNERTraining()
    data_df = worker.load_and_prepare_data(str(PROJECT_ROOT / "data" / "ner_dataset.csv"))

    model_name, _ = configure_model_environment()

    multiplier_raw = (os.environ.get("THESIS_EXP08_MULTIPLIER") or "3").strip()
    try:
        multiplier = max(1, int(multiplier_raw))
    except ValueError:
        multiplier = 3

    train_sentences = load_split(train_path)
    eval_sentences = load_split(eval_path)

    generated_sents, _ = exp08._augment_training_data(
        train_sentences,
        data_df,
        model_name,
        multiplier=multiplier,
        rng_seed=42,
    )
    augmented_train = train_sentences + generated_sents

    aug_train_file = f"{TARGET_VARIANT}_augmented_train.json"
    aug_eval_file = f"{TARGET_VARIANT}_eval.json"

    save_split(augmented_train, EXP07_AUG_SPLITS_DIR / aug_train_file)
    save_split(eval_sentences, EXP07_AUG_SPLITS_DIR / aug_eval_file)

    variants = [v for v in meta07_aug.get("variants", []) if v.get("variant") != TARGET_VARIANT]
    variants.append(
        {
            "variant": TARGET_VARIANT,
            "label": vm.get("label", TARGET_VARIANT),
            "description": f"{vm.get('description', vm.get('label', TARGET_VARIANT))} + LLM mask-fill augmentation",
            "original_train_sentences": len(train_sentences),
            "generated_sentences": len(generated_sents),
            "augmented_train_sentences": len(augmented_train),
            "train_file": aug_train_file,
            "eval_file": aug_eval_file,
        }
    )

    # Keep order consistent with exp07 variants order where possible.
    order = {str(v.get("variant", "")): i for i, v in enumerate(meta07.get("variants", []))}
    variants.sort(key=lambda x: order.get(str(x.get("variant", "")), 10**9))

    meta07_aug["variants"] = variants
    meta07_aug["augmentation_multiplier"] = multiplier
    meta07_aug["ner_model_for_resolution"] = model_name
    meta07_aug_path.write_text(json.dumps(meta07_aug, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Added missing exp07+aug variant:")
    print(f"  variant: {TARGET_VARIANT}")
    print(f"  train file: {aug_train_file}")
    print(f"  eval file: {aug_eval_file}")
    print(f"  generated sentences: {len(generated_sents)}")


if __name__ == "__main__":
    main()
