"""LLM mask-filling augmentation (experiment 08) for benchmark train splits."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from split_io import save_split  # noqa: E402

from splits import _load_sentence_json, load_split_meta  # noqa: E402


def _import_exp08():
    return importlib.import_module("experiment_08_llm_augmentation")


def _load_corpus_dataframe(corpus_csv: Path) -> pd.DataFrame:
    os.environ.setdefault("THESIS_SKIP_HEBREW_TEXT_VALIDATION", "1")
    core_dir = PROJECT_ROOT / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    from hebrew_text_io import read_ner_dataset_csv

    df, _enc = read_ner_dataset_csv(corpus_csv)
    return df


def _augmentation_multiplier() -> int:
    raw = (os.environ.get("THESIS_EXP08_MULTIPLIER") or "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _resolve_augmentation_model(ner_model_id: str) -> str:
    mod = _import_exp08()
    return mod._resolve_augmentation_model_name(ner_model_id)


def _augmented_train_rel(regime: str, variant: str, seed: int) -> str:
    safe = variant.replace(" ", "_")
    return f"{regime}/{safe}_seed{seed}_augmented_train.json"


def augmentation_covers_meta(meta: dict[str, Any], data_root: Path) -> bool:
    """True if every seed entry has a materialized augmented train file."""
    splits_root = data_root / "splits"
    for _regime, block in (meta.get("regimes") or {}).items():
        for vm in block.get("variants") or []:
            variant = str(vm.get("variant", ""))
            for seed_str, files in (vm.get("seed_files") or {}).items():
                rel = files.get("augmented_train_file")
                if not rel:
                    rel = _augmented_train_rel(str(_regime), variant, int(seed_str))
                path = splits_root / str(rel).replace("\\", "/")
                if not path.is_file():
                    return False
    return True


def prepare_augmented_train_splits(
    *,
    data_root: Path,
    ner_model_id: str,
    benchmark_display: str,
    force: bool = False,
    log_fn: Any = print,
) -> dict[str, Any]:
    """Run exp08 mask-fill on each baseline train split; eval JSON unchanged."""
    os.environ.setdefault("THESIS_SKIP_HEBREW_TEXT_VALIDATION", "1")
    meta = load_split_meta(data_root)
    corpus_csv = Path(meta["corpus_csv"])
    if not corpus_csv.is_file():
        raise FileNotFoundError(f"Missing corpus CSV: {corpus_csv}")

    mod = _import_exp08()
    augment_fn = mod._augment_training_data
    data_df = _load_corpus_dataframe(corpus_csv)
    multiplier = _augmentation_multiplier()
    aug_model = _resolve_augmentation_model(ner_model_id)
    splits_root = data_root / "splits"

    jobs = 0
    skipped = 0
    generated_jobs = 0
    t0 = time.time()

    for regime, block in (meta.get("regimes") or {}).items():
        for vm in block.get("variants") or []:
            variant = str(vm.get("variant", ""))
            label = str(vm.get("label", variant))
            seed_files = vm.get("seed_files") or {}
            for seed_str, files in seed_files.items():
                seed = int(seed_str)
                jobs += 1
                train_rel = str(files.get("train_file", "")).replace("\\", "/")
                if not train_rel:
                    continue
                train_path = splits_root / train_rel
                aug_rel = _augmented_train_rel(str(regime), variant, seed)
                aug_path = splits_root / aug_rel

                if not force and aug_path.is_file():
                    skipped += 1
                    files["augmented_train_file"] = aug_rel
                    continue

                log_fn(
                    f"  Augment train | {benchmark_display} | {regime} / {label} [s{seed}] "
                    f"(model={aug_model}, x{multiplier})"
                )
                train_sentences = _load_sentence_json(train_path)
                generated, _log_df = augment_fn(
                    train_sentences,
                    data_df,
                    ner_model_id,
                    multiplier=multiplier,
                    rng_seed=seed,
                    augmentation_model_name=aug_model,
                )
                augmented_train = train_sentences + list(generated)
                save_split(augmented_train, aug_path)
                files["augmented_train_file"] = aug_rel
                files["augmentation"] = {
                    "method": "llm_mask_filling_exp08",
                    "multiplier": multiplier,
                    "augmentation_model": aug_model,
                    "ner_model_id": ner_model_id,
                    "baseline_train_sentences": len(train_sentences),
                    "generated_sentences": len(generated),
                    "augmented_train_sentences": len(augmented_train),
                    "updated_at": datetime.now().isoformat(),
                }
                generated_jobs += 1

    meta.setdefault("augmentation", {})
    meta["augmentation"].update(
        {
            "method": "llm_mask_filling_exp08",
            "multiplier": multiplier,
            "augmentation_model": aug_model,
            "prepared_at": datetime.now().isoformat(),
        }
    )
    meta_path = data_root / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    summary = {
        "jobs": jobs,
        "skipped": skipped,
        "generated": generated_jobs,
        "elapsed_seconds": round(elapsed, 1),
        "augmentation_model": aug_model,
        "multiplier": multiplier,
    }
    log_fn(
        f"Augmentation done for {benchmark_display}: "
        f"{generated_jobs} generated, {skipped} skipped, {jobs} total ({elapsed:.1f}s)"
    )
    return summary
