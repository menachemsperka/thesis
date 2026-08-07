"""Exp07-style random + paper multilabel stratified splits on benchmark corpora."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
_benchmark_s = str(BENCHMARK_ROOT)
if _benchmark_s not in sys.path:
    sys.path.insert(0, _benchmark_s)
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from configs import (
    REGIME_FULL,
    REGIME_SMALL,
    REGIMES,
    SMALL_POOL_SIZE,
    SPLIT_RATIO,
    SPLIT_VARIANTS,
)
from corpus_loaders import load_benchmark_splits, sample_sentence_pool, write_corpus_csv

from exp07_split_artifacts import (  # noqa: E402
    BEFORE_VARIANT,
    SPLIT_FNS,
    THESIS_LABELS,
    VARIANT_DESCRIPTIONS,
)
from split_io import save_split  # noqa: E402
from split_stats import summarize_sentences  # noqa: E402


def _load_sentence_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{"text": str(x.get("text", "")), "labels": list(x.get("labels", []))} for x in raw]


def prepare_benchmark_assets(
    *,
    dataset_key: str,
    data_root: Path,
    cache_dir: Path,
    pool_seed: int = 42,
) -> dict[str, Any]:
    """Download corpus, write corpus.csv, sentence pools, and all split JSON files."""
    os.environ.setdefault("THESIS_SKIP_HEBREW_TEXT_VALIDATION", "1")
    data_root.mkdir(parents=True, exist_ok=True)

    splits = load_benchmark_splits(dataset_key, cache_dir)
    train = splits["train"]
    test = splits.get("test") or splits.get("validation") or train
    all_for_labels = list(train)
    if "validation" in splits and splits["validation"] is not test:
        all_for_labels.extend(splits["validation"])
    all_for_labels.extend(test)

    corpus_csv = write_corpus_csv(all_for_labels, data_root / "corpus.csv")
    pool = sample_sentence_pool(train, SMALL_POOL_SIZE, pool_seed)
    pool_path = data_root / "pool_300.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    full_pool_path = data_root / "sentences_full_train.json"
    full_pool_path.write_text(
        json.dumps([{"text": s["text"], "labels": s["labels"]} for s in train], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "corpus_csv": corpus_csv,
        "pool_path": pool_path,
        "full_pool_path": full_pool_path,
        "n_train_official": len(train),
        "n_test_official": len(test),
    }


def generate_regime_splits(
    *,
    data_root: Path,
    regime: str,
    seeds: list[int],
    split_ratio: float = SPLIT_RATIO,
) -> list[dict[str, Any]]:
    """Write train/eval JSON for each (variant × seed) on the regime sentence pool."""
    if regime == REGIME_SMALL:
        pool = _load_sentence_json(data_root / "pool_300.json")
        if len(pool) != SMALL_POOL_SIZE:
            raise ValueError(f"Expected {SMALL_POOL_SIZE} sentences in pool, got {len(pool)}")
        source_sentences = pool
    elif regime == REGIME_FULL:
        source_sentences = _load_sentence_json(data_root / "sentences_full_train.json")
    else:
        raise ValueError(f"Unknown regime: {regime}")

    splits_root = data_root / "splits" / regime
    variant_meta: list[dict[str, Any]] = []

    for variant in SPLIT_VARIANTS:
        fn = SPLIT_FNS.get(variant)
        if fn is None:
            raise KeyError(f"Unknown split variant: {variant}")
        seed_files: dict[str, dict[str, str]] = {}
        for seed in seeds:
            train_sents, eval_sents = fn(source_sentences, split_ratio, seed)
            safe = variant.replace(" ", "_")
            train_name = f"{safe}_seed{seed}_train.json"
            eval_name = f"{safe}_seed{seed}_eval.json"
            save_split(train_sents, splits_root / train_name)
            save_split(eval_sents, splits_root / eval_name)
            seed_files[str(seed)] = {
                "train_file": f"{regime}/{train_name}",
                "eval_file": f"{regime}/{eval_name}",
                "pool_n_sentences": len(source_sentences),
                "split_ratio": split_ratio,
                "stats": {
                    "train": summarize_sentences(train_sents),
                    "eval": summarize_sentences(eval_sents),
                },
            }

        variant_meta.append(
            {
                "variant": variant,
                "label": THESIS_LABELS.get(variant, variant),
                "description": VARIANT_DESCRIPTIONS.get(variant, variant),
                "seed_files": seed_files,
                "is_baseline": variant == BEFORE_VARIANT,
            }
        )

    return variant_meta


def write_split_meta(
    *,
    data_root: Path,
    benchmark_key: str,
    dataset_key: str,
    corpus_csv: Path,
    regimes: list[str],
    seeds: list[int],
    regime_variants: dict[str, list[dict[str, Any]]],
) -> Path:
    meta = {
        "benchmark_key": benchmark_key,
        "dataset_key": dataset_key,
        "corpus_csv": str(corpus_csv),
        "split_ratio": SPLIT_RATIO,
        "small_pool_size": SMALL_POOL_SIZE,
        "baseline_variant": BEFORE_VARIANT,
        "paper_variant": "after_multilabel_iterative_paper",
        "seeds": seeds,
        "regimes": {},
    }
    for regime in regimes:
        meta["regimes"][regime] = {"variants": regime_variants.get(regime, [])}
    meta_path = data_root / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def prepare_all_splits(
    *,
    benchmark_key: str,
    dataset_key: str,
    data_root: Path,
    cache_dir: Path,
    seeds: list[int],
    pool_seed: int,
    regimes: list[str] | None = None,
) -> Path:
    use_regimes = list(regimes or REGIMES)
    assets = prepare_benchmark_assets(
        dataset_key=dataset_key,
        data_root=data_root,
        cache_dir=cache_dir,
        pool_seed=pool_seed,
    )
    regime_variants: dict[str, list[dict[str, Any]]] = {}
    for regime in use_regimes:
        regime_variants[regime] = generate_regime_splits(
            data_root=data_root,
            regime=regime,
            seeds=seeds,
        )
    return write_split_meta(
        data_root=data_root,
        benchmark_key=benchmark_key,
        dataset_key=dataset_key,
        corpus_csv=assets["corpus_csv"],
        regimes=use_regimes,
        seeds=seeds,
        regime_variants=regime_variants,
    )


def load_split_meta(data_root: Path) -> dict[str, Any]:
    path = data_root / "split_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split meta: {path}. Run --prepare-only first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_conditions(
    *,
    cfg_key: str,
    cfg_display: str,
    data_root: Path,
    regimes: list[str] | None = None,
    variant_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
) -> list[dict[str, Any]]:
    meta = load_split_meta(data_root)
    corpus_csv = Path(meta["corpus_csv"])
    use_regimes = regimes or list(meta.get("regimes", {}).keys())
    conditions: list[dict[str, Any]] = []
    splits_dir = data_root / "splits"

    for regime in use_regimes:
        block = meta.get("regimes", {}).get(regime, {})
        for vm in block.get("variants", []):
            variant = str(vm.get("variant", ""))
            if variant_filter and variant not in variant_filter:
                continue
            seed_files = vm.get("seed_files") or {}
            for seed_str, files in seed_files.items():
                seed = int(seed_str)
                if seed_filter and seed not in seed_filter:
                    continue
                train_rel = files.get("train_file")
                eval_rel = files.get("eval_file")
                if not train_rel or not eval_rel:
                    continue
                train_path = data_root / "splits" / str(train_rel).replace("\\", "/")
                eval_path = data_root / "splits" / str(eval_rel).replace("\\", "/")
                label = vm.get("label", variant)
                cond_key = f"{cfg_key}__{regime}__{variant}__seed{seed}"
                conditions.append(
                    {
                        "source": regime,
                        "benchmark_key": cfg_key,
                        "benchmark_label": cfg_display,
                        "corpus_csv": corpus_csv,
                        "key": cond_key,
                        "base_condition_key": f"{cfg_key}__{regime}__{variant}",
                        "base_condition_short": f"{regime} / {label}",
                        "variant": variant,
                        "regime": regime,
                        "label": f"[{cfg_display}] [{regime}] {label} [seed {seed}]",
                        "short_label": f"{regime} / {label} [s{seed}]",
                        "description": str(vm.get("description", label)),
                        "train_path": train_path,
                        "eval_path": eval_path,
                        "seed": seed,
                        "is_baseline": bool(vm.get("is_baseline", variant == BEFORE_VARIANT)),
                    }
                )
    return conditions
