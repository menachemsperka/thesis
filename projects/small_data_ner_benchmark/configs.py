"""Benchmark definitions: dataset × encoder pairings and split protocol constants."""

from __future__ import annotations

from dataclasses import dataclass

# Re-use exp07 variant keys (same as run_cross_data_model_comparison paper subset).
SPLIT_VARIANT_RANDOM = "before_exp01_baseline"
SPLIT_VARIANT_PAPER = "after_multilabel_iterative_paper"
SPLIT_VARIANTS: tuple[str, ...] = (SPLIT_VARIANT_RANDOM, SPLIT_VARIANT_PAPER)

SPLIT_RATIO = 0.7
SMALL_POOL_SIZE = 300
REGIME_SMALL = "small_300"
REGIME_FULL = "full"
REGIMES: tuple[str, ...] = (REGIME_SMALL, REGIME_FULL)

DEFAULT_NUM_SEEDS = 5
DEFAULT_SEED_START = 42
DEFAULT_BASE_SEED = 42

EXPERIMENT_IDS: tuple[str, ...] = ("10_regular", "10_cascade", "10_svm_ready")


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str
    display_name: str
    dataset_key: str
    model_id: str
    model_short: str


BENCHMARKS: tuple[BenchmarkConfig, ...] = (
    BenchmarkConfig(
        key="conll2003_bert",
        display_name="CoNLL-2003 + bert-base-uncased",
        dataset_key="conll2003",
        model_id="bert-base-uncased",
        model_short="bert-base-uncased",
    ),
    BenchmarkConfig(
        key="nemo_dictabert",
        display_name="NEMO + DictaBERT",
        dataset_key="nemo",
        model_id="dicta-il/dictabert",
        model_short="dictabert",
    ),
    BenchmarkConfig(
        key="bc5cdr_pubmedbert",
        display_name="BC5CDR + PubMedBERT",
        dataset_key="bc5cdr",
        model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        model_short="PubMedBERT",
    ),
)
