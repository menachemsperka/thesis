# Experiments Directory

This directory contains one Python module per thesis experiment plus shared
utilities.  Each experiment can be run **standalone** or via the project-level
runners (`run_all_experiments.py`, `run_split_comparison.py`).

## Modules

| File | Purpose |
|------|---------|
| `common.py` | Shared helpers: model/proxy configuration, output paths, `write_result_excel`, `write_result_json` |
| `split_io.py` | Save/load pre-computed train/eval sentence splits as JSON; format converters; thesis documentation builder |
| `experiment_01_regular_ner.py` | Baseline NER: fine-tune DictaBERT on the Hebrew NER corpus |
| `experiment_02_imbalance_llm_duplication.py` | Data augmentation via LLM-generated and duplicated sentences |
| `experiment_03_auc_2t.py` | AUC-2T: auxiliary uncertainty classification with entity + BIO heads |
| `experiment_04_auc_cascaded_pipeline.py` | Three-step cascaded NER pipeline (entity → BIO → type) |
| `experiment_05_auc_cascaded_pipeline_step3_consistency.py` | Experiment 04 with Step-3 B/I entity-type reconciliation |
| `experiment_06_fusion_regular_and_cascaded.py` | Fuses experiment 01 and 04 predictions via confidence arbitration |
| `experiment_07_sentence_split_strategy.py` | Compares 8 sentence-split strategies and saves the best splits |
| `experiment_08_llm_augmentation.py` | LLM mask-filling augmentation for rare labels (see `experiment_08_README.md`) |

## Pre-Split Mechanism

Experiments 03–06 support **pre-computed train/eval splits** produced by
experiment 07.  When the environment variables below are set, the experiment
will use the specified JSON files instead of performing its own random split:

```
THESIS_PRESPLIT_TRAIN_JSON=path/to/train.json
THESIS_PRESPLIT_EVAL_JSON=path/to/eval.json
```

The JSON files contain lists of sentence dicts:
```json
[
  {"text": "token1 token2 ...", "labels": ["B-PER", "I-PER", "O", ...]},
  ...
]
```

`split_io.py` provides helpers to convert these into:
- **DataFrame rows** (for exp03 which operates on pandas DataFrames).
- **Cascaded-pipeline format** dicts with `tokens`, `bio_tags`, `entity_types`
  (for exp04/05 via `auc_cascaded_pipeline.py`).

## Running a Single Experiment

```bash
# Run with a specific seed:
set THESIS_SPLIT_SEED=42
python experiments/experiment_03_auc_2t.py

# Run with automatic multi-seed mode (when THESIS_SPLIT_SEED is not set):
python experiments/experiment_03_auc_2t.py
```

## Running Split Comparison (03–06 vs Exp07 Variants)

Use the project-level runner to compare experiments 03–06 across all available
experiment-07 split variants.

```bash
# Reuse saved split artifacts when available
python run_split_comparison.py --exp07-source auto

# Force experiment 07 rerun to regenerate split artifacts
python run_split_comparison.py --exp07-source rerun

# Require saved split artifacts only
python run_split_comparison.py --exp07-source saved

# Restrict to specific experiments
python run_split_comparison.py --exp07-source auto --experiments 04,05,06
```

Useful environment-variable equivalents:

```
THESIS_EXP07_SOURCE=auto|saved|rerun
THESIS_SPLIT_COMPARISON_EXPERIMENTS=03,04,05,06
```

## Output Convention

Every experiment writes:
1. A **timestamped JSON** to `outputs/expNN/<name>_YYYYMMDD_HHMMSS.json`
2. A **latest.json** symlink/copy.
3. A **timestamped Excel** workbook with metrics, detailed predictions, and a
   `documentation` sheet suitable for academic citation.
