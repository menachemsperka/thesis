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
| `experiment_06_fusion_normalized.py` | Calibrated fusion: applies temperature scaling to both confidence sources before arbitration |
| `experiment_06_fusion_entropy.py` | Entropy-weighted fusion: downweights uncertain predictions before arbitration |
| `experiment_06_fusion_learned_weights.py` | Learned-weights fusion: learns a global regular-vs-cascade weight on validation data |
| `experiment_06_fusion_svm.py` | SVM-router fusion: keeps agreements and learns disagreement routing on the train split |
| `experiment_06_fusion_ensemble_rules.py` | Ensemble-rules fusion: uses agreement and confidence-gap rules instead of raw confidence only |
| `experiment_06_fusion_regular_and_exp05_ready.py` | No-retraining fusion using ready outputs: Regular side (from Exp06 detailed output) + Exp05 cascaded predictions (raw confidence arbitration) |
| `experiment_06_fusion_exp05_learned_weights_ready.py` | No-retraining fusion using ready outputs: Regular side + Exp05, with learned global arbitration weights |
| `fusion_ready_sources.py` | Shared loader module for ready-results fusion: loads Exp01 + Exp04 outputs, merges, provides generic entry point |
| `experiment_05_ready.py` | Exp05 without retraining: loads Exp04 output and applies B/I entity-type consistency post-processing |
| `experiment_06_fusion_ready.py` | Ready base fusion: confidence comparison from Exp01 + Exp04 — no retraining |
| `experiment_06_fusion_normalized_ready.py` | Ready calibrated fusion: temperature scaling from Exp01 + Exp04 — no retraining |
| `experiment_06_fusion_entropy_ready.py` | Ready entropy-weighted fusion from Exp01 + Exp04 — no retraining |
| `experiment_06_fusion_learned_weights_ready.py` | Ready learned-weights fusion from Exp01 + Exp04 — no retraining |
| `experiment_06_fusion_ensemble_rules_ready.py` | Ready ensemble-rules fusion from Exp01 + Exp04 — no retraining |
| `experiment_06_fusion_svm_ready.py` | Ready SVM-router fusion from Exp01 + Exp04 — no retraining |
| `experiment_07_sentence_split_strategy.py` | Compares 8 sentence-split strategies and saves the best splits |
| `experiment_08_llm_augmentation.py` | LLM mask-filling augmentation for rare labels (see `experiment_08_README.md`) |

Related core module:

| File | Purpose |
|------|---------|
| `../core/confidence_calibration.py` | Learns and applies per-model temperature scaling for confidence normalization |
| `../core/fusion_strategies.py` | Shared prototypes for alternative fusion ideas (entropy, learned weighting, rule-based arbitration) |

## Experiment 04 in One Paragraph

`experiment_04_auc_cascaded_pipeline.py` runs NER as a staged pipeline rather than one end-to-end label prediction. It first predicts whether a token belongs to an entity, then predicts the BIO boundary label, and finally predicts the entity type. This decomposition makes the model easier to inspect and gives later stages a more focused decision problem, which is why Exp04 is also the cascaded source model used by the fusion experiments.

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

## Calibrated Fusion Notes (Exp 06_fusion_normalized)

`experiment_06_fusion_normalized.py` is intended for cases where fusion underperforms
regular NER because regular and cascaded confidences are on mismatched scales.

What changes vs `experiment_06_fusion_regular_and_cascaded.py`:

- Learns one temperature for regular confidence and one for cascaded confidence.
- Applies calibrated confidence before selecting which model wins disagreements.
- Exports calibration metadata (temperature and ECE before/after) in output metrics.

Typical usage:

```bash
# Single experiment
python experiments/experiment_06_fusion_normalized.py

# Compare both fusion variants in cross-comparison
python run_cross_data_model_comparison.py --experiments 06,06_fusion_normalized
```

## Fusion Results Snapshot

The main practical comparison for fusion is whether a fusion method beats both of its source models:

- Exp `01` Regular NER
- Exp `04` AUC Cascaded Pipeline

From `outputs/cross_comparison/cross_comparison_ready_all_methods_latest.json`, the fusion variants currently rank as follows:

| Method | Beats Exp01 | Beats Exp04 | Beats both | Mean delta vs Exp01 | Mean delta vs Exp04 |
|------|------|------|------|------|------|
| `06` | 7 / 12 | 7 / 12 | 3 / 12 | +0.0026 | +0.0202 |
| `06_fusion_normalized` | 6 / 12 | 6 / 12 | 4 / 12 | +0.0000 | +0.0176 |
| `06_fusion_entropy` | 9 / 12 (+1 tie) | 7 / 12 | 7 / 12 | +0.0101 | +0.0277 |
| `06_fusion_learned_weights` | 11 / 12 | 10 / 12 | 10 / 12 | +0.0327 | +0.0503 |
| `06_fusion_ensemble_rules` | 9 / 12 (+1 tie) | 7 / 12 | 6 / 12 | +0.0180 | +0.0356 |

Operational takeaway: `experiment_06_fusion_learned_weights.py` is the best current fusion implementation. It gives the strongest average gain and the highest rate of beating both source models on the same condition.

The remaining losses do not mean the method failed conceptually. Fusion only affects disagreement cases, and those are typically the hardest boundary or rare-label decisions. A single global weight cannot perfectly model every context, so a few disagreements are still resolved in favor of the wrong model. As a result, learned-weight fusion is clearly the best average strategy, but it is still an informed selector rather than an oracle upper bound.

## Exp05 Fusion From Ready Results (No Retraining)

Two additional scripts fuse Regular NER with **Exp05** (cascaded + Step3 consistency) directly from already-saved artifacts:

- `experiment_06_fusion_regular_and_exp05_ready.py`
- `experiment_06_fusion_exp05_learned_weights_ready.py`

Important behavior:

- These scripts **do not retrain** any model.
- They read token-level outputs from existing files:
  - Regular side: from Exp06 `detailed_results` (contains `regular_pred_label`, `regular_prob`)
  - Cascaded side: from Exp05 `detailed_results` (`pred_bio`, `pred_etype`, `entity_prob`, `bio_prob`)

Default file resolution:

- `outputs/exp06/latest.json` → `metrics_file`
- `outputs/exp05/latest.json` → `metrics_file`

Optional overrides:

```bash
set THESIS_READY_EXP06_METRICS_XLSX=path/to/exp06_metrics.xlsx
set THESIS_READY_EXP05_METRICS_XLSX=path/to/exp05_metrics.xlsx
```

Run commands:

```bash
# Raw confidence arbitration (Regular vs Exp05)
python experiments/experiment_06_fusion_regular_and_exp05_ready.py

# Learned global weights arbitration (Regular vs Exp05)
python experiments/experiment_06_fusion_exp05_learned_weights_ready.py
```

Outputs are written to:

- `outputs/exp06_fusion_exp05_ready/`
- `outputs/exp06_fusion_exp05_learned_ready/`

## Output Convention

Every experiment writes:
1. A **timestamped JSON** to `outputs/expNN/<name>_YYYYMMDD_HHMMSS.json`
2. A **latest.json** symlink/copy.
3. A **timestamped Excel** workbook with metrics, detailed predictions, and a
   `documentation` sheet suitable for academic citation.
