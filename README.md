# Thesis NER Experiments (GitHub-Ready)

This folder is an organized, upload-ready project for running the thesis experiments with a single entry point.

## Included Experiments

| ID | Name | Description |
|----|------|-------------|
| 01 | Regular NER with DictaBERT | Baseline fine-tuning of DictaBERT for Hebrew NER |
| 02 | Imbalance handling | LLM sentence generation and sentence duplication |
| 03 | AUC-2T | Auxiliary Uncertainty Classification with 2-task heads |
| 04 | AUC Cascaded Pipeline | Three-step cascaded NER (entity detection → BIO → type) |
| 05 | Cascaded Step-3 Consistency | Exp 04 with B/I entity-type reconciliation |
| 06 | Fusion | Combines regular NER and cascaded pipeline predictions |
| 06_fusion_normalized | Calibrated Fusion | Exp 06 + confidence normalization (temperature scaling) before arbitration |
| 06_fusion_entropy | Entropy-Weighted Fusion | Exp 06 + inverse-entropy weighting so uncertain predictions count less |
| 06_fusion_learned_weights | Learned-Weights Fusion | Exp 06 + validation-learned global weighting between regular and cascaded confidence |
| 06_fusion_svm | SVM Router Fusion | Exp 06 + train-split linear SVM router for disagreement resolution |
| 06_fusion_ensemble_rules | Ensemble-Rules Fusion | Exp 06 + explicit agreement/confidence-gap arbitration rules |
| 07 | Sentence Split Strategy | Compares 8 train/eval split strategies for rare-label coverage |

## Project Structure

```
├── run_all_experiments.py          # Main runner for experiments 01–06
├── run_split_comparison.py         # Run exp03–06 with baseline vs exp07-best split
├── experiments/
│   ├── common.py                   # Shared utilities (paths, model config, Excel/JSON I/O)
│   ├── split_io.py                 # Save/load pre-computed train/eval sentence splits
│   ├── experiment_01_regular_ner.py
│   ├── experiment_02_imbalance_llm_duplication.py
│   ├── experiment_03_auc_2t.py
│   ├── experiment_04_auc_cascaded_pipeline.py
│   ├── experiment_05_auc_cascaded_pipeline_step3_consistency.py
│   ├── experiment_06_fusion_regular_and_cascaded.py
│   ├── experiment_06_fusion_normalized.py
│   ├── experiment_06_fusion_entropy.py
│   ├── experiment_06_fusion_learned_weights.py
│   ├── experiment_06_fusion_svm.py
│   ├── experiment_06_fusion_ensemble_rules.py
│   └── experiment_07_sentence_split_strategy.py
├── core/
│   ├── th_functions.py             # Token classification, split, training
│   ├── NERtraining.py              # PrepDataSetNERTraining class
│   ├── auc_2t_training.py          # AUC-2T model
│   ├── auc_cascaded_pipeline.py    # Three-step cascaded NER pipeline
│   └── confidence_calibration.py   # Temperature scaling for cross-model confidence normalization
├── data/                           # Dataset CSV files
├── models/                         # Local model checkpoints
└── outputs/                        # All experiment results
    ├── exp01/ .. exp07/            # Per-experiment timestamped outputs
    ├── split_comparison/           # Baseline vs exp07-best comparison
    └── summary.json
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Required Data

Place these CSV files in `data/` (recommended):

- `ner_dataset.csv`
- `ner_training_generated.csv`
- `ner_training_duplicated.csv`

All required files are already copied to `data/` in this project.

## Run All Experiments

```bash
python run_all_experiments.py
```

Each experiment is run **5 times** with different split seeds (`42..46`).

## Data Split Protocol (Documented)

All experiments use sentence-level **statistical stratified** splitting with:

- **Train = 70%**, **Validation = 30%**
- **5 independent random runs** per experiment (`THESIS_SPLIT_SEED=42..46`)
- **Distribution objective**: the training split is selected so non-`O` label frequencies in train are as close as possible to full-dataset frequencies.

How this is implemented:

- Splits are created at the **sentence level** (grouped by sentence `id` / `Sentence #`) to avoid token leakage between train and validation.
- For each non-`O` label $\ell$, target train count is computed as:
	$$
	t_\ell = p \cdot N_\ell
	$$
	where $p=0.7$ and $N_\ell$ is the total count of label $\ell$ in the full dataset.
- Sentences are added greedily to train to minimize deviation from target counts (squared-error objective), with a strong tie-break preference to cover still-missing labels.
- Seeded randomness (`THESIS_SPLIT_SEED`) is used for deterministic tie-breaking and reproducibility.

Practical guarantee:

- Train preserves label distribution **approximately** (best possible under sentence-level constraints).
- Each non-`O` label is kept in train on a **best-effort** basis; exact preservation may be impossible for very rare labels.

Reproducibility controls:

- `THESIS_SPLIT_SEED` controls the split seed for a run.
- `run_all_experiments.py` sets the seed automatically for each of the 5 runs.

Default mode is quiet (`DEBUG=False`): only important progress and final results are printed.

For verbose/debug logs:

```bash
set THESIS_DEBUG=1
python run_all_experiments.py
```

The main script prints:

- short description per experiment
- per-run F1 values and aggregate F1 statistics (mean/best/worst)
- a final summary table

It also saves:

- `outputs/summary.json`
- `outputs/summary_splits.xlsx` (global per-split rows + per-experiment summary)
- timestamped JSON files per experiment in dedicated folders:
	- `outputs/exp01/`
	- `outputs/exp02/`
	- `outputs/exp03/`
	- `outputs/exp04/`
- `outputs/expXX/latest.json` for quick access to the most recent run
- `outputs/expXX/split_runs_latest.xlsx` and `outputs/expXX/split_runs_YYYYMMDD_HHMMSS.xlsx` (one row per split/run)

When running an individual experiment script directly (for example `python experiments/experiment_03_auc_2t.py`) without setting `THESIS_SPLIT_SEED`, it will automatically run multiple splits and write `split_runs_latest.xlsx` in that experiment output folder.

Controls for direct script multi-split mode:

- `THESIS_DIRECT_SPLIT_RUNS` (default: `5`)
- `THESIS_DIRECT_BASE_SEED` (default: `42`)
- If `THESIS_SPLIT_SEED` is explicitly set, the script runs a single split only.

## Excel Outputs (Scores + Detailed Predictions)

Each experiment now also writes timestamped Excel output into its experiment folder (`outputs/expXX/`) and keeps a latest copy:

- `outputs/exp01/regular_ner_results_*.xlsx` and `regular_ner_results_latest.xlsx`
- `outputs/exp02/imbalance_llm_duplication_results_*.xlsx` and `imbalance_llm_duplication_results_latest.xlsx`
- `outputs/exp03/auc_2t_results_*.xlsx` and `auc_2t_results_latest.xlsx`
- `outputs/exp04/cascaded_pipeline_results_*.xlsx`

Excel sheet structure:

- `metrics` — experiment scores (F1/precision/recall and run metadata)
- `detailed_results` — detailed prediction rows with true labels and predicted labels

Per-experiment details:

- **exp01**: single-run baseline scores + sentence-level true vs predicted labels
- **exp02**: per-variant scores (`ner_dataset.csv`, `ner_training_generated.csv`, `ner_training_duplicated.csv`) + detailed rows tagged by `dataset_name`
- **exp03**: AUC-2T scores + sentence-level true BIO labels vs predicted BIO labels
- **exp04**: cascaded pipeline metrics + token-level detailed results exported by the core pipeline

## Notes

- Some experiments are computationally expensive and may take significant time.
- All experiments write timestamped Excel files with metrics and detailed prediction outputs.
- The cascaded script uses `data/ner_dataset.csv` by default.

## Experiment 07 — Sentence Split Strategy

Experiment 07 evaluates 8 different train/eval split strategies and identifies
the one that maximises entity-level F1 by ensuring rare labels are well
represented in the training set.

```bash
python experiments/experiment_07_sentence_split_strategy.py
```

**Key output:** after running, the baseline and best-variant train/eval sentence
lists are saved to `outputs/exp07/splits/` as JSON files:

| File | Description |
|------|-------------|
| `baseline_train.json` | Training sentences from simple random split |
| `baseline_eval.json`  | Eval sentences from simple random split |
| `best_train.json`     | Training sentences from the best split strategy |
| `best_eval.json`      | Eval sentences from the best split strategy |
| `split_meta.json`     | Metadata: which variant was best, F1 mean, seed |

These files are consumed by `run_split_comparison.py`.

## Split Comparison — Impact on Experiments 03–06

You can measure split-strategy impact by running each downstream experiment
across **all experiment-07 split variants** (3 variants).

```bash
# Reuse saved split artifacts if valid; rerun exp07 only if needed
python run_split_comparison.py --exp07-source auto

# Force rerun experiment 07 before comparison
python run_split_comparison.py --exp07-source rerun

# Use saved split artifacts only; fail if missing/incomplete
python run_split_comparison.py --exp07-source saved

# Run subset of experiments only
python run_split_comparison.py --exp07-source auto --experiments 04,05,06
```

Results are saved to `outputs/split_comparison/`:
- `split_comparison_latest.xlsx` — Excel with sheets: results, summary_pivot,
  variant_summary, deltas, experiment_details, and documentation.
- `split_comparison_latest.json` — machine-readable equivalent.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THESIS_EXP07_SOURCE` | `auto` | Split artifact policy for exp07: `auto`, `saved`, `rerun` |
| `THESIS_SPLIT_COMPARISON_EXPERIMENTS` | `03,04,05,06` | Which experiments to include |
| `THESIS_SPLIT_SEED` | `42` | Random seed for splits |
| `THESIS_NUM_EPOCHS` | `3` | Training epochs |
| `THESIS_MODEL_NAME` | auto-detected | Model path or HuggingFace ID |
| `THESIS_DEBUG` | `0` | Set to `1` for verbose output |
| `THESIS_PRESPLIT_TRAIN_JSON` | — | Path to pre-split train JSON (set automatically by runner) |
| `THESIS_PRESPLIT_EVAL_JSON` | — | Path to pre-split eval JSON (set automatically by runner) |

## Experiment 06_fusion_normalized — Confidence-Normalized Fusion

This experiment is a calibrated version of fusion (Exp 06). It was added to
solve a known issue: the two confidence sources in fusion are not naturally on
the same scale.

- Regular NER confidence: max softmax probability.
- Cascaded confidence: `(1 - entity_prob)` for `O`, otherwise `entity_prob * bio_prob`.

Because these are produced by different models and different objectives, direct
comparison can cause suboptimal arbitration. Exp `06_fusion_normalized` learns a
temperature per source and applies calibration before choosing between regular
and cascaded predictions.

Core idea:

$$
p_{calibrated} = \sigma\left(\frac{\text{logit}(p)}{T}\right)
$$

Where $T$ is learned from prediction correctness (maximum-likelihood calibration).

Run examples:

```bash
# Run only calibrated fusion in cross-comparison
python run_cross_data_model_comparison.py --experiments 06_fusion_normalized

# Compare original fusion vs calibrated fusion
python run_cross_data_model_comparison.py --experiments 06,06_fusion_normalized
```

Outputs for the new experiment are written under:

- `outputs/exp06_fusion_normalized/`

Detailed design and formulas are documented in:

- `CALIBRATION_README.md`

## Experiment 04 — AUC Cascaded Pipeline

This experiment implements NER as a three-stage cascaded pipeline instead of a single token-classification step.

- Step 1: detect whether each token is part of an entity or `O`.
- Step 2: assign the BIO boundary tag.
- Step 3: predict the entity type.

The main idea is to decompose the full NER decision into simpler subtasks. This can make boundary decisions and entity typing more explicit, and it also exposes intermediate outputs that are useful for analysis and downstream fusion.

Outputs for this experiment are written under:

- `outputs/exp04/`

## Fusion Results Summary

Using the ready-results comparison export (`outputs/cross_comparison/cross_comparison_ready_all_methods_latest.json` / `.xlsx`), the current fusion variants compare as follows against the two relevant baselines:

- Regular NER: Exp `01`
- Cascaded Pipeline: Exp `04`

| Fusion method | vs Exp01 | vs Exp04 | Beats both Exp01 and Exp04 | Mean delta vs Exp01 | Mean delta vs Exp04 | Interpretation |
|-------|-------|-------|-------|-------|-------|-------|
| `06` | 7 wins / 5 losses | 7 wins / 5 losses | 3 / 12 | +0.0026 | +0.0202 | Baseline fusion helps sometimes, but is not consistently better than its source models |
| `06_fusion_normalized` | 6 wins / 6 losses | 6 wins / 6 losses | 4 / 12 | +0.0000 | +0.0176 | Confidence calibration alone does not reliably solve fusion underperformance |
| `06_fusion_entropy` | 9 wins / 2 losses / 1 tie | 7 wins / 5 losses | 7 / 12 | +0.0101 | +0.0277 | Better than raw fusion; uncertainty-weighting helps on many disagreements |
| `06_fusion_learned_weights` | 11 wins / 1 loss | 10 wins / 2 losses | 10 / 12 | +0.0327 | +0.0503 | Best overall variant; strongest and most consistent improvement |
| `06_fusion_ensemble_rules` | 9 wins / 2 losses / 1 tie | 7 wins / 5 losses | 6 / 12 | +0.0180 | +0.0356 | Interpretable and useful, but still less consistent than learned weights |

Current conclusion: `06_fusion_learned_weights` is the best fusion method in this repository. It is the only variant that is consistently strong against both source systems, beating both baselines in 10 of 12 shared model-condition comparisons.

Why are there still 2 losses? Fusion is not an oracle that always knows which source model is correct. `06_fusion_learned_weights` learns one global weighting between regular and cascaded confidence, so it improves arbitration on average, but some disagreement cases remain inherently ambiguous. In those cases the wrong model can still have the stronger confidence signal, and a small number of token-level mistakes can reduce entity-level F1 even when the fusion rule is generally better overall.

## Excel Workbook Structure

All experiment Excel files share a common structure:

| Sheet | Content |
|-------|---------|
| `metrics` | Primary scores: F1, precision, recall, accuracy |
| `detailed_results` | Per-seed or per-token prediction detail |
| `documentation` | Key–value pairs explaining the experiment, dataset, and metrics for academic citation |

Experiment 07's workbook additionally includes:
- `score_summary_numeric` — mean/std/CI95 for each variant
- `score_ranking_f1` — variants ranked by F1
- `score_deltas_vs_baseline` — paired deltas
- `training_label_count` — before/after label frequency table with percentages and deltas
