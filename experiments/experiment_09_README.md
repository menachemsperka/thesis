# Experiment 09 — Cross-Data × Multi-Model Comparison

## Executive Summary

This experiment systematically evaluates **two Hebrew BERT models** —
**DictaBERT** (`dicta-il/dictabert`) and **BEREL 3.0** (`dicta-il/BEREL_3.0`) —
across **all training-data conditions** produced by Experiment 07 (sentence-split
strategies) and Experiment 08 (LLM data augmentation).  Each data condition is
fed into four downstream NER architectures (Experiments 03–06), yielding a total
of **2 models × 10 data conditions × 4 experiments = 80 trained runs**.

### Key Results (placeholder — fill after running)

| Metric | DictaBERT | BEREL 3.0 |
|--------|-----------|-----------|
| Best overall F1 (any condition) | `<F1_DICTABERT_BEST>` | `<F1_BEREL_BEST>` |
| Best exp07 split variant | `<VARIANT_DICTABERT>` | `<VARIANT_BEREL>` |
| Exp08 augmentation Δ F1 (avg) | `<DELTA_DICTABERT_EXP08>` | `<DELTA_BEREL_EXP08>` |
| Head-to-head wins (out of 40) | `<WINS_DICTABERT>` | `<WINS_BEREL>` |

| Downstream Experiment | Best Model | Best Data Condition | F1 |
|----------------------|------------|--------------------|----|
| Exp03 – AUC-2T | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp04 – AUC Cascaded Pipeline | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp05 – AUC Cascaded + Step3 | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp06 – Fusion (Regular + Cascaded) | `<MODEL>` | `<CONDITION>` | `<F1>` |

> **Thesis one-liner (placeholder):**  `<MODEL>` with `<CONDITION>` training data
> achieved the highest entity-level F1 of `<F1>`, confirming that both model choice
> and data-preparation strategy materially affect Hebrew NER performance.

---

## 1. Background — What This Experiment Does and Why

### 1.1 The Problem

Named Entity Recognition (NER) for Hebrew is challenging because:

1. **Hebrew is morphologically rich** — words are inflected and often written
   without vowels, creating ambiguity.
2. **Entity types are imbalanced** — some labels (e.g., `B-PERS`, `I-LOC`) are
   far more common than others (e.g., `B-MISC`), making rare entities hard to
   learn.
3. **Training data preparation matters** — how you split sentences into training
   and evaluation sets, and whether you augment the training data, can change
   results dramatically.

Previous experiments in this thesis explored these dimensions independently:

| Experiment | What It Tested |
|------------|---------------|
| **Exp01** | Baseline NER with standard DictaBERT |
| **Exp02** | Addressing label imbalance via LLM-based sentence duplication |
| **Exp03** | AUC-2T: auxiliary uncertainty classification with entity + BIO heads |
| **Exp04** | Three-step cascaded NER pipeline (entity detection → BIO tagging → type classification) |
| **Exp05** | Exp04 with Step-3 B/I entity-type reconciliation for consistency |
| **Exp06** | Fusion: combines Exp01 (regular NER) and Exp04 (cascaded) predictions via confidence arbitration |
| **Exp07** | Compared 8 sentence-split strategies and found that label-aware splitting improves F1 |
| **Exp08** | LLM mask-filling augmentation: generates synthetic training sentences |

**Experiment 09 ties it all together**: it asks *"What happens when we combine
the best data-preparation strategies with different model architectures?"*

### 1.2 The Two Models

| Model | ID | Description | Strengths |
|-------|----|-------------|-----------|
| **DictaBERT** | `dicta-il/dictabert` | General-purpose Hebrew BERT, pre-trained on modern Hebrew text | Strong on modern Hebrew NER; well-studied baseline |
| **BEREL 3.0** | `dicta-il/BEREL_3.0` | Hebrew BERT variant trained on Biblical and Rabbinical Hebrew corpora | May capture historical or formal register tokens better |

Both models use the standard BERT architecture and are fine-tuned for token
classification (NER) during each experiment run.

### 1.3 The Data Conditions

Data conditions come from two sources:

**From Experiment 07 — Sentence-Split Strategies (8 variants):**

Each variant uses the same Hebrew NER dataset but splits sentences into training
(70%) and evaluation (30%) using a different strategy:

| # | Variant | Strategy |
|---|---------|----------|
| 1 | Baseline (simple random) | Random sentence-level split; no label awareness |
| 2 | Label-aware greedy | Greedy optimization: ensures non-O label distribution in train ≈ full dataset |
| 3 | Rare-label boosted | Oversamples sentences containing rare entity labels into training |
| 4 | Inverse-freq weighted | Weights each sentence by the inverse frequency of its labels |
| 5 | Min-max equalized | Aims to equalize label frequency differences between train and full data |
| 6 | Inv-freq token-weighted | Like #4, but weights at the token level instead of sentence level |
| 7 | Inv-freq eval-guaranteed | Like #4, plus guarantees at least one instance of each label in eval |
| 8 | Inv-freq log-scaled | Inverse-frequency with logarithmic dampening (prevents extreme outlier scores) |

**From Experiment 08 — LLM Data Augmentation (2 conditions):**

| # | Condition | Description |
|---|-----------|-------------|
| 1 | Baseline (no augmentation) | Standard training data, no synthetic sentences |
| 2 | Augmented (LLM mask-fill) | Training data + LLM-generated sentences via mask-filling |

### 1.4 The Downstream Experiments (03–06)

Each data condition is tested on four NER architectures:

| Experiment | Architecture | Key Idea |
|------------|-------------|----------|
| **Exp03** — AUC-2T | Auxiliary Uncertainty Classification, Two Tasks | Adds a second classification head (entity boundary detection) to reduce uncertain predictions |
| **Exp04** — Cascaded Pipeline | Three-step cascaded NER | Step 1: Is this token an entity? Step 2: BIO tagging. Step 3: Entity type classification. Each step is a separate classifier. |
| **Exp05** — Cascaded + Step3 Consistency | Exp04 with reconciliation | After Step 3, enforces consistency between B/I tags and entity types (e.g., if B-PER, then I-PER must follow) |
| **Exp06** — Fusion | Confidence arbitration | Runs Exp01 (regular NER) and Exp04 (cascaded) in parallel, then picks the more confident prediction token-by-token |

---

## 2. How the Script Works — Step by Step

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────┐
│              run_cross_data_model_comparison.py          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Prepare exp07 split files (8 variants)              │
│  2. Prepare exp08 split files (2 conditions)            │
│  3. For each MODEL (DictaBERT, BEREL):                  │
│       For each EXPERIMENT (03, 04, 05, 06):             │
│         For each DATA CONDITION (10 total):             │
│           → Set model via THESIS_MODEL_NAME env var     │
│           → Set data via THESIS_PRESPLIT_* env vars     │
│           → Call experiment.run()                       │
│           → Record F1, precision, recall                │
│  4. Build analytical DataFrames:                        │
│       - Summary pivot table                             │
│       - Exp07 deltas (variant vs baseline)              │
│       - Exp08 deltas (augmented vs baseline)            │
│       - Head-to-head model comparison                   │
│       - Variant summary statistics                      │
│  5. Write Excel workbook + JSON output                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Experiment 07                   Experiment 08
   run()                            run()
     │                                │
     ▼                                ▼
outputs/exp07/splits/           outputs/exp08/splits/
├── split_meta.json             ├── split_meta.json
├── before_exp01_baseline_      ├── baseline_train.json
│   train.json / eval.json      ├── baseline_eval.json
├── after_label_aware_split_    ├── augmented_train.json
│   train.json / eval.json      └── augmented_eval.json
├── after_rare_boosted_...
├── after_inverse_freq_...
├── ... (6 more variants)
│
└───────────┬───────────────────────┘
            │
            ▼
   run_cross_data_model_comparison.py
            │
            │  For each (model × condition):
            │    os.environ["THESIS_MODEL_NAME"] = model_id
            │    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = train.json
            │    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = eval.json
            │    experiment_XX.run()
            │
            ▼
   outputs/cross_comparison/
   ├── cross_comparison_<ts>.xlsx
   ├── cross_comparison_latest.xlsx
   ├── cross_comparison_<ts>.json
   └── cross_comparison_latest.json
```

### 2.3 How Models Are Switched

The script sets `os.environ["THESIS_MODEL_NAME"]` to the Hugging Face model ID
before calling each experiment. Inside each experiment:

- **Exp03 and Exp06** call `configure_model_environment()` from `common.py`, which
  reads `THESIS_MODEL_NAME` and returns the model path.
- **Exp04 and Exp05** launch `core/auc_cascaded_pipeline.py` as a subprocess. The
  environment variables (including `THESIS_MODEL_NAME`) are inherited by the subprocess.

This means no experiment code needs to be modified — model selection is fully
controlled by environment variables.

### 2.4 How Pre-Computed Splits Are Injected

Each experiment checks two environment variables before running:

```python
presplit_train = os.environ.get("THESIS_PRESPLIT_TRAIN_JSON")
presplit_eval  = os.environ.get("THESIS_PRESPLIT_EVAL_JSON")
```

If both point to existing JSON files, the experiment loads those files instead of
performing its own train/eval split. The JSON files contain sentence dictionaries:

```json
[
  {
    "text": "token1 token2 token3",
    "labels": ["O", "B-PER", "I-PER"]
  },
  ...
]
```

The `split_io.py` module provides conversion functions:
- `sentences_to_dataframe()` → converts to pandas DataFrame for Exp03
- `sentences_to_cascaded()` → converts to cascaded format for Exp04/05

---

## 3. Output Files Explained

### 3.1 Excel Workbook Sheets

| Sheet | Content | How to Read It |
|-------|---------|---------------|
| **summary_pivot** | One row per (model × experiment). Columns show F1 for each data condition, plus the best F1 and which condition achieved it. | Start here. Find the highest `best_f1` values. |
| **all_runs** | Complete per-run detail. Every row = one trained model. Columns: model, experiment, condition, F1, precision, recall, status, timing. | Use for full reproducibility. Every number in the pivot can be traced back here. |
| **deltas_exp07** | For each (model × experiment), shows the F1 improvement (or degradation) of each exp07 variant vs the baseline (simple random split). | Look for consistently positive `delta_f1` — that variant helps across architectures. |
| **deltas_exp08** | For each (model × experiment), shows the F1 difference between augmented and non-augmented training data. | Positive `delta_f1` means augmentation helped. |
| **model_comparison** | Head-to-head: for each (experiment × condition), shows both models' F1 and which one won. | Look at the `better_model` column and the aggregate win counts. |
| **variant_summary** | Aggregated statistics per data condition: mean F1 (± std), min, max, across all models and experiments. | The top rows (sorted by mean F1) show the overall best data conditions. |
| **experiment_details** | Extended detail including file paths, descriptions, and split strategies. | For appendix / reproducibility documentation. |
| **documentation** | Metadata describing the experiment design, column meanings, and interpretation rules. | Reference sheet for anyone reading the Excel without this README. |

### 3.2 JSON Output

The JSON file contains the same data in machine-readable format, structured as:

```json
{
  "name": "Cross-Data × Multi-Model Comparison ...",
  "models": ["dicta-il/dictabert", "dicta-il/BEREL_3.0"],
  "experiments": ["03", "04", "05", "06"],
  "results": [ ... ],
  "summary_pivot": [ ... ],
  "deltas_exp07": [ ... ],
  "deltas_exp08": [ ... ],
  "model_comparison": [ ... ],
  "variant_summary": [ ... ],
  "status": "ok"
}
```

---

## 4. Metrics Explained

### 4.1 Entity-Level F1 (Primary Metric)

**F1 score** is the harmonic mean of precision and recall. For NER, it is computed
at the **entity level** using the `seqeval` library in **strict mode**:

- A predicted entity is **correct** only if both its **type** (e.g., PER, LOC) and
  its **span** (exact start and end token positions) match the gold annotation.
- Partial matches count as errors (both a false positive and a false negative).

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Where:
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

### 4.2 Delta F1

The improvement (or degradation) compared to a baseline:

$$
\Delta F1 = F1_{\text{variant}} - F1_{\text{baseline}}
$$

- **Positive Δ F1**: the variant improved over the baseline.
- **Negative Δ F1**: the baseline was better.
- **Near-zero Δ F1**: no meaningful difference.

### 4.3 Precision and Recall

- **Precision**: Of all entities the model predicted, how many were correct?
  High precision → few false alarms.
- **Recall**: Of all entities that actually exist, how many did the model find?
  High recall → few missed entities.

A model with high precision but low recall is *conservative* (finds few entities
but is usually right). A model with high recall but low precision is *aggressive*
(finds most entities but makes many mistakes).

---

## 5. How to Run

### 5.1 Prerequisites

- Python 3.10+ with packages from `requirements.txt`
- Experiment 07 must have been run at least once (to generate split files),
  OR the script will auto-run it
- Experiment 08 must have been run at least once (to generate augmented data),
  OR the script will auto-run it
- Both models must be available locally under `models/hf_models/` or downloadable
  via Hugging Face Hub

### 5.2 Running the Script

```bash
# Default: both models, all experiments, auto-prepare splits
python run_cross_data_model_comparison.py

# Force fresh splits from exp07 and exp08
python run_cross_data_model_comparison.py --exp07-source rerun --force-exp08

# Run only specific experiments
python run_cross_data_model_comparison.py --experiments 03,04

# Run only DictaBERT
python run_cross_data_model_comparison.py --models dictabert

# Run only BEREL
python run_cross_data_model_comparison.py --models berel
```

### 5.3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THESIS_EXP07_SOURCE` | `auto` | Exp07 split policy: `auto`, `saved`, `rerun` |
| `THESIS_CROSS_EXPERIMENTS` | `03,04,05,06` | Which downstream experiments to run |
| `THESIS_CROSS_MODELS` | `dictabert,berel` | Which models to evaluate |
| `THESIS_EXP08_FORCE_RERUN` | `0` | Set to `1` to force exp08 regeneration |
| `THESIS_DEBUG` | `0` | Set to `1` for verbose output |

### 5.4 Expected Runtime

Each individual experiment run (one model × one condition × one experiment) takes
approximately 2–10 minutes depending on hardware. With 80 total runs, expect
**3–12 hours** on a machine with GPU, or significantly longer on CPU only.

---

## 6. Interpreting Results — A Beginner's Guide

### 6.1 Start with the Summary Pivot

Open the Excel file and go to the **summary_pivot** sheet. Each row represents one
(model × experiment) combination. The rightmost columns show:

- `best_f1`: the highest F1 achieved across all data conditions
- `best_condition`: which data condition achieved that F1

**What to look for:**
- Which model consistently achieves higher F1?
- Which data condition appears most often as `best_condition`?

### 6.2 Check the Exp07 Deltas

Go to **deltas_exp07**. This shows how much each split strategy helped (or hurt)
compared to the baseline random split.

**What to look for:**
- Are the delta values consistently positive for some variant? That variant is
  likely a universally better split strategy.
- Does the best variant differ between DictaBERT and BEREL? If yes, the optimal
  split strategy depends on the model.

### 6.3 Check the Exp08 Deltas

Go to **deltas_exp08**. This shows the impact of LLM data augmentation.

**What to look for:**
- Is `delta_f1` positive or negative? Positive means augmentation helped.
- Is the effect consistent across experiments, or does augmentation help some
  architectures more than others?

### 6.4 Head-to-Head Model Comparison

Go to **model_comparison**. Each row compares DictaBERT and BEREL on the same
(experiment × condition) pair.

**What to look for:**
- The `better_model` column tells you who won each matchup.
- The `delta_f1 (model)` column shows the margin. Positive = DictaBERT better.
- Count the wins: if one model wins most matchups, it is the stronger overall model
  for this task.

### 6.5 Variant Summary

Go to **variant_summary**. This ranks data conditions by their mean F1 across all
models and experiments.

**What to look for:**
- The top row is the overall best data condition.
- Check `f1_std` — a low standard deviation means the condition is *robust*
  (works well regardless of model or architecture).

---

## 7. How to Cite in a Thesis

### 7.1 Describing the Experiment

> We conducted a systematic cross-comparison evaluating two Hebrew BERT models
> (DictaBERT and BEREL 3.0) across 10 training-data conditions — 8 sentence-split
> strategies from Experiment 07 and 2 data-augmentation conditions from Experiment
> 08 — applied to 4 downstream NER architectures (AUC-2T, Cascaded Pipeline,
> Cascaded + Step-3 Consistency, and Fusion). This design yielded 80 trained model
> instances, enabling paired comparisons of both model choice and data preparation
> strategy on entity-level F1 performance.

### 7.2 Reporting Results (Template)

> Table X shows the entity-level F1 scores for all (model × experiment × condition)
> combinations. The best overall result was achieved by `<MODEL>` with the
> `<CONDITION>` data condition on the `<EXPERIMENT>` architecture
> (F1 = `<F1_VALUE>`).
>
> Across all conditions, `<MODEL>` outperformed `<OTHER_MODEL>` in `<N>` out of 40
> head-to-head matchups (average Δ F1 = `<AVG_DELTA>`), suggesting [it is / there
> is no clear] advantage for [model] on this Hebrew NER task.
>
> Label-aware sentence splitting (Experiment 07) improved F1 by an average of
> `<AVG_DELTA_EXP07>` points over the baseline random split. The
> `<BEST_EXP07_VARIANT>` strategy was most effective.
>
> LLM data augmentation (Experiment 08) [improved / did not improve] F1 by an
> average of `<AVG_DELTA_EXP08>` points. The effect was [consistent / inconsistent]
> across architectures.

### 7.3 Referencing Output Files

> Full per-run results are available in the supplementary Excel workbook
> (`cross_comparison_latest.xlsx`). The `summary_pivot` sheet provides an overview;
> `deltas_exp07` and `deltas_exp08` sheets contain paired comparisons;
> `model_comparison` shows head-to-head results. The machine-readable JSON version
> (`cross_comparison_latest.json`) provides identical data.

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| **NER** | Named Entity Recognition — identifying and classifying named entities (people, places, organizations, etc.) in text |
| **BIO tagging** | Begin/Inside/Outside annotation scheme: B-PER marks the first token of a person name, I-PER marks continuation tokens, O marks non-entity tokens |
| **F1 score** | Harmonic mean of precision and recall; ranges from 0 (worst) to 1 (best) |
| **Precision** | Fraction of predicted entities that are correct |
| **Recall** | Fraction of true entities that were found by the model |
| **Delta (Δ)** | Difference between two F1 scores; positive = improvement |
| **BERT** | Bidirectional Encoder Representations from Transformers — a neural network architecture for language understanding |
| **Fine-tuning** | Taking a pre-trained model and training it further on a specific task (here: NER) |
| **Token** | A word or sub-word unit that the model processes |
| **Sentence split** | Dividing the dataset into training and evaluation portions at the sentence level |
| **Label-aware split** | A splitting strategy that considers the distribution of entity labels to ensure balanced representation |
| **Data augmentation** | Creating additional synthetic training examples to improve model performance |
| **LLM mask-filling** | Using a language model to generate new sentences by randomly masking tokens and having the model predict replacements |
| **Cascaded pipeline** | A multi-step approach where each step feeds into the next, as opposed to a single end-to-end model |
| **Confidence arbitration** | Choosing between two model predictions based on which model is more certain about its answer |
| **seqeval** | A Python library for evaluating sequence labeling tasks using entity-level metrics |
