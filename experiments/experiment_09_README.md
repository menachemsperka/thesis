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
| Exp06_fusion_normalized – Calibrated Fusion | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp06_fusion_entropy – Entropy-Weighted Fusion | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp06_fusion_learned_weights – Learned-Weights Fusion | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp06_fusion_svm – SVM Router Fusion | `<MODEL>` | `<CONDITION>` | `<F1>` |
| Exp06_fusion_ensemble_rules – Ensemble-Rules Fusion | `<MODEL>` | `<CONDITION>` | `<F1>` |

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
| **Exp06_fusion_normalized** | Calibrated fusion: applies temperature scaling to regular and cascaded confidences before arbitration |
| **Exp06_fusion_entropy** | Entropy-weighted fusion: reduces the influence of uncertain predictions before arbitration |
| **Exp06_fusion_learned_weights** | Learned-weights fusion: learns the best regular-vs-cascaded weighting on training data |
| **Exp06_fusion_svm** | SVM-router fusion: keeps agreements and learns a disagreement routing rule from training data |
| **Exp06_fusion_ensemble_rules** | Rule-based fusion: uses agreement and confidence-gap logic rather than plain confidence comparison |
| **Exp07** | Compared 3 sentence-split strategies and found that label-aware splitting improves F1 |
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

**From Experiment 07 — Sentence-Split Strategies (4 variants):**

Each variant uses the same Hebrew NER dataset but splits sentences into training
(70%) and evaluation (30%) using a different strategy:

| # | Variant | Strategy |
|---|---------|----------|
| 1 | Baseline (simple random) | Random sentence-level split; no label awareness |
| 2 | Label-aware greedy | Greedy optimization: ensures non-O label distribution in train ≈ full dataset |
| 3 | Multilabel stratified | Iterative multilabel stratification preserving per-label proportions in both folds |
| 4 | Multilabel stratified (paper-style) | Same proportional goal as Method 3, but with paper-style tie-breaking: rare-label need, then fold capacity, then random |

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
| **Exp06_fusion_normalized** — Calibrated Fusion | Confidence normalization + arbitration | Same as Exp06, but calibrates both confidence sources first so arbitration compares scores on a more compatible scale |
| **Exp06_fusion_entropy** — Entropy-Weighted Fusion | Uncertainty-aware arbitration | Same as Exp06, but downweights uncertain predictions so high-entropy decisions are less likely to win arbitration |
| **Exp06_fusion_learned_weights** — Learned-Weights Fusion | Training-learned arbitration | Learns one global regular-vs-cascade weighting from training data and applies that weighting during eval/test arbitration |
| **Exp06_fusion_svm** — SVM Router Fusion | Meta-learned disagreement arbitration | Keeps agreed labels, and for disagreements uses a linear SVM trained on train-split token features to choose regular vs cascaded prediction |
| **Exp06_fusion_ensemble_rules** — Ensemble-Rules Fusion | Heuristic arbitration | Uses agreement, confidence-gap, and conservative fallback rules instead of raw confidence alone |

In simple terms, **Learned-Weights Fusion** chooses how much to trust each source model by searching for the best single weighting value on the training split.

It tries many values of $\alpha$ on a grid from 0.00 to 1.00 (inclusive, step 0.01), so there are 101 candidates:

$$
\alpha \in \{0.00, 0.01, 0.02, \ldots, 0.99, 1.00\}
$$

For each candidate $\alpha$, token-level fusion is done with one simple rule:

1. If both models predict the same BIO label, keep that agreed label immediately.
2. If they disagree, compute two weighted confidence scores:

$$
s_{regular} = \alpha \cdot p_{regular}, \quad
s_{cascade} = (1-\alpha) \cdot p_{cascade}
$$

3. Choose the label from the model with the larger weighted score.

After all tokens are fused for that $\alpha$, the method computes entity-level precision/recall/F1 on the training split. The best $\alpha$ is:

$$
\alpha^* = \arg\max_{\alpha} F1_{train}(\alpha)
$$

Then that single $\alpha^*$ is frozen and reused for eval/test inference.

Interpretation of the endpoints is useful:

- $\alpha = 1.00$: in disagreements, trust regular NER only.
- $\alpha = 0.00$: in disagreements, trust cascaded NER only.
- $0 < \alpha < 1$: compromise between the two.

Quick example for one disagreement token:

- Regular predicts `B-PER` with $p_{regular}=0.62$.
- Cascade predicts `B-ORG` with $p_{cascade}=0.80$.
- If $\alpha=0.70$: $0.70\cdot0.62=0.434$ vs $0.30\cdot0.80=0.240$, so choose `B-PER`.
- If $\alpha=0.30$: $0.30\cdot0.62=0.186$ vs $0.70\cdot0.80=0.560$, so choose `B-ORG`.

So the sweep is effectively selecting the disagreement policy that maximizes full-sequence entity F1, rather than fixing a hand-tuned or arbitrary 50/50 trust split.

Why did the other fusion variants perform worse on average?

- **Raw fusion (`06`)** is simple and sometimes helpful, but it assumes raw confidence scales are already comparable between models; this is often false, so disagreement decisions are noisy.
- **Calibrated fusion (`06_fusion_normalized`)** improves score comparability, but calibration alone does not learn which source is usually more reliable for this task, so gains are limited.
- **Entropy-weighted fusion (`06_fusion_entropy`)** helps by discounting uncertain predictions, but entropy is still an indirect signal of correctness and can miss systematic model bias.
- **Ensemble-rules fusion (`06_fusion_ensemble_rules`)** is interpretable and stronger than raw fusion, yet fixed hand-crafted rules/thresholds are less adaptable than a weight chosen directly to maximize F1.

This pattern matches the observed results in this README: `06_fusion_learned_weights` is the most consistent method (beats both source systems in 10/12 shared comparisons), while the other variants improve some cases but remain less stable across model-condition pairs.

Short description of Exp04: it is the main non-fusion structured alternative to regular NER in this thesis. Instead of predicting the final tag in one shot, it breaks NER into entity detection, BIO boundary prediction, and type classification. That staged design makes Exp04 a useful comparison point on its own and also the cascaded source model that all Exp06 fusion variants are built around.

---

## 2. How the Script Works — Step by Step

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────┐
│              run_cross_data_model_comparison.py          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Prepare exp07 split files (3 variants)              │
│  2. Prepare exp08 split files (2 conditions)            │
│  3. For each MODEL (DictaBERT, BEREL):                  │
│       For each EXPERIMENT (03, 04, 05, 06):             │
│         For each DATA CONDITION (5 total):              │
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
├── after_multilabel_stratified_
│   train.json / eval.json
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

# Compare original fusion vs calibrated fusion
python run_cross_data_model_comparison.py --experiments 06,06_fusion_normalized

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

### 5.4 Why Include Exp06_fusion_normalized

Use `06_fusion_normalized` when Exp06 is close to (or below) Exp01 and you
suspect the fusion decision is impacted by confidence-scale mismatch.

- Exp06 compares raw confidences from two different pipelines.
- Exp06_fusion_normalized calibrates both using temperature scaling before arbitration.

Recommended thesis comparison command:

```bash
python run_cross_data_model_comparison.py --experiments 01,06,06_fusion_normalized --models dictabert,berel
```

Recommended reporting deltas:

- `delta_f1(06_fusion_normalized - 06)` to isolate normalization gain.
- `delta_f1(06_fusion_normalized - 01)` to verify whether calibrated fusion beats regular NER.

### 5.5 Fusion Result Summary

The ready-results export (`outputs/cross_comparison/cross_comparison_ready_all_methods_latest.json`) now makes it possible to compare each fusion method directly against the two source systems it is supposed to improve on:

- Exp `01`: Regular NER
- Exp `04`: AUC Cascaded Pipeline

Across 12 shared model-condition comparisons:

| Fusion method | Beats Exp01 | Beats Exp04 | Beats both source models | Mean delta vs Exp01 | Mean delta vs Exp04 |
|------|------|------|------|------|------|
| `06` | 7 / 12 | 7 / 12 | 3 / 12 | +0.0026 | +0.0202 |
| `06_fusion_normalized` | 6 / 12 | 6 / 12 | 4 / 12 | +0.0000 | +0.0176 |
| `06_fusion_entropy` | 9 / 12 (+1 tie) | 7 / 12 | 7 / 12 | +0.0101 | +0.0277 |
| `06_fusion_learned_weights` | 11 / 12 | 10 / 12 | 10 / 12 | +0.0327 | +0.0503 |
| `06_fusion_ensemble_rules` | 9 / 12 (+1 tie) | 7 / 12 | 6 / 12 | +0.0180 | +0.0356 |

Interpretation:

- Raw fusion (`06`) is only mildly better than the source models.
- Confidence normalization alone (`06_fusion_normalized`) is not enough to make fusion reliably stronger.
- Entropy weighting and rule-based arbitration both help, but remain mixed.
- `06_fusion_learned_weights` is the best current fusion method by a clear margin. It is the only variant that is consistently strong against both source systems, beating both baselines in 10 of 12 comparisons.

The two remaining losses are expected in a realistic fusion setup. Fusion is not an oracle that can always identify which source prediction is correct; it can only use proxy signals such as confidence or a learned global weight. This works well on average, but disagreement cases are precisely the difficult examples where confidence may not track correctness perfectly. Because evaluation is sequence-level F1, even a small number of wrong arbitration decisions on entity boundaries can produce a measurable drop for an otherwise strong fusion rule.

### 5.6 Expected Runtime

Each individual experiment run (one model × one condition × one experiment) takes
approximately 2–10 minutes depending on hardware. With 80 total runs, expect
**3–12 hours** on a machine with GPU, or significantly longer on CPU only.

### 5.7 Ready-Results Mode (Skip Retraining)

To avoid retraining all models for every fusion or consistency variant, the
pipeline supports **ready-results mode**.  The idea:

1. **Train once:** run Exp01 (Regular NER) and Exp04 (Cascaded Pipeline) — these
   are the only experiments that require GPU training.
2. **Derive everything else from saved outputs:**
   - `experiment_05_ready.py` loads Exp04 output and applies the B/I entity-type
     consistency rule as post-processing — no retraining.
   - All `experiment_06_*_ready.py` variants load the Exp01 and Exp04 token-level
     outputs, merge them, and apply their fusion strategy — no retraining.

**Ready experiment IDs** (for `--experiments`):

| ID | File | Strategy |
|----|------|----------|
| `05_ready` | `experiment_05_ready.py` | Cascaded + consistency from Exp04 |
| `06_ready` | `experiment_06_fusion_ready.py` | Confidence comparison |
| `06_normalized_ready` | `experiment_06_fusion_normalized_ready.py` | Temperature-calibrated |
| `06_entropy_ready` | `experiment_06_fusion_entropy_ready.py` | Entropy-weighted |
| `06_learned_ready` | `experiment_06_fusion_learned_weights_ready.py` | Learned alpha weights |
| `06_ensemble_ready` | `experiment_06_fusion_ensemble_rules_ready.py` | Rule-based ensemble |
| `06_svm_ready` | `experiment_06_fusion_svm_ready.py` | SVM disagreement router |

**Quick-start: train once, then iterate on fusion rules:**

```bash
# Step 1 — train the two base models (slow, once per split)
python run_cross_data_model_comparison.py --experiments 01,04

# Step 2 — run ALL fusion variants on ready outputs (fast, seconds each)
python run_cross_data_model_comparison.py --experiments 05_ready,06_ready,06_normalized_ready,06_entropy_ready,06_learned_ready,06_ensemble_ready,06_svm_ready
```

**Environment variables for explicit source paths:**

| Variable | Default resolution | Description |
|----------|-------------------|-------------|
| `THESIS_READY_EXP01_XLSX` | `outputs/exp01/latest.json → metrics_file` | Exp01 output with `token_predictions` sheet |
| `THESIS_READY_EXP04_XLSX` | `outputs/exp04/latest.json → metrics_file` | Exp04 cascaded pipeline output |
| `THESIS_READY_EXP05_XLSX` | `outputs/exp05/latest.json → metrics_file` | Exp05 output (if using exp05 as cascade source) |

**Important**: Exp01 must be rerun with the updated code that saves the
`token_predictions` sheet (with `prob`, `entropy`, `margin` columns).  Old Exp01
outputs without this sheet will produce an error.

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
