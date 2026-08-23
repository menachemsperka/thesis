# Small-data / cross-domain NER benchmark

Technical reference for the cross-benchmark runner (`run_cross_benchmark_comparison.py`). It documents **datasets**, **split and augmentation protocols**, **NER methods**, **fusion**, and **evaluation** in plain language, for thesis review and paper writing.

**Colab workflow:** [COLAB.md](COLAB.md)

---

## Table of contents

1. [Research goal](#1-research-goal)
2. [Datasets and encoders](#2-datasets-and-encoders)
3. [Small-data sampling protocol](#3-small-data-sampling-protocol)
4. [Train/evaluation split methods](#4-trainevaluation-split-methods)
5. [Training-set augmentation (Experiment 08)](#5-training-set-augmentation-experiment-08)
6. [NER methods](#6-ner-methods)
7. [Default comparison design](#7-default-comparison-design)
8. [Evaluation metrics](#8-evaluation-metrics)
9. [Running the benchmark](#9-running-the-benchmark)
10. [Outputs and reproducibility](#10-outputs-and-reproducibility)
11. [Repository layout](#11-repository-layout)

---

## 1. Research goal

Named Entity Recognition (NER) tags each token in a sentence with a label such as `B-PER` (begin person), `I-PER` (inside person), or `O` (outside any entity). In **small-data** settings—few hundred labeled sentences—models often fail on **rare entity types** and **unstable train/test splits**.

This benchmark asks, on **public corpora** and **domain-matched transformers**:

1. How does a **simple baseline** (standard fine-tuned BERT NER, Experiment 01) behave under repeated random small samples?
2. Does a **stronger pipeline**—**paper-style stratified splits**, **LLM mask-filling augmentation**, **BERT-CRF** models, and **SVM fusion** of two CRF experts (Experiment 10)—improve over that baseline on the same seeds?

The runner fixes the **protocol** (splits, seeds, augmentation, metrics) so results are comparable across **English news**, **Hebrew**, and **biomedical** data.

---

## 2. Datasets and encoders

Each **benchmark** pairs one Hugging Face NER corpus with one **encoder** (pretrained language model). Official **train** sentences are used for sampling and splitting; **test** (or validation, if test is missing) is merged into a single `corpus.csv` for reference labels only—**evaluation during benchmark runs always uses the held-out 30% eval split** produced by the runner, not the official test set, unless you change the protocol.

| Benchmark key | Corpus | Language / domain | Encoder | Typical entity types (BIO) |
|---------------|--------|-------------------|---------|----------------------------|
| `conll2003_bert` | [CoNLL-2003](https://huggingface.co/datasets/conll2003) | English news | `bert-base-uncased` | PER, ORG, LOC, MISC |
| `nemo_dictabert` | [NEMO](https://huggingface.co/datasets/onlplab/nemo) (fallback: `imvladikon/nemo_corpus`) | Hebrew | `dicta-il/dictabert` | Corpus-specific BIO set from HF features |
| `bc5cdr_pubmedbert` | [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr) | Biomedical abstracts | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Chemical, disease (and related BIO tags) |

**Data format.** Each sentence is stored as:

- `text`: space-joined tokens  
- `labels`: parallel BIO tag sequence (same length as tokens)

Loaders live in `corpus_loaders.py`. Non-Hebrew runs set `THESIS_SKIP_HEBREW_TEXT_VALIDATION=1` automatically.

**Why three benchmarks?** They stress **different label inventories and scripts** while sharing the same code path—useful for a thesis claim about **cross-domain small-data NER**, not only one dataset.

---

## 3. Small-data sampling protocol

Two **regimes** control how many sentences enter the split pipeline:

| Regime | Pool | Typical use |
|--------|------|-------------|
| `small_300` | **300 sentences** | Small-data stress test |
| `full` | **All official train** sentences | Upper bound with more data |

### Per-seed sampling (`small_300`)

For regime `small_300`, **each random seed** draws its **own** 300-sentence subset from the official training split:

1. Let \(N\) be the number of official train sentences and \(S = 300\).
2. With seed \(s\), sample indices \(I_s \subset \{1,\ldots,N\}\), \(|I_s| = S\), **without replacement** (NumPy `default_rng(s)`).
3. On pool \(D_s = \{x_i : i \in I_s\}\), apply a **70% / 30%** sentence-level train/eval split (see §4).

So different seeds see **different sentences** and **different train/eval partitions**. Default: **20 seeds** (42–61).

### Full regime

Pool = all official train sentences; same 70/30 split variants and seeds per sentence pool.

### Split ratio

Train fraction \(\rho = 0.7\) (config: `SPLIT_RATIO`). For a pool of \(n\) sentences:

\[
n_{\text{train}} = \max\bigl(1,\; \min(n-1,\; \lfloor n \cdot \rho \rfloor)\bigr), \quad n_{\text{eval}} = n - n_{\text{train}}.
\]

For \(n = 300\), this is about **210 train** and **90 eval** sentences per seed (before augmentation adds synthetic train rows only).

---

## 4. Train/evaluation split methods

Split logic matches **Experiment 07** in the parent repository (`experiments/exp07_split_artifacts.py`). Two variants are materialized for every seed and regime:

| Variant key | Name | Role in default benchmark |
|-------------|------|---------------------------|
| `before_exp01_baseline` | Simple random split | **Baseline** (Exp01): shuffle sentences uniformly, cut at 70% |
| `after_multilabel_iterative_paper` | Paper-style multilabel stratified split | **Treatment** (Exp10 SVM path): rarest-label-first iterative assignment |

Both use the **same seed** for splitting as for pool sampling in `small_300` (reproducibility per seed).

### 4.1 Simple random split (baseline control)

1. Copy the sentence list and shuffle with `random.Random(seed)`.
2. Split index \(k = \lfloor n \cdot \rho \rfloor\) (clamped so both sides are non-empty).
3. Train = first \(k\) sentences; eval = remainder.

**Properties:** No label awareness; rare types may be **absent** from train or eval by chance—realistic for naive small-data practice.

### 4.2 Multilabel stratified split (paper-style)

Each sentence is a **multilabel instance**: the set of **entity types** appearing in it (non-`O` labels, ignoring B/I prefix for stratification). The algorithm follows the spirit of **Sechidis et al. (2011)** iterative stratification:

- Target train/eval **capacities**: \(n_{\text{train}} = \lfloor n\rho \rfloor\), \(n_{\text{eval}} = n - n_{\text{train}}\).
- For each entity type (label) \(\ell\), target counts in train/eval are \(n_\ell \cdot \rho\) and \(n_\ell \cdot (1-\rho)\), where \(n_\ell\) is the number of sentences containing \(\ell\).
- Repeatedly pick a **priority label** among types still present in unassigned sentences—**fewest remaining examples**, random tie-break.
- Assign unassigned sentences that contain that label to the fold (train or eval) that **most needs** that label, with tie-breaks on fold capacity, then randomness.
- Sentences with only `O` fill remaining slots by fold capacity.

**Properties:** Train and eval **mirror label proportions** more closely than random splitting—important when some types appear in only a few sentences.

Implementation details (tie-break order) differ slightly from the alternate `after_multilabel_stratified` variant in Exp07; this benchmark uses **`after_multilabel_iterative_paper` only**.

---

## 5. Training-set augmentation (Experiment 08)

For the **treatment** arm, training uses **augmented** JSON: original train sentences plus **synthetic** sentences generated by **masked language modeling** (same family as the benchmark encoder unless `THESIS_AUGMENTATION_MODEL_NAME` is set).

Augmentation is implemented in `experiments/experiment_08_llm_augmentation.py` and invoked from `augmentation.py` during `--prepare-only` / `--prepare-augmentation-only`.

**Principles (simple):**

1. **Eval split is never modified**—fair comparison on identical eval sentences.
2. **Rare labels** (few sentences in train) get more generation budget.
3. Strategies include **single-token mask fill**, **multi-position masks** on entity tokens, and **duplication** for extremely rare types.
4. **`THESIS_EXP08_MULTIPLIER`** (default **3**) scales how many synthetic sentences are produced relative to label deficit.

Augmented files are named `*_augmented_train.json` under `data/<benchmark>/splits/<regime>/`.

---

## 6. NER methods

The default benchmark compares two **end-to-end systems**. The treatment system internally trains **two CRF experts** before fusion; those steps are not separate reported experiments in the default run.

### 6.1 Experiment 01 — Regular NER (baseline)

**Module:** `experiments/experiment_01_regular_ner.py`

**Architecture:**

```
Tokens → BERT encoder → linear layer → softmax over BIO tags (per token)
```

**Training:** Hugging Face `Trainer` on the **baseline** train JSON (no augmentation), **random** split variant. **Inference:** argmax tag per token (with standard WordPiece alignment).

**Role:** Standard **strong but simple** small-data baseline—no CRF, no cascade, no fusion.

### 6.2 Experiment 10 — BERT-CRF regular (`10_regular`)

**Module:** `experiments/experiment_10_regular_ner_crf.py`, core in `core/bert_crf_training.py` and `core/crf_layer.py`.

**Architecture:** BERT → linear **emissions** \(e_t(k)\) for each tag \(k\) at token \(t\) → **linear-chain CRF**.

**Path score** for tag sequence \(\mathbf{y} = (y_1,\ldots,y_T)\):

\[
s(\mathbf{y}) = \sum_{t=1}^{T} \bigl( A_{y_{t-1},\, y_t} + e_t(y_t) \bigr)
\]

where \(A\) is a learned transition matrix (including START/STOP).

**Training loss** (negative log-likelihood of gold tags \(\mathbf{y}^*\)):

\[
\mathcal{L} = -\log \frac{\exp(s(\mathbf{y}^*))}{\sum_{\mathbf{y}'} \exp(s(\mathbf{y}'))}
\]

The denominator is computed with the **forward algorithm**; decoding uses **Viterbi** (\(\arg\max_{\mathbf{y}} s(\mathbf{y})\)).

**Class imbalance:** bias of the `O` tag is initialized to **+6.0** (Souza et al., 2019 practice) so the model does not collapse to all-`O` early in training.

### 6.3 Experiment 10 — Cascaded CRF pipeline (`10_cascade`)

**Module:** `experiments/experiment_10_cascaded_pipeline_crf.py`

**Idea:** Keep the **three-step cascade** from Experiment 04 (entity detection → B/I → entity type) for diagnostics, and add a **joint full-BIO CRF head** on shared BERT representations. Span-level predictions use **Viterbi** on the joint head; optional B/I–type reconciliation (`THESIS_STEP3_BI_TYPE_RECONCILE`) matches Exp05-style consistency repair.

**Role:** Second expert with **different inductive bias** (modular + joint CRF) for fusion.

### 6.4 Experiment 10 — SVM router fusion (`10_svm_ready`)

**Module:** `experiments/experiment_10_fusion_svm_ready.py`

**Inputs:** Token-level predictions from **10_regular** and **10_cascade** on the **same** eval split (aligned by `sentence_id`, `token_idx`).

**Fusion rule:**

1. If both models **agree** on the tag → use that tag.
2. If they **disagree** → train a **LinearSVC** router on **disagreement tokens** where **exactly one** model is correct:
   - Features: e.g. `regular_prob`, `cascade_prob`, margins, probability differences, predicted BIO/type categories (see `_NUMERIC_FEATURES` / `_CATEGORICAL_FEATURES` in the experiment file).
   - Target: choose `"regular"` or `"cascade"`.
   - Preprocessing: `StandardScaler` on numeric features, `OneHotEncoder` on categoricals, `class_weight="balanced"`.

3. Apply the router on disagreement tokens; reported **F1** is on the **fused** tag sequence.

**Important limitation (documented in code):** the router is **trained and applied on the same eval split** used for reporting. Treat fusion gains as an **exploratory upper bound**; a strictly held-out router set would be tighter for thesis claims.

**Default treatment run:** augmented train + **paper stratified** split; runner calls `_ensure_base_artifacts_crf` to train/cache `10_regular` and `10_cascade` on that condition, then runs `10_svm_ready`.

Further teaching material: `experiments/experiment_10_README.md`.

---

## 7. Default comparison design

Unless you pass custom `--experiments`, each **(benchmark, regime, seed)** runs **two** conditions:

| Arm | Experiment | Train data | Split variant |
|-----|------------|------------|---------------|
| **Baseline** | `01` | Non-augmented | Simple random (`before_exp01_baseline`) |
| **Treatment** | `10_svm_ready` | Augmented (Exp08) | Paper multilabel stratified (`after_multilabel_iterative_paper`) |

**Fairness notes for the thesis:**

- Both arms use the **same 300-sentence pool** for a given seed (in `small_300`), but **different split functions** assign train/eval differently—baseline vs treatment compares **pipelines**, not only split fairness. If you need the **same partition** with only augmentation differing, change `_condition_matches_experiment` in the runner or add a flag.
- **Eval sets differ** between arms when random vs stratified splits differ—reported F1 is **not** on identical eval sentences across baseline and treatment unless you redesign the pairing.

**Runs per configuration (default):** 3 benchmarks × 1 regime (`small_300` typical) × 20 seeds × 2 arms = **120** training/eval jobs (plus internal CRF training for treatment).

---

## 8. Evaluation metrics

**Primary metric:** **entity-level F1** from the `seqeval` library (same as Exp01/Exp10 trainers)—precision, recall, and F1 over **predicted entity spans** with correct type, after BIO decoding.

**Aggregation across seeds:** the Excel export includes per-run F1 and sheets such as **`deltas_treatment_vs_baseline`** (treatment − baseline per seed) and **`paired_treatment_summary`** (mean ± std of deltas per benchmark/regime).

**Warnings:** on tiny eval sets, some entity types may have **no predictions**; `seqeval` then emits undefined-precision warnings and may zero-out rare-type scores—expected in small-data regimes.

---

## 9. Running the benchmark

From repository root:

```bash
# 1) Download corpora + build splits (both split variants, all seeds)
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --prepare-only

# 2) LLM augmentation for treatment splits (GPU recommended)
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --prepare-augmentation-only

# 3) Dry-run: list matched (experiment, condition) pairs
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --dry-run --regimes small_300

# 4) Train/evaluate (default: exp01 + exp10_svm_ready)
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --resume --regimes small_300 --num-seeds 20
```

`run_benchmark.py` is an alias for the same runner.

### Useful CLI flags

| Flag | Purpose |
|------|---------|
| `--benchmarks` | Subset: `conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert` |
| `--regimes` | `small_300`, `full`, or both |
| `--num-seeds` / `--seeds` | Paired seeds (default 20, starting 42) |
| `--experiments` | Default `01,10_svm_ready`; override for full Exp10 grid |
| `--train-modes` | `baseline,augmented` (needed to prepare augmentation) |
| `--skip-augmentation` | Baseline-only (treatment will fail unless removed from experiments) |
| `--base-mode` | `auto` / `reuse` / `retrain` for Exp10 CRF cache |
| `--resume` | Continue checkpoint for **current run plan** |
| `--fresh` | Ignore prior checkpoint progress (backs up old file) |
| `--prepare-only` / `--prepare-augmentation-only` | Data prep only |

**Colab:** set `THESIS_RUN_ENV=colab` before running; see [COLAB.md](COLAB.md).

---

## 10. Outputs and reproducibility

Under `projects/small_data_ner_benchmark/outputs/cross_comparison/`:

| Artifact | Description |
|----------|-------------|
| `cross_comparison_<timestamp>.xlsx` / `_latest.xlsx` | Main results workbook |
| `cross_comparison_<timestamp>.json` | Same rows, machine-readable |
| `benchmark_cross_comparison_checkpoint.json` | Resume state (includes **run-plan fingerprint**) |
| `cross_comparison_base_crf_ready_index.json` | Cache index for Exp10 regular + cascade |
| `consolidated_error_analysis_exp10_*.xlsx` | Merged token error sheets (treatment / Exp10 runs only) |
| `dataset_details_*.json` | Per-split sentence/token/entity counts |

**Excel sheets (typical):** `dataset_details`, `summary_pivot`, `all_runs`, `deltas_treatment_vs_baseline`, `paired_treatment_summary`, `deltas_split_variants` (when both split variants exist for the same experiment), `documentation`, `exp10_error_analysis`.

Per-run metrics also write under repo `outputs/exp01/`, `outputs/exp10_regular/`, `outputs/exp10_cascade/`, `outputs/exp10_svm_ready/` with paths recorded in `all_runs`.

**Checkpoint behavior:** if you change experiments, regimes, seeds, or conditions, the runner **drops stale checkpoint rows** or use **`--fresh`** to start clean—avoids merging hundreds of old Exp10-grid runs into a new 120-run plan.

---

## 11. Repository layout

```
projects/small_data_ner_benchmark/
  run_cross_benchmark_comparison.py   # main runner
  run_benchmark.py                    # alias
  configs.py                          # benchmarks, regimes, default experiments
  corpus_loaders.py                   # HF dataset → sentence JSON
  splits.py                           # pools, split variants, split_meta.json
  augmentation.py                     # Exp08 wrapper for benchmark splits
  split_stats.py                      # dataset_details export
  data/<benchmark>/
    corpus.csv
    sentences_full_train.json
    split_meta.json
    splits/<regime>/                  # train/eval JSON per variant × seed
  outputs/cross_comparison/
```

Parent thesis code: `experiments/experiment_01_regular_ner.py`, `experiment_08_llm_augmentation.py`, `experiment_10_*`, `experiments/exp07_split_artifacts.py`, `run_cross_data_model_comparison.py` (Hebrew cross-model runner with the same split keys).

---

## References (methods)

- **CoNLL-2003:** Tjong Kim Sang & De Meulder (2003) — shared-task NER data.  
- **Iterative stratification:** Sechidis et al. (2011) — multilabel stratified splits.  
- **BERT-CRF NER:** Lample et al. (2016); Souza et al. (2019) — neural emissions + CRF, training practice.  
- **BC5CDR:** Li et al. (2016) — chemical–disease relation corpus.  
- **DictaBERT / NEMO:** Hebrew PLM and NEMO corpus via Hugging Face (see dataset cards for citations).

For equation-level CRF detail and lab exercises, see `experiments/experiment_10_README.md` in the parent repository.
