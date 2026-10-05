# Thesis Overview (Full Labeled Corpus)

Consolidated thesis documentation for the **full NER dataset** (~300 sentences), cross-comparison runner, Exp01–Exp06/OOF fusion, fair training, and IEEE journal exports.

*Merged on 2026-09-30 from all overview markdown files in this repository.*

## Master table of contents

| Part | Section | Source (archived) |
|------|---------|-------------------|
| **I** | Pipeline methodology (~300 sentences default) | Former `theisis overview.md` |
| **II** | Fair training hyperparameters (Exp01 / Exp04) | Former `cross_comparison_fair_training_overview.md` |
| **III** | OOF router fusion (5-fold stratified CV) | Former `thesis_overview_fusion_oof_cv.md` |
| **IV** | Journal paper: one command, results & error analysis | Former `thesis_overview_journal_paper_results.md` |
| **V** | Statistical significance testing (seed-count power, paired t-test / Wilcoxon, reporting template) | Former `STATISTICAL_SIGNIFICANCE_GUIDE.md` |

**150-sentence pilot:** see companion file [`thesis_overview_150_sentences.md`](thesis_overview_150_sentences.md).

---



# Part I — Pipeline methodology

## 1. Purpose of This Methodology

This document explains the scientific architecture triggered by the command:

```bash
python run_cross_data_model_comparison.py \
--resume \
--base-mode auto \
--experiments 01,04,05_ready,06_ready,06_svm_oof \
--models dictabert,berel,hero,alephbertgimmel,xlm_roberta,mt5 \
--condition-sources exp07,exp07+aug \
--num-seeds 20
```

The command runs a controlled comparison of Hebrew Named Entity Recognition (NER) methods. It is not only a script that calls other scripts. It is a complete experimental architecture with:

1. several Hebrew and multilingual transformer encoders,
2. several train/evaluation split strategies,
3. augmentation-based data variants,
4. base NER systems,
5. post-processing systems,
6. fusion systems,
7. paired multi-seed statistical testing,
8. cached base artifacts for reproducibility and efficiency.

The main scientific question is:

> How do different Hebrew transformer models behave under different sentence split and augmentation conditions, and can cascaded or fused prediction architectures improve entity-level NER performance?

The pipeline compares five **core** experiments (01–06 ready track):

| Experiment ID | Method name | Core idea |
|---|---|---|
| `01` | Regular NER | Single transformer token-classification model predicts full BIO entity labels directly. |
| `04` | AUC Cascaded Pipeline | NER is decomposed into entity detection, BIO position, and entity type prediction. |
| `05_ready` | Cascaded B/I Consistency | Post-processes `04` outputs to repair inconsistent `B-X` followed by `I-Y` predictions. |
| `06_ready` | Confidence Fusion | Combines `01` and `04`; if they disagree, compares per-source scalar confidence (`regular_prob` vs `cascade_prob`; see **§11.2**). |
| `06_svm_oof` | Linear SVM Router Fusion (**primary**) | **5-fold** sentence-level stratified CV + **OOF** stacking; held-out router generalization (**§12C**). |
| `06_svm_kernel_oof` | Kernel SVM Router Fusion (**primary**) | Same protocol; RBF `SVC` (**§12C**). |
| `06_nb_oof` | Naive Bayes Router Fusion (**primary**) | Same protocol (**§12C**). |
| `06_lr_oof` | Logistic Regression Router Fusion (**primary**) | Same protocol (**§12C**). |
| `06_rf_oof` | Random Forest Router Fusion (**primary**) | Same protocol (**§12C**). |
| `06_mlp_oof` | MLP Router Fusion (**primary**) | Same protocol (**§12C**). |
| `06_svm_ready` | Linear SVM Router Fusion (**appendix**) | In-sample routing upper bound on Exp07 eval (**§12.4**). |
| `06_svm_kernel_ready` | Kernel SVM Router Fusion (**appendix**) | Same ready protocol (**§12.4**). |
| `06_nb_ready` | Naive Bayes Router Fusion (**appendix**) | Same ready protocol (**§12B.1**). |
| `06_lr_ready` | Logistic Regression Router Fusion (**appendix**) | Same ready protocol (**§12B.2**). |
| `06_rf_ready` | Random Forest Router Fusion (**appendix**) | Same ready protocol (**§12B.3**). |
| `06_mlp_ready` | MLP Router Fusion (**appendix**) | Same ready protocol (**§12B.4**). |

**Primary** ML routers: `fusion_router_oof_common.py` — see also [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md).  
**Appendix** ready routers: `fusion_router_ready_common.py` (no NER retraining; router fit on same eval tokens as scoring).

**Optional extension — Experiment 10 (BERT-CRF track):** same cross-comparison runner, separate base cache, adds CRF decoding and CRF fusion (see **Section 12A** and `experiments/experiment_10_README.md`).

| Experiment ID | Method name | Core idea |
|---|---|---|
| `10_regular` | Regular BERT-CRF | Exp01-style single pass, but emissions + **linear-chain CRF** (Viterbi decode); **O-bias = 6**. |
| `10_cascade` | Cascaded + CRF | Exp04-style three heads **plus** full-tag CRF head; Step-3 **B/I consistency** after decode. |
| `10_fusion_ready` | CRF confidence fusion | Fuses `10_regular` and `10_cascade` ready Excel outputs (no retraining). |
| `10_svm_ready` | CRF linear SVM router | Same as `06_svm_ready` on CRF sources (**§12A.4**). |
| `10_svm_kernel_ready` | CRF kernel SVM router | RBF `SVC` on CRF disagreements (**§12.5**). |
| `10_nb_ready` | CRF Naive Bayes router | GaussianNB (**§12B.1**). |
| `10_lr_ready` | CRF logistic regression router | (**§12B.2**). |
| `10_rf_ready` | CRF random forest router | (**§12B.3**). |
| `10_mlp_ready` | CRF MLP router | (**§12B.4**). |

The selected models are:

| Model key | Model name | Scientific role |
|---|---|---|
| `dictabert` | DictaBERT | General-purpose Hebrew BERT baseline. |
| `berel` | BEREL 3.0 | Hebrew model with Biblical/Rabbinical orientation. |
| `hero` | HeRo | Hebrew RoBERTa-style model. |
| `alephbertgimmel` | AlephBERT-Gimmel | Hebrew BERT-family model. |
| `xlm_roberta` | XLM-RoBERTa-base | **Multilingual** RoBERTa (100 languages); strong cross-lingual encoder baseline for Hebrew NER without Hebrew-only pretraining. |
| `mt5` | mT5-base | **Multilingual** T5 (101 languages); compared using the **encoder stack + token-classification head** (same Exp01/04 head design as BERT-family models). |

**Preset strings** (see `run_cross_data_model_comparison.py`):

| Preset | Value |
|---|---|
| Hebrew only | `dictabert,berel,hero,alephbertgimmel` |
| Multilingual only | `xlm_roberta,mt5` |
| Full comparison (Hebrew + multilingual) | `dictabert,berel,hero,alephbertgimmel,xlm_roberta,mt5` |

Multilingual models use the same token-classification and cascaded-pipeline code paths as Hebrew encoders (`core/model_backbone.py` loads mT5 via `MT5EncoderModel` when needed).

The selected condition sources are:

| Source | Meaning |
|---|---|
| `exp07` | Sentence split strategies without augmentation. |
| `exp07+aug` | The same Exp07 split variants, but the training side is augmented using the Exp08 LLM mask-fill augmentation mechanism. |

The command uses `--num-seeds 20`, so the design is repeated with **20 fixed training seeds** $S$:

1. On the first run for a given `--output-dir`, the runner draws **20 random unique integers** and saves them to `{output-dir}/training_seeds.json`.
2. On every later run (`--resume` or full rerun), the **same** list is loaded from that file so `THESIS_SPLIT_SEED` values stay stable.

Optional: `THESIS_CROSS_SEEDS_MASTER=<int>` when creating the file (reproducible draw); `--regenerate-training-seeds` to replace the list (requires rebuilding exp07+aug artifacts if used).

**Growing the seed count mid-study (e.g. 10 → 20) without losing completed runs:** `--regenerate-training-seeds` draws a **brand-new fully random** list, discarding the old one — every run already completed under the original seeds becomes orphaned (new random seeds almost never match the old ones). To add seeds instead, append to `training_seeds.json` in place: keep all existing entries, draw additional unique random integers up to the new `--num-seeds` target, and write back the same `{"seeds": [...], "num_seeds": N}` schema, leaving `--regenerate-training-seeds` off. Checkpoint rows are keyed by seed value, so a plain `--resume` afterward trains only the newly-added seeds and reuses every already-completed run unchanged. See `thesis_overview_150_sentences.md` §4.3 for the exact snippet (used by `colab_150_sentences_journal_10seeds.ipynb` cell "7a2").

This makes the experiment a paired repeated-measures design rather than a one-off model run.

**IEEE journal profile (1–3 training seeds by default, 5-fold OOF fusion, journal export sheets):** see [`thesis_overview_journal_paper_results.md`](thesis_overview_journal_paper_results.md). `--journal-paper` defaults to `--num-seeds 3` only when `--num-seeds` is **not** explicitly passed; pass `--num-seeds 20` (or any value) explicitly to use more seeds under the journal profile — this is now honored correctly (previously, an explicit value that happened to equal the full-thesis default of `20` collided with the default-detection logic and was silently reset to `3`; fixed in `run_cross_data_model_comparison.py`). Primary significance is **paired across `training_seed × outer_fold`** in the `paired_fold_tests` sheet of `unified_protocol_table.xlsx` (Wilcoxon + t-test, Holm-adjusted, all fusion methods vs. Regular NER and Cascade NER) — more seeds directly increases the number of paired observations, so this is not an either/or against 5-fold OOF. Build it with `build_unified_protocol_table.py`; the runner's `journal_paired_fold_deltas` sheet is **not** emitted (Part V §V.1 explains why).

### 1.1 Full corpus vs 150-sentence pilot (isolated outputs)

Two standard cross-comparison profiles share the **same model list** and experiments; they differ only in corpus size and output folder:

| Profile | Corpus | CLI |
|---|---|---|
| **Full labeled set** (~300 sentences) | `data/ner_dataset.csv` | Default paths; omit `--subset-sentences`. Use `--output-dir outputs/cross_comparison_full` (recommended) so checkpoints do not mix with pilot runs. |
| **150-sentence pilot** | Random 150 sentence ids (fixed by `--subset-seed`) | `--subset-sentences 150 --subset-seed 42 --output-dir outputs/cross_comparison_150sent` (splits + subset CSV live under that folder). **Methods:** [`thesis_overview_150_sentences_pilot.md`](thesis_overview_150_sentences_pilot.md); **OOF routers:** [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md). |
| **Full corpus OOF** (~300 sentences) | Full `ner_dataset.csv` | Same OOF protocol; ~60 sentences per outer fold. [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md). |

**Full corpus (all six models, 20 seeds, no augmentation, no CRF):**

```bash
python run_cross_data_model_comparison.py \
  --output-dir outputs/cross_comparison_full \
  --models dictabert,berel,hero,alephbertgimmel,xlm_roberta,mt5 \
  --experiments 01,04,05_ready,06_ready,06_svm_oof \
  --skip-augmentation \
  --condition-sources exp07 \
  --exp07-source auto \
  --base-mode auto \
  --num-seeds 20 \
  --consolidated-error-analysis all
```

**150-sentence pilot (same models and exports; resume with `--resume`):**

```bash
python run_cross_data_model_comparison.py \
  --output-dir outputs/cross_comparison_150sent \
  --subset-sentences 150 \
  --subset-seed 42 \
  --models dictabert,berel,hero,alephbertgimmel,xlm_roberta,mt5 \
  --experiments 01,04,05_ready,06_ready,06_svm_oof \
  --skip-augmentation \
  --condition-sources exp07 \
  --exp07-source rerun \
  --base-mode auto \
  --num-seeds 20 \
  --consolidated-error-analysis all
```

**Fair training (all models):** The cross-comparison runner applies one profile automatically
(`core/training_defaults.py`). Full tables, Colab snippets, and resume instructions:
****Part II** (below)**.

Reference (matches completed DictaBERT/BEREL 150-sent Colab run): Exp01 **3 epochs**, **5e-5** LR,
fp16 on, weight decay **0**; Exp04 **10 epochs** (encoder **2e-5**), cascaded loss weights
$\lambda_{bio}=10$, $\lambda_{type}=5$ unless replaced after the validation grid (**§9.4**). Re-run XLM-R/mT5 with wrong
checkpoint rows cleared if they used older multilingual-only settings.

Artifacts for the pilot: `{output-dir}/data/ner_dataset_150_seed42.csv`, `{output-dir}/exp07/splits/`, `{output-dir}/run_manifest.json`, plus `cross_comparison_*.xlsx/json` and consolidated error-analysis workbooks. The full-corpus run does **not** overwrite `outputs/exp07/splits/` when the pilot uses its own `--output-dir` and subset flags.

---

## 2. Beginner-Friendly View of the Whole Pipeline

Think of the pipeline as a large table of jobs. Each job has four coordinates:

$$
\text{job} = (m, e, c, s),
$$

where:

- $m$ is the model,
- $e$ is the experiment/method,
- $c$ is the data condition,
- $s$ is the random seed.

For this command:

$$
M = \{\text{DictaBERT}, \text{BEREL}, \text{HeRo}, \text{AlephBERT-Gimmel}\},
$$

$$
E = \{01,04,05\_ready,06\_ready,06\_svm\_oof\},
$$

$$
S = \{42,\ldots,61\}.
$$

The data condition set $C$ is built from saved Exp07 split variants and Exp07+Aug variants. If three Exp07 variants are available and three matching Exp07+Aug variants are available, then:

$$
|C| = 3 + 3 = 6.
$$

The number of seeded conditions is:

$$
|C_S| = |C| \times |S|.
$$

If $|C|=6$ and $|S|=20$:

$$
|C_S| = 6 \times 20 = 120.
$$

The maximum number of result rows is:

$$
|M| \times |E| \times |C_S| = 4 \times 5 \times 120 = 2400.
$$

A beginner can understand the command as:

1. prepare split and augmentation files,
2. for every model,
3. for every experiment,
4. for every data condition,
5. for every seed,
6. run or reuse the method,
7. record F1, precision, recall,
8. aggregate results,
9. test whether differences are statistically meaningful.

---

## 3. Data Representation

The corpus is a Hebrew NER dataset stored as token-level BIO labels. Each token has one label:

$$
y_i \in \mathcal{Y},
$$

where:

$$
\mathcal{Y} = \{O\} \cup \{B\text{-}t, I\text{-}t : t \in \mathcal{T}\}.
$$

Here:

- $O$ means the token is not part of an entity,
- $B\text{-}t$ means the token begins an entity of type $t$,
- $I\text{-}t$ means the token continues an entity of type $t$,
- $\mathcal{T}$ is the set of entity types, such as `PER`, `LOC`, `ORG`, or other dataset-specific types.

A sentence is represented as:

$$
x^{(j)} = (w_1, w_2, \ldots, w_n),
$$

with labels:

$$
y^{(j)} = (y_1, y_2, \ldots, y_n).
$$

The saved split JSON format stores each sentence as:

```json
{
  "text": "token1 token2 token3",
  "labels": ["B-PER", "I-PER", "O"]
}
```

This matters because the pipeline splits at sentence level, not token level. Sentence-level splitting prevents leakage where part of a sentence appears in training and another part appears in evaluation.

---

## 4. Train/Evaluation Split Architecture

The command uses condition sources `exp07` and `exp07+aug`.

### 4.1 Exp07: Split Strategy Conditions

Exp07 creates different ways of dividing the same sentence set into training and evaluation partitions.

Let the full sentence set be:

$$
D = \{(x^{(j)}, y^{(j)})\}_{j=1}^{N}.
$$

Each split condition creates:

$$
D_{train}^{(c,s)} \subset D,
$$

$$
D_{eval}^{(c,s)} = D \setminus D_{train}^{(c,s)},
$$

with approximate ratio:

$$
\frac{|D_{train}^{(c,s)}|}{|D|} \approx 0.7,
\qquad
\frac{|D_{eval}^{(c,s)}|}{|D|} \approx 0.3.
$$

The split is sentence-level:

$$
D_{train}^{(c,s)} \cap D_{eval}^{(c,s)} = \varnothing.
$$

The split variants include ideas such as:

1. **Simple random split**: shuffle sentences and take the first 70% as training.
2. **Label-aware greedy split**: choose sentences so rare non-`O` labels are preserved in the training set.
3. **Paper-style iterative multilabel stratification**: treat each sentence as a set of labels and assign it to train/eval while preserving label proportions.

### 4.2 Simple Random Split

The simplest split chooses a random permutation:

$$
\pi_s(D),
$$

where $s$ is the seed. Then:

$$
D_{train} = \pi_s(D)_{1:\lfloor 0.7N \rfloor},
$$

$$
D_{eval} = \pi_s(D)_{\lfloor 0.7N \rfloor + 1:N}.
$$

This is easy but risky. If a rare label appears in only a few sentences, all of those sentences may accidentally go to evaluation, leaving no training examples for that label.

### 4.3 Label-Aware Split

The label-aware split tries to preserve non-`O` entity label coverage.

For each label $\ell \neq O$, define the count in a sentence subset $A$:

$$
count_A(\ell) = \sum_{(x,y) \in A} \sum_{i=1}^{|x|} \mathbf{1}[y_i = \ell].
$$

The target training count is approximately:

$$
target_{train}(\ell) = 0.7 \cdot count_D(\ell).
$$

A natural split objective is to minimize distribution mismatch:

$$
\mathcal{L}_{split}(A)
=
\sum_{\ell \in \mathcal{Y}\setminus\{O\}}
\left(count_A(\ell) - target_{train}(\ell)\right)^2.
$$

The algorithm greedily builds a training set that keeps this loss small and tries to ensure that rare entity labels appear in the training set.

### 4.4 Multilabel Iterative Split

The paper-style iterative strategy treats each sentence as a multilabel item. A sentence may contain several entity labels, so it receives a set:

$$
L_j = \{\ell : \ell \neq O, \ell \text{ appears in sentence } j\}.
$$

The algorithm prioritizes rare labels first. For each label $\ell$, it estimates desired train/eval counts:

$$
desired_{train}(\ell) = 0.7 \cdot n_\ell,
$$

$$
desired_{eval}(\ell) = 0.3 \cdot n_\ell,
$$

where $n_\ell$ is the number of sentences containing label $\ell$.

It assigns sentences to the fold with the greater remaining need. For fold $f \in \{train, eval\}$, define:

$$
need_f(j) = \sum_{\ell \in L_j}
\left(desired_f(\ell) - current_f(\ell)\right).
$$

The chosen fold is:

$$
f^*(j) = \arg\max_f need_f(j),
$$

with deterministic seed-based tie-breaking.

---

## 5. Exp07+Aug: Augmented Split Conditions

The `exp07+aug` condition source starts with each Exp07 split variant and augments only the training set.

For a condition $c$ and seed $s$:

$$
D_{train}^{aug(c,s)} = D_{train}^{(c,s)} \cup G^{(c,s)},
$$

where $G^{(c,s)}$ is a generated set of synthetic training sentences.

The evaluation set is not augmented:

$$
D_{eval}^{aug(c,s)} = D_{eval}^{(c,s)}.
$$

This is a critical constraint. Augmenting evaluation data would change the test target and make results unfair. The method only increases the training examples.

### 5.1 Rare-Label Motivation

For each entity label $\ell$, define sentence frequency:

$$
f(\ell) = \sum_{j=1}^{N} \mathbf{1}[\ell \in L_j].
$$

Rare labels have smaller $f(\ell)$. The augmentation method tries to reduce imbalance by generating more examples for underrepresented labels.

Let:

$$
f_{max} = \max_{\ell} f(\ell).
$$

A simple deficit score is:

$$
\Delta(\ell) = f_{max} - f(\ell).
$$

The generation multiplier $r$ is controlled by `THESIS_EXP08_MULTIPLIER`, defaulting to $3$. The approximate target number of generated examples for label $\ell$ is:

$$
g(\ell) \approx r \cdot \Delta(\ell).
$$

### 5.2 Mask-Fill Generation

The augmentation mechanism uses a masked-language-model style approach. For a sentence containing an entity token, the method masks a relevant position and asks the language model to propose replacements.

A sentence:

$$
x = (w_1,\ldots,w_k,\ldots,w_n)
$$

is transformed into:

$$
x_{mask} = (w_1,\ldots,[MASK],\ldots,w_n).
$$

The language model estimates:

$$
P(w \mid x_{mask}).
$$

Candidate replacements are selected from entity-compatible vocabulary items. The generated sentence keeps the context but varies the entity surface form.

The final augmented training set is:

$$
D_{train}^{aug} = D_{train} \cup \{\tilde{x}_1,\tilde{x}_2,\ldots,\tilde{x}_K\}.
$$

The labels for generated sentences preserve the intended BIO structure. The purpose is not to invent a new evaluation target, but to expose the model to more rare-label contexts during training.

---

## 6. Model Resolution and Reproducibility Controls

For every selected model key, the runner maps it to a model identifier:

$$
m \mapsto \text{HuggingFace model ID or local model path}.
$$

The runner prefers local model files when available. If local files are found, it sets offline flags so the transformer library does not unnecessarily download model files.

For each run, the following environment variables are used to bind the scientific context:

| Variable | Meaning |
|---|---|
| `THESIS_MODEL_NAME` | Which transformer model to use. |
| `THESIS_SPLIT_SEED` | Which random seed controls the current condition. |
| `THESIS_PRESPLIT_TRAIN_JSON` | Exact train split file. |
| `THESIS_PRESPLIT_EVAL_JSON` | Exact evaluation split file. |
| `THESIS_CURRENT_EXP_ID` | Current experiment ID for checkpoint/model saving. |
| `THESIS_CURRENT_CONDITION_KEY` | Current condition key for artifact isolation. |
| `THESIS_READY_EXP01_XLSX` | Exp01 output file used by ready experiments. |
| `THESIS_READY_EXP04_XLSX` | Exp04 output file used by ready experiments. |
| `THESIS_EXP04_LAMBDA_BIO` | Weight on boundary (B/I) loss in Exp04; default **10** (see **§9.4**). |
| `THESIS_EXP04_LAMBDA_TYPE` | Weight on entity-type loss in Exp04; default **5** (see **§9.4**). |
| `THESIS_EXP04_LAMBDA_GRID_VALUES` | Optional comma list for the validation grid (default `1,5,10`). |

The important beginner idea is:

> Ready experiments do not guess which previous output to use. The runner explicitly points them to the matching Exp01 and Exp04 files for the same model, condition, and seed.

---

## 7. Artifact Reuse and `--base-mode auto`

Experiments `01` and `04` are expensive because they train transformer-based systems. Experiments `05_ready` and `06_ready` are cheaper because they operate on already saved predictions. **`06_*_oof`** is the most expensive fusion path (repeated Exp01 + Exp04 training inside nested CV); **`06_*_ready`** is cheap appendix inference on cached bases.

**Exp04 loss weights:** Before locking multi-seed cross-comparison runs, run the validation grid once (`experiments/experiment_04_loss_weight_grid.py`) and export the selected `THESIS_EXP04_LAMBDA_BIO` / `THESIS_EXP04_LAMBDA_TYPE` for all later Exp04 training (including OOF retrains). Details: **§9.4**, [`cross_comparison_fair_training_overview.md`](cross_comparison_fair_training_overview.md), [`thesis_overview_150_sentences_pilot.md`](thesis_overview_150_sentences_pilot.md) §8.

Experiment **10** adds another **training** pair (`10_regular`, `10_cascade`) with the same cost profile as `01`/`04`, plus **inference-only** fusion IDs (`10_fusion_ready`, `10_svm_ready`) analogous to the 06 ready track. Base artifacts are stored in `cross_comparison_base_crf_ready_index.json` (separate from the Exp01/Exp04 cache).

The command uses:

```text
--base-mode auto
```

This means:

1. if valid Exp01 and Exp04 artifacts already exist for the same model and condition, reuse them;
2. otherwise, train Exp01 and Exp04 once;
3. save their output paths in the base artifact index;
4. pass those output paths into ready experiments.

The cache key is conceptually:

$$
k = (m, c, path(D_{train}), path(D_{eval})).
$$

An artifact is valid only if all required files exist:

$$
\text{valid}(k) =
\mathbf{1}[F_{01}^{xlsx} \land F_{01}^{json} \land F_{04}^{xlsx} \land F_{04}^{json}].
$$

The `--resume` flag adds another layer. If a previous cross-comparison checkpoint already contains a successful row for a given:

$$
(m,e,c,s),
$$

that row is skipped. Failed rows are not treated as complete and may be retried.

---

## 8. Experiment 01: Regular Transformer NER

Experiment `01` is the baseline direct NER architecture.

### 8.1 Input and Output

Input sentence:

$$
x = (w_1,w_2,\ldots,w_n).
$$

Gold labels:

$$
y = (y_1,y_2,\ldots,y_n), \qquad y_i \in \mathcal{Y}.
$$

The transformer tokenizer may split words into subword pieces. The model predicts labels at token/subtoken level, then outputs token-level predictions aligned back to the dataset.

### 8.2 Transformer Encoder

Each token is converted into a contextual vector:

$$
H = Transformer_m(x),
$$

where:

$$
H = (h_1,h_2,\ldots,h_n), \qquad h_i \in \mathbb{R}^d.
$$

Here $m$ is one of the selected Hebrew transformer models.

### 8.3 Token Classification Head

A linear classification head maps each hidden vector to label logits:

$$
z_i = W h_i + b.
$$

The probability of label $k$ is computed with softmax:

$$
P(y_i=k \mid x) =
\frac{\exp(z_{i,k})}{\sum_{k' \in \mathcal{Y}} \exp(z_{i,k'})}.
$$

The predicted label is:

$$
\hat{y}_i = \arg\max_{k \in \mathcal{Y}} P(y_i=k \mid x).
$$

### 8.4 Training Objective

The model is trained by minimizing cross-entropy over valid tokens:

$$
\mathcal{L}_{NER}
=
-\sum_{i=1}^{n}
\log P(y_i \mid x).
$$

In implementation, special tokens and ignored subtokens receive label `-100`, so they do not contribute to the loss:

$$
\mathcal{L}_{NER}
=
-\sum_{i: y_i \neq -100}
\log P(y_i \mid x).
$$

### 8.5 Confidence Features Exported for Fusion

Exp01 exports more than the final label. For each token, it saves:

1. predicted label,
2. maximum probability,
3. entropy,
4. probability margin.

The maximum probability is:

$$
p_i^{reg} = \max_k P(y_i=k \mid x).
$$

Entropy is:

$$
H_i^{reg} = -\sum_{k \in \mathcal{Y}} P(y_i=k \mid x)\log(P(y_i=k \mid x)+\epsilon).
$$

The margin is the difference between the top two probabilities:

$$
margin_i^{reg} = p_{i,(1)} - p_{i,(2)}.
$$

A large margin means the model strongly prefers its top label over the second-best label.

---

## 9. Experiment 04: Cascaded Multi-Step NER

Experiment `04` decomposes NER into three simpler prediction problems.

Instead of directly predicting full labels like `B-PER`, the model predicts:

1. whether a token is an entity,
2. whether an entity token is `B` or `I`,
3. which entity type the token has.

This is called a cascaded architecture because the final prediction depends on several steps.

### 9.1 Label Decomposition

A full BIO label $y_i$ is decomposed into:

$$
e_i \in \{0,1\},
$$

$$
b_i \in \{0,1\},
$$

$$
t_i \in \mathcal{T}.
$$

Where:

- $e_i=1$ means token $i$ is part of an entity,
- $e_i=0$ means token $i$ is outside an entity,
- $b_i=1$ means `B`,
- $b_i=0$ means `I`,
- $t_i$ is the entity type.

For label `O`:

$$
e_i=0, \qquad b_i=-100, \qquad t_i=-100.
$$

The value `-100` means “ignore this token for that task.”

### 9.2 Shared Encoder with Three Heads

The cascaded model uses one shared transformer encoder:

$$
h_i = Encoder_m(x)_i.
$$

Then it applies three heads:

Entity detection head:

$$
z_i^{entity} = W_e h_i + b_e,
$$

BIO head:

$$
z_i^{bio} = W_b h_i + b_b,
$$

Type head:

$$
z_i^{type} = W_t h_i + b_t.
$$

The probabilities are:

$$
p_i^{entity} = \sigma(z_i^{entity}),
$$

$$
p_i^{bio} = \sigma(z_i^{bio}),
$$

$$
P(t_i=k \mid x) =
\frac{\exp(z_{i,k}^{type})}{\sum_{k' \in \mathcal{T}}\exp(z_{i,k'}^{type})}.
$$

Here $\sigma$ is the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### 9.3 Cascaded Decision Rule

The pipeline converts probabilities into decisions using thresholds.

Entity decision:

$$
\hat{e}_i =
\begin{cases}
1, & p_i^{entity} \geq \tau_e,\\
0, & p_i^{entity} < \tau_e.
\end{cases}
$$

BIO decision:

$$
\hat{b}_i =
\begin{cases}
B, & p_i^{bio} \geq \tau_b,\\
I, & p_i^{bio} < \tau_b.
\end{cases}
$$

Type decision:

$$
\hat{t}_i = \arg\max_{k \in \mathcal{T}} P(t_i=k \mid x).
$$

Final label:

$$
\hat{y}_i^{cas}=
\begin{cases}
O, & \hat{e}_i=0,\\
\hat{b}_i\text{-}\hat{t}_i, & \hat{e}_i=1.
\end{cases}
$$

### 9.4 Masked Conditional Training

The architecture trains later heads only where they make sense.

Entity loss is computed for all valid tokens:

$$
\mathcal{L}_{entity} = \sum_{i:e_i\neq -100} \ell_{bin}(z_i^{entity}, e_i).
$$

BIO loss is computed only for true entity tokens:

$$
\mathcal{L}_{bio} = \sum_{i:b_i\neq -100} \ell_{bin}(z_i^{bio}, b_i).
$$

Type loss is also computed only for true entity tokens:

$$
\mathcal{L}_{type} = \sum_{i:t_i\neq -100} \ell_{CE}(z_i^{type}, t_i).
$$

The total loss is weighted:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{entity}
+ \lambda_{bio}\mathcal{L}_{bio}
+ \lambda_{type}\mathcal{L}_{type}.
$$

Each $\mathcal{L}_{\cdot}$ is a **mean** over its supervised tokens (entity loss on all valid tokens; boundary and type losses only on gold entity tokens). The $\lambda$ factors trade off gradient contribution through the shared encoder, not raw token counts in a sum.

Default implementation values (used unless overridden by the validation grid):

$$
\lambda_{bio}=10,
\qquad
\lambda_{type}=5.
$$

**Validation grid (loss-weight selection).**  
The coefficients are not chosen by hand alone. On the same 70/30 sentence split used for threshold tuning, we run a coarse grid

$$
\lambda_{bio}, \lambda_{type} \in \{1, 5, 10\}
$$

(nine configurations) and select the pair that maximizes **validation pipeline span F1** in predicted mode after the internal threshold sweep—the same endpoint as the main cascaded evaluation. The held-out test set is not used for this search.

Reproduce the grid (use the same `THESIS_SPLIT_SEED` / Exp07 split as the study you are calibrating for):

```bash
export THESIS_SPLIT_SEED=42
python experiments/experiment_04_loss_weight_grid.py
```

PowerShell: `$env:THESIS_SPLIT_SEED = "42"`. Optional smoke run: `THESIS_EXP04_FAST=1`.

Outputs are written under `outputs/exp04_lambda_grid/` (`grid_runs`, `heatmap_f1`, `selection` sheets + JSON). Each Exp04 metrics workbook also records the pair used in sheet **`loss_config`**.

Apply the selected pair to all full multi-seed Exp04 runs (and document it in `run_manifest.json` via env before launching the cross-comparison runner):

```bash
export THESIS_EXP04_LAMBDA_BIO=<selected>
export THESIS_EXP04_LAMBDA_TYPE=<selected>
```

Environment overrides are read in `core/auc_cascaded_pipeline.py` (`THESIS_EXP04_LAMBDA_BIO`, `THESIS_EXP04_LAMBDA_TYPE`). If the grid selects $(10,5)$ or lies on a flat region near those defaults, existing main results remain valid; the grid still supplies the ablation evidence for the paper.

### 9.5 Focal Loss for Imbalance

The binary heads can use focal loss. This is useful because most tokens are usually `O`, so easy negative examples can dominate training.

For a binary target $y \in \{0,1\}$ and predicted probability $p$, define:

$$
p_t =
\begin{cases}
p, & y=1,\\
1-p, & y=0.
\end{cases}
$$

Focal loss is:

$$
\mathcal{L}_{focal}
= -\alpha_t(1-p_t)^\gamma \log(p_t).
$$

With:

$$
\alpha=0.25,
\qquad
\gamma=2.0.
$$

Beginner explanation: if a token is already easy, then $p_t$ is high and $(1-p_t)^\gamma$ becomes small. The loss focuses more on hard examples.

### 9.6 Threshold Optimization

After training, the cascaded pipeline searches over threshold values:

$$
\tau_e, \tau_b \in \{0.10,0.15,0.20,\ldots,0.90\}.
$$

It chooses the pair that maximizes a combined score:

$$
(\tau_e^*,\tau_b^*)
=
\arg\max_{\tau_e,\tau_b}
\left(F1_{entity}(\tau_e)+F1_{bio}(\tau_b)\right).
$$

The final exported result uses the optimized thresholds.

### 9.7 BIO Constraint Enforcement

The cascaded system also repairs invalid BIO transitions. For example, an `I` tag should not start an entity immediately after `O`.

If:

$$
\hat{b}_i = I
\quad\text{and}\quad
(i=1 \text{ or } \hat{b}_{i-1}=O),
$$

then it changes:

$$
\hat{b}_i := B.
$$

This turns an invalid continuation into a valid beginning.

---

## 10. Experiment 05_ready: Cascaded B/I Consistency Post-Processing

Experiment `05_ready` does not train a new model. It loads Exp04 predictions and applies a local correction rule.

### 10.1 The Problem

A cascaded model can predict inconsistent neighboring labels, such as:

```text
B-PER I-LOC
```

This says: “start a person entity, then continue it as a location entity,” which is structurally inconsistent.

The problematic pattern is:

$$
\hat{b}_i = B,
\qquad
\hat{b}_{i+1} = I,
\qquad
\hat{t}_i \neq \hat{t}_{i+1}.
$$

### 10.2 Consistency Rule

Let $q_i$ be the BIO confidence for token $i$, saved as `bio_prob`.

If a `B-X` token is followed by `I-Y` and $X \neq Y$, the method trusts the token with higher BIO confidence.

Formally:

$$
(\hat{t}_i^*, \hat{t}_{i+1}^*) =
\begin{cases}
(\hat{t}_i, \hat{t}_i), & q_i \geq q_{i+1},\\
(\hat{t}_{i+1}, \hat{t}_{i+1}), & q_i < q_{i+1}.
\end{cases}
$$

Beginner explanation:

- if the `B` token is more confident, force the following `I` token to use the same entity type;
- if the `I` token is more confident, change the `B` token to match the `I` token type.

The method then reconstructs BIO labels and recomputes entity-level F1.

---

## 11. Experiment 06_ready: Confidence Fusion of Regular and Cascaded NER

Experiment `06_ready` combines two different prediction sources:

1. regular NER from Exp01,
2. cascaded NER from Exp04.

It does not retrain either source model.

### 11.1 Token Alignment

Exp01 and Exp04 outputs are inner-joined by:

$$
(sentence\_id, token\_idx).
$$

A token can be fused only if both systems produced a prediction for the same sentence and token index.

For token $i$:

- regular prediction: $\hat{y}_i^{reg}$,
- cascaded prediction: $\hat{y}_i^{cas}$,
- regular confidence: $p_i^{reg}$ (stored as `regular_prob`),
- cascaded confidence: $p_i^{cas}$ (stored as `cascade_prob`).

Implementation: `experiments/fusion_ready_sources.py` (`load_regular_from_exp01`, `load_cascade_from_exp04`, `merge_regular_cascade`).

### 11.2 Per-Source Confidence (Canonical Definitions)

Ready fusion (`06_ready`, `06_svm_ready`, and Exp10 analogues via the same loader) always compares two **scalar** scores per aligned token. Other columns (entropy, margin) are exported for analysis and for the SVM router, but **confidence fusion itself uses only** $p_i^{reg}$ and $p_i^{cas}$.

#### 11.2.1 Regular path (Exp01 softmax NER, or Exp10 regular CRF via the same sheet layout)

At evaluation time each valid word token gets a tag distribution from the final linear head (softmax over BIO-type labels).

**Exp01 (`experiment_01_regular_ner.py`, `token_predictions` sheet):**

- Softmax probabilities $P(y_i=k\mid x)$ over the full tag set.
- Predicted tag: $\hat{y}_i^{reg} = \arg\max_k P(y_i=k\mid x)$.
- Fusion confidence (column `prob` → `regular_prob`):

$$
p_i^{reg} = \max_k P(y_i=k\mid x).
$$

**Also exported (not used by confidence fusion):**

$$
H_i^{reg} = -\sum_k P(y_i=k\mid x)\log(P(y_i=k\mid x)+\epsilon),
\qquad
margin_i^{reg} = p_{i,(1)} - p_{i,(2)}.
$$

**Exp10 regular BERT-CRF (`experiment_10_regular_ner_crf.py`):** decoding is **Viterbi** on the CRF, but confidence for fusion still comes from a **local emission softmax** at the token. Let $\hat{y}_i^{reg}$ be the Viterbi tag and $P_em(k\mid x)$ the softmax over emission scores at that position:

$$
p_i^{reg} = P_em(\hat{y}_i^{reg}\mid x).
$$

This is the probability of the **decoded** tag, which can differ from $\max_k P_em(k\mid x)$ when the CRF path overrides the local argmax.

**Legacy Exp01 workbooks** without `token_predictions` receive neutral placeholders (`regular_prob=0.5`) so fusion scripts still run; those scores are not meaningful for calibration.

#### 11.2.2 Cascaded path (Exp04 / Exp10 cascade, `detailed_results`, `eval_mode=predicted`)

The cascade writes per-token `entity_prob` and `bio_prob` from the entity and B/I heads (sigmoid scores at decision time; see §9.3). Let $p_i^{entity}$ and $p_i^{bio}$ denote those stored values. The fusion loader builds full BIO-type labels from `pred_bio` + `pred_etype`, then sets a **single** cascaded confidence:

If the cascaded prediction is `O` (`pred_bio = O`):

$$
p_i^{cas} = 1 - p_i^{entity}.
$$

If the cascaded prediction is an entity (`B` or `I`):

$$
p_i^{cas} = p_i^{entity} \cdot p_i^{bio}.
$$

Interpretation: for `O`, confidence is “how strongly non-entity”; for entities, confidence requires both entity detection and B/I position to be confident (product of the two sigmoid probabilities).

**Derived in the loader (not used by confidence fusion):** binary entropy and a margin on $p_i^{cas}$ treated as a Bernoulli parameter:

$$
H_i^{cas} = -p_i^{cas}\log(p_i^{cas}+\epsilon) - (1-p_i^{cas})\log(1-p_i^{cas}+\epsilon),
$$

$$
margin_i^{cas} = 2\,|p_i^{cas} - 0.5|.
$$

#### 11.2.3 Merged-table extras used later by the SVM router

After the inner join, the loader also computes:

$$
prob\_diff_i = p_i^{reg} - p_i^{cas},
\quad
|prob\_diff|_i,
\quad
\max\_prob_i = \max(p_i^{reg}, p_i^{cas}),
$$

and splits each side’s predicted label into BIO + entity-type parts (`regular_bio`, `regular_etype`, `cascade_bio`, `cascade_etype`).

### 11.3 Confidence Fusion Rule

If both systems agree:

$$
\hat{y}_i^{fused} = \hat{y}_i^{reg} = \hat{y}_i^{cas}.
$$

If they disagree, choose the label from the more confident system:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & p_i^{reg} \geq p_i^{cas},\\
\hat{y}_i^{cas}, & p_i^{reg} < p_i^{cas}.
\end{cases}
$$

This is a simple but scientifically meaningful ensemble rule. It assumes that confidence is a useful proxy for correctness.

**Recorded output:** `selected_confidence` is the confidence of the source that was chosen (`regular_prob` or `cascade_prob`). On agreement tokens both sources share the same label; the implementation still stores the regular-side confidence when it picks the shared label.

---

## 12. ML Router Fusion (Ready): Shared Protocol

Experiments `06_svm_ready`, `06_svm_kernel_ready`, `06_nb_ready`, `06_lr_ready`, `06_rf_ready`, and `06_mlp_ready` (and Exp10 analogues `10_*`) use a **learned meta-classifier** instead of a fixed confidence rule.

The central idea is:

> When Exp01 and Exp04 disagree, learn which source is more likely to be correct.

### 12.1 Disagreement Tokens

A disagreement token is one where:

$$
\hat{y}_i^{reg} \neq \hat{y}_i^{cas}.
$$

The router is trained only on disagreement tokens where exactly one source is correct.

Define:

$$
r_i = \mathbf{1}[\hat{y}_i^{reg} = y_i],
$$

$$
c_i = \mathbf{1}[\hat{y}_i^{cas} = y_i].
$$

The router target is:

$$
z_i =
\begin{cases}
regular, & r_i=1 \land c_i=0,\\
cascade, & r_i=0 \land c_i=1,\\
\text{discard}, & \text{otherwise}.
\end{cases}
$$

Ambiguous cases are discarded:

- both correct,
- both wrong.

### 12.2 Router Features

The disagreement router does **not** see the raw token string, sentence id, or gold label. It sees only **calibrated-style scores and predicted label parts** from Exp01 (regular) and Exp04 (cascade) on tokens where the two systems were successfully aligned (`merge_regular_cascade` inner join on `(sentence_id, token_idx)`).

Feature definitions are centralized in `experiments/fusion_ready_sources.py` as `FUSION_ROUTER_NUMERIC_FEATURES` and `FUSION_ROUTER_CATEGORICAL_FEATURES` (legacy aliases `SVM_ROUTER_*`). All ready ML routers use `experiments/fusion_router_ready_common.py` with the same schema.

**When features are used:**

- **Training:** only rows with `disagree = True` **and** a non-ambiguous router target (§12.1).
- **Routing at fusion time:** the fitted pipeline’s `predict` is called on **all** disagreement rows (same feature columns).
- **Agreement rows:** no router features are needed; the fused label is the shared prediction.

**Entropy is exported** (`regular_entropy`, `cascade_entropy`) for analysis and other fusion variants, but **is not a router input** in the ready ML implementations.

#### 12.2.1 Numeric features (7 columns → `StandardScaler`)

After merge, seven scalar columns form $\phi_i^{num}$. Each is z-scored **on the router training matrix only** (sklearn `StandardScaler` inside the pipeline fit on usable disagreement tokens).

$$
\phi_i^{num} = [
 p_i^{reg},
 p_i^{cas},
 margin_i^{reg},
 margin_i^{cas},
 p_i^{reg}-p_i^{cas},
 |p_i^{reg}-p_i^{cas}|,
 \max(p_i^{reg},p_i^{cas})
].
$$

| Code column | Definition | Source / computation | Role for routing |
|---|---|---|---|
| `regular_prob` | $p_i^{reg}$ | Exp01 `token_predictions.prob` (or Exp10 regular sheet); §11.2.1 | How strongly the **direct** model backs its tag (softmax max or emission prob of decoded CRF tag). |
| `cascade_prob` | $p_i^{cas}$ | Built in `load_cascade_from_exp04`: $(1-p^{entity})$ if `pred_bio=O`, else $p^{entity}\cdot p^{bio}$; §11.2.2 | How strongly the **cascaded** pipeline backs its composed tag. |
| `regular_margin` | $p_{i,(1)}-p_{i,(2)}$ | Exp01 export: gap between top two softmax probabilities (`experiment_01_regular_ner.py`); legacy sheets default `0.0` | Separates “clear winner” vs “almost tied” on the regular side even when $p_i^{reg}$ is moderate. |
| `cascade_margin` | $2|p_i^{cas}-0.5|$ | Loader after $p_i^{cas}$ is fixed (`fusion_ready_sources.py`) | Treats cascaded confidence as a Bernoulli-style score in $[0,1]$; large value ⇒ far from chance. |
| `prob_diff` | $p_i^{reg}-p_i^{cas}$ | `merged["prob_diff"]` | Signed advantage of regular over cascade (same signal as §11.3, but the SVM can combine it with non-confidence cues). |
| `abs_prob_diff` | $|p_i^{reg}-p_i^{cas}|$ | `abs(prob_diff)` | Magnitude of confidence gap only (symmetric; useful when direction is encoded elsewhere). |
| `max_prob` | $\max(p_i^{reg},p_i^{cas})$ | Row-wise max of the two confidences | “Overall peaking” — both models very confident vs both cautious, independent of which side wins. |

Together, these let a **linear** router learn patterns such as “cascade is right when entity×bio product is high **and** regular margin is low **and** the two predicted entity types differ,” which pure §11.3 comparison cannot express.

#### 12.2.2 Categorical features (4 columns → `OneHotEncoder`)

Each full predicted label string (e.g. `B-PER`, `I-LOC`, `O`) is split into a **boundary** part and an **entity-type** part before encoding:

$$
\text{split}(y) =
\begin{cases}
(\texttt{O}, \texttt{None}), & y = \texttt{O},\\
(B\text{ or }I, \text{type}), & y = \texttt{B-type} \text{ or } \texttt{I-type}.
\end{cases}
$$

Implementation: `_split_label` in `merge_regular_cascade` for regular; cascade `cascade_bio` / `cascade_etype` come from Exp04 `pred_bio` / `pred_etype` (`etype` filled as `None` when absent).

| Code column | Meaning | Typical values |
|---|---|---|
| `regular_bio` | BIO prefix from $\hat{y}_i^{reg}$ | `O`, `B`, `I` |
| `regular_etype` | Entity type from $\hat{y}_i^{reg}$ | `PER`, `LOC`, `ORG`, … or `None` for `O` |
| `cascade_bio` | BIO prefix from $\hat{y}_i^{cas}$ | `O`, `B`, `I` |
| `cascade_etype` | Entity type from $\hat{y}_i^{cas}$ | same type inventory or `None` |

$$
\phi_i^{cat} = [
 \text{BIO}_i^{reg},
 \text{TYPE}_i^{reg},
 \text{BIO}_i^{cas},
 \text{TYPE}_i^{cas}
].
$$

(The four fields above are `regular_bio`, `regular_etype`, `cascade_bio`, `cascade_etype`.)

**Encoding:** `OneHotEncoder(handle_unknown="ignore")` on the four string columns. Each distinct training value becomes a binary indicator; at prediction time, **unseen** category levels (rare types in eval) contribute **no** active bit rather than crashing the router. There is **no** shared embedding and **no** manual feature cross — any interaction with numeric scores must be learned through the linear SVM weights on the concatenated one-hot and scaled numeric block.

**Why categoricals matter:** on disagreements, models often conflict on **structure** (e.g. `O` vs `B-PER`, or `B-PER` vs `B-LOC`) while confidences are similar. Label-part features tell the router *what kind* of disagreement it is, not only *how confident* each side is.

#### 12.2.3 Full vector and sklearn pipeline

The classifier input is the horizontally concatenated preprocessed blocks:

$$
\phi_i = [\phi_i^{num,\ \text{scaled}},\ \phi_i^{cat,\ \text{one-hot}}].
$$

```text
ColumnTransformer(
  ("num", StandardScaler(), 7 numeric columns),
  ("cat", OneHotEncoder(handle_unknown="ignore"), 4 categorical columns),
)
→ LinearSVC(C=1.0, class_weight="balanced", ...)
```

Constants: `FUSION_ROUTER_*` feature tuples and per-classifier `ROUTER_*_PARAMS` in `fusion_ready_sources.py`.

#### 12.2.4 Features deliberately excluded from the router

| Not used | Reason |
|---|---|
| `regular_entropy`, `cascade_entropy` | Available in workbooks; omitted to keep the router focused on confidence level, margin, and label structure (entropy variants are separate fusion experiments). |
| `entity_prob`, `bio_prob` (raw cascade heads) | Already summarized into $p_i^{cas}$ and `cascade_margin`; raw heads would be redundant unless type-specific calibration differed. |
| Token text, `sentence_id`, `token_idx` | Would encourage memorization / spurious correlations; router is token-local and model-agnostic. |
| Gold `true_label` | Used only to define training targets (§12.1), never as an input feature at inference. |
| Fused or third-model predictions | Router chooses between **existing** Exp01 vs Exp04 outputs only. |

### 12.3 Linear SVM (`06_svm_ready`) — Objective and Hyperparameters

Experiment `06_svm_ready` uses a **linear** support vector classifier (`sklearn.svm.LinearSVC`), not a kernel SVM. In simplified binary form, it learns:

$$
f(\phi_i) = w^T\phi_i + b.
$$

The predicted source is:

$$
\hat{z}_i =
\begin{cases}
regular, & f(\phi_i) \geq 0,\\
cascade, & f(\phi_i) < 0.
\end{cases}
$$

The SVM minimizes hinge loss with regularization:

$$
\min_{w,b}
\frac{1}{2}\|w\|^2
+
C\sum_i \max(0, 1 - y_i^{svm}(w^T\phi_i+b)).
$$

Ready fusion uses `sklearn.svm.LinearSVC` inside a `Pipeline`:

| Parameter / component | Value |
|---|---|
| Classifier | `LinearSVC` |
| Regularization `C` | `1.0` |
| `class_weight` | `"balanced"` |
| `random_state` | `42` |
| `max_iter` | `5000` |
| Numeric preprocessing | `StandardScaler()` on §12.2 numeric columns |
| Categorical preprocessing | `OneHotEncoder(handle_unknown="ignore")` |
| Training rows | Disagreement tokens where **exactly one** of regular / cascade matches gold (§12.1) |
| Minimum data | Both classes (`regular` and `cascade` targets) must appear; else fallback |
| Fallback | Same rule as §11.3 (higher scalar confidence wins) |

Class weights are balanced so that the router does not simply prefer the majority source when one side is correct more often on disagreements.

### 12.4 Final SVM Fusion Rule

For agreement tokens:

$$
\hat{y}_i^{fused} = \hat{y}_i^{reg} = \hat{y}_i^{cas}.
$$

For disagreement tokens:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & \hat{z}_i = regular,\\
\hat{y}_i^{cas}, & \hat{z}_i = cascade.
\end{cases}
$$

If the router cannot be trained, for example because there are not enough usable disagreement examples, the method falls back to the confidence fusion rule.

Important limitation (**appendix** ready ML routers only — `06_*_ready`):

> The ready variants train and evaluate the router on the same ready output set. Treat their F1 as an **in-sample routing upper bound** for appendix figures only. **Primary** thesis fusion results use **`06_*_oof`** (§12C).

### 12C. Primary protocol — Nested stratified CV + OOF (`06_*_oof`)

**Reference:** [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md)  
**Code:** `experiments/fusion_router_oof_common.py`, `multilabel_stratified_kfold_assignments` in `exp07_split_artifacts.py`.

For each Exp07 **condition** and training **seed**, the OOF runner merges that condition’s train + eval sentence JSON (150 or ~300 sentences depending on profile), then:

1. Assigns sentences to **5** outer folds (default) with multilabel stratification.
2. On outer-train (~80%): runs **4**-fold inner OOF Exp01 + Exp04 training to build clean disagreement features for router training.
3. Fits the sklearn router (same §12.1–§12.2 features/targets as ready).
4. Retrains Exp01 + Exp04 on full outer-train; evaluates fused predictions on outer-test (~20%).
5. Pools outer-test predictions → **one** entity-level F1 over all sentences (no router leakage).

Environment: `THESIS_ROUTER_OOF_OUTER_FOLDS` (default `5`), `THESIS_ROUTER_OOF_INNER_FOLDS` (default `4`), cache `{--output-dir}/oof_router_cache/`.

| Runner ID | Classifier |
|---|---|
| `06_svm_oof` | `LinearSVC` |
| `06_svm_kernel_oof` | RBF `SVC` |
| `06_nb_oof` | `GaussianNB` |
| `06_lr_oof` | `LogisticRegression` |
| `06_rf_oof` | `RandomForestClassifier` |
| `06_mlp_oof` | `MLPClassifier` |

Legacy single-split train/eval router training (no nested CV) remains in `experiment_06_fusion_svm.py` for ad-hoc runs.

### 12.5 Kernel SVM (`06_svm_kernel_ready`, appendix)

Uses `sklearn.svm.SVC` with **`kernel="rbf"`**, `gamma="scale"`, `C=1.0`, and `class_weight="balanced"` (`ROUTER_SVC_RBF_PARAMS`). Features, targets, agreement passthrough, and §11.3 fallback match §12.1–§12.4.

**Linear vs kernel:** `LinearSVC` learns a single hyperplane in the preprocessed feature space (fast, interpretable weights). RBF `SVC` can represent **non-linear** boundaries between “pick regular” and “pick cascade,” at the cost of higher compute and less transparent decision rules. Both are trained on the **same** disagreement subset.

### 12B. Other Classical ML Routers (Ready)

All variants below reuse §12.1 targets, §12.2 features, sklearn `Pipeline` preprocessing (`StandardScaler` + `OneHotEncoder`), and the fusion rule in §12.4. Hyperparameters live in `fusion_ready_sources.py`.

#### 12B.1 Naive Bayes (`06_nb_ready`) — `GaussianNB`

**Method:** `sklearn.naive_bayes.GaussianNB` with default parameters on the same scaled one-hot feature matrix as the SVM routers.

**Reservation (important for the thesis):** Naive Bayes assumes **conditional independence** of features given the class. The router inputs violate this assumption in two ways:

1. **Correlated numerics:** `regular_prob`, `cascade_prob`, `prob_diff`, `abs_prob_diff`, and `max_prob` are deterministically related; margins are also correlated with probabilities.
2. **One-hot categoricals:** BIO/type indicators are sparse and co-occur with numeric confidence patterns in structured ways.

Gaussian NB can still be run as a **baseline** for completeness, but it is **not theoretically well matched** to this feature design. Treat its F1 as empirical only; prefer logistic regression, linear/kernel SVM, random forest, or MLP when interpreting which meta-learner fits the routing task.

#### 12B.2 Logistic Regression (`06_lr_ready`)

`LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)`. A linear probabilistic classifier on the same $\phi_i$ as §12.3; often a strong, calibrated baseline for binary routing.

#### 12B.3 Random Forest (`06_rf_ready`)

`RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)`. Non-linear ensemble; handles feature interactions (e.g. confidence gap × BIO conflict) without manual crosses; less interpretable than linear models.

#### 12B.4 MLP (`06_mlp_ready`)

`MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42)`. Small feed-forward network on the preprocessed features; can capture non-linear routing rules but may overfit when usable disagreement tokens are scarce (same limitation as all ready routers).

#### 12B.5 Code map

| Runner ID | Module |
|---|---|
| `06_svm_ready` | `experiment_06_fusion_svm_ready.py` |
| `06_svm_kernel_ready` | `experiment_06_fusion_svm_kernel_ready.py` |
| `06_nb_ready` | `experiment_06_fusion_nb_ready.py` |
| `06_lr_ready` | `experiment_06_fusion_lr_ready.py` |
| `06_rf_ready` | `experiment_06_fusion_rf_ready.py` |
| `06_mlp_ready` | `experiment_06_fusion_mlp_ready.py` |

Shared logic: `fusion_router_ready_common.py`.

---

## 12A. Experiment 10 — BERT-CRF Extension (Optional Cross-Comparison Branch)

Experiment 10 implements **future-work items** from the thesis plan: replace (or augment) independent softmax tagging with a **Conditional Random Field** so **BIO transition structure** is learned during training, not only repaired afterward (compare Exp05_ready).

It is **additive**: Experiments 01, 04, and 05/06 ready paths are unchanged. Exp10 uses its **own output folders** (`outputs/exp10_regular/`, `outputs/exp10_cascade/`, …) and **own base cache index**.

**Teaching documentation:** `experiments/experiment_10_README.md`  
**Code map:** `core/crf_layer.py`, `core/bert_crf_training.py`, `core/cascaded_crf_runtime.py`

Example command:

```bash
python run_cross_data_model_comparison.py \
  --experiments 10_regular,10_cascade,10_fusion_ready,10_svm_ready \
  --models dictabert,berel \
  --base-mode auto
```

### 12A.1 Experiment 10_regular — BERT + Emissions + CRF

Architecture:

1. Hebrew transformer encoder (same registry as Exp01).
2. Linear layer producing emission scores $e_t(k)$ for each tag $k$ at token $t$.
3. **Linear-chain CRF** with transition matrix $A$ of size $(K+2)\times(K+2)$ (START/STOP states).

Training minimizes **CRF negative log-likelihood** (forward algorithm for $\log Z(x)$). Inference uses **Viterbi** to obtain $\hat{\mathbf{y}}$.

**Class imbalance:** bias of the `O` tag in the emission layer is initialized to **6** (Souza et al. 2019).

Mathematically, for gold sequence $\mathbf{y}^*$:

$$
\mathcal{L}_{CRF} = \log Z(x) - s(\mathbf{y}^*),
\quad
s(\mathbf{y}) = \sum_t \left( A_{y_{t-1},y_t} + e_t(y_t) \right).
$$

This is the same structural idea as classical NER-CRF, but emissions come from **BERT** rather than hand-crafted features.

### 12A.2 Experiment 10_cascade — Cascaded Heads + Full-Tag CRF

Exp04 trains three heads (entity, B/I, type) and **composes** their outputs. Exp10 **retains** those heads for diagnostic step F1, and adds a **joint head** over full BIO-type labels with CRF loss.

**Pipeline span F1** for reporting uses **Viterbi** on the joint head (word-level aggregated emissions). Optional post-decode rule:

$$
\text{if } \hat{b}_i=B-X \text{ and } \hat{b}_{i+1}=I-Y \text{ with } X\neq Y,\ \text{reconcile types (Exp05 idea)}.
$$

Controlled by `THESIS_STEP3_BI_TYPE_RECONCILE=1` in the cascaded wrapper script.

### 12A.3 Experiment 10_fusion_ready — Confidence Fusion on CRF Outputs

Identical arbitration to §11, but inputs are:

- regular CRF token predictions (`token_predictions` sheet), and
- cascaded CRF token predictions (`detailed_results`, `eval_mode=predicted`).

Confidence definitions follow §11.2.1 (emission softmax on the Viterbi tag for regular) and §11.2.2 (entity × bio product for cascade). Loader entry point: `experiments/fusion_crf_ready_sources.py` → `fusion_ready_sources.py`.

For token $i$, if $\hat{y}_i^{reg} \neq \hat{y}_i^{cas}$:

$$
\hat{y}_i^{fused} =
\begin{cases}
\hat{y}_i^{reg}, & p_i^{reg} \geq p_i^{cas},\\
\hat{y}_i^{cas}, & p_i^{reg} < p_i^{cas}.
\end{cases}
$$

### 12A.4 Experiment 10 ML routers on CRF disagreements

Same **router feature schema** (§12.2.1–§12.2.3) and **classifier choices** as §12.3–§12B (`10_svm_ready`, `10_svm_kernel_ready`, `10_nb_ready`, `10_lr_ready`, `10_rf_ready`, `10_mlp_ready`), applied to **CRF** disagreement tokens. Only $p_i^{reg}$ definition differs (§11.2.1 emission on Viterbi tag); all seven numeric and four categorical columns are built by the same merge loader. Router targets:

$$
z_i \in \{regular, cascade\}
$$

only when exactly one of $\hat{y}_i^{reg}, \hat{y}_i^{cas}$ equals $y_i$.

### 12A.5 Caching, Cleanup, and Error Analysis

**Base cache:** `_ensure_base_artifacts_crf` trains or reuses `10_regular` + `10_cascade` per $(m,c)$ and records paths in `cross_comparison_base_crf_ready_index.json`.

**Disk cleanup:** after training, checkpoint directories may be deleted (`THESIS_DELETE_MODELS_AFTER_TRAIN=1`) while **Excel/JSON metrics are retained**.

**Consolidated error analysis:** when any Exp10 ID is selected, the runner merges error-analysis workbooks into `outputs/cross_comparison/consolidated_error_analysis_exp10_<timestamp>.xlsx`.

---

## 13. Evaluation Metric: Entity-Level Precision, Recall, and F1

The primary metric is strict entity-level F1 using seqeval-style evaluation.

A predicted entity is correct only if both boundary and type match.

A true entity span is:

$$
(s,e,t),
$$

where:

- $s$ is start token index,
- $e$ is exclusive end token index,
- $t$ is entity type.

Let:

- $G$ be the set of gold spans,
- $P$ be the set of predicted spans.

Then:

$$
TP = |P \cap G|,
$$

$$
FP = |P \setminus G|,
$$

$$
FN = |G \setminus P|.
$$

Precision:

$$
Precision = \frac{TP}{TP+FP}.
$$

Recall:

$$
Recall = \frac{TP}{TP+FN}.
$$

F1:

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}.
$$

Beginner explanation:

- precision asks: “Of the entities the model predicted, how many were correct?”
- recall asks: “Of the true entities in the data, how many did the model find?”
- F1 balances both.

---

## 14. Paired Seed Design

The same seed list is used across all models, methods, and compatible conditions:

$$
S = \{42,43,\ldots,61\}.
$$

This creates paired observations. For example, the F1 of split-only seed 42 can be compared directly with split+augmentation seed 42.

For a pair of conditions $A$ and $B$, define:

$$
x_s = F1(A,s),
$$

$$
y_s = F1(B,s),
$$

$$
d_s = y_s - x_s.
$$

The mean difference is:

$$
\bar{d} = \frac{1}{n}\sum_{s=1}^{n}d_s.
$$

The sample standard deviation is:

$$
s_d = \sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(d_s-\bar{d})^2}.
$$

The paired t-statistic is:

$$
t = \frac{\bar{d}}{s_d/\sqrt{n}}.
$$

The null hypothesis is:

$$
H_0: \mu_d = 0.
$$

The alternative hypothesis is:

$$
H_1: \mu_d \neq 0.
$$

The runner also computes a Wilcoxon signed-rank test, which is less dependent on normality assumptions.

For Wilcoxon:

1. compute differences $d_s$,
2. remove zero differences,
3. rank $|d_s|$,
4. attach signs,
5. test whether positive and negative signed ranks are balanced.

The significance threshold is:

$$
\alpha = 0.05.
$$

A result with:

$$
p < 0.05
$$

is marked as statistically significant.

---

## 15. Aggregation Across Runs

For each group:

$$
(m,e,c),
$$

the runner aggregates over seeds.

Mean F1:

$$
\overline{F1}_{m,e,c}
=
\frac{1}{|S|}\sum_{s \in S}F1_{m,e,c,s}.
$$

Standard deviation:

$$
SD(F1)_{m,e,c}
=
\sqrt{\frac{1}{|S|-1}\sum_{s\in S}(F1_{m,e,c,s}-\overline{F1}_{m,e,c})^2}.
$$

The same aggregation is also computed for precision and recall.

The runner computes deltas such as:

Exp07 split variant improvement:

$$
\Delta_{exp07}
=
\overline{F1}_{variant}-\overline{F1}_{baseline}.
$$

Exp07+Aug improvement over the matching split-only condition:

$$
\Delta_{aug}
=
\overline{F1}_{exp07+aug}-\overline{F1}_{exp07}.
$$

Positive delta means improvement.

Negative delta means degradation.

---

## 16. Data Flow of the Complete Command

The full command follows this order.

### Step 1: Parse Configuration

The runner reads:

- selected experiments,
- selected models,
- selected condition sources,
- seed count,
- resume mode,
- base artifact mode.

It validates that the experiment IDs and model keys exist.

### Step 2: Prepare Exp07 Splits

The runner checks:

```text
outputs/exp07/splits/split_meta.json
```

If the saved split metadata and files exist, it reuses them. If they are missing and `--exp07-source auto` is active, it regenerates Exp07 split artifacts.

### Step 3: Prepare Exp08 Machinery

Even though the selected condition sources are `exp07` and `exp07+aug`, the runner prepares Exp08 split/augmentation machinery because Exp07+Aug uses Exp08-style mask-fill augmentation.

### Step 4: Prepare Exp07+Aug Splits

For each Exp07 variant and each seed, the runner creates or reuses:

$$
D_{train}^{aug(c,s)},
\qquad
D_{eval}^{c,s}.
$$

The augmented files are saved under:

```text
outputs/exp07_augmented/splits/
```

### Step 5: Build Base Conditions

The runner creates a list of base condition dictionaries. Each condition stores:

- source,
- condition key,
- variant name,
- human-readable label,
- train split path,
- eval split path,
- baseline marker,
- optional seed-specific file map.

### Step 6: Expand Conditions by Seed

For every base condition $c$ and seed $s$:

$$
c_s = expand(c,s).
$$

The expanded condition receives a key like:

```text
exp07_after_label_aware_split__seed42
```

or:

```text
exp07aug_after_label_aware_split__seed42
```

### Step 7: Loop Over Models, Experiments, and Conditions

For every model $m$, experiment $e$, and seeded condition $c_s$, the runner executes one run.

The nested conceptual loop is:

$$
\text{for } m \in M:
\quad
\text{for } e \in E:
\quad
\text{for } c_s \in C_S:
\quad
run(m,e,c_s).
$$

### Step 8: Ensure Base Artifacts

For `01`, `04`, and all ready experiments **on the 01/04 track**, the runner ensures that matching Exp01 and Exp04 artifacts exist.

For Experiment **10**, Step 8 applies the same idea via `_ensure_base_artifacts_crf` for `10_regular` + `10_cascade` (separate index file).

If missing:

1. set train/eval split environment variables,
2. run Exp01,
3. run Exp04,
4. store artifact paths.

If already available:

1. reuse the existing files,
2. avoid retraining.

### Step 9: Run Ready Experiments

For `05_ready`, `06_ready`, and `06_svm_ready`, the runner injects:

```text
THESIS_READY_EXP01_XLSX
THESIS_READY_EXP04_XLSX
```

For `10_fusion_ready` and `10_svm_ready`, the runner injects:

```text
THESIS_READY_EXP10_REGULAR_XLSX
THESIS_READY_EXP10_CASCADE_XLSX
```

Then the ready experiment reads exactly those matched files.

### Step 10: Save Checkpoint After Every Run

After each run, the runner writes progress to:

```text
outputs/cross_comparison/cross_comparison_progress_latest.json
```

This supports `--resume`.

### Step 11: Aggregate Results and Export

At the end, the runner writes Excel and JSON outputs under:

```text
outputs/cross_comparison/
```

---

## 17. Output Files and Their Scientific Meaning

The main output workbook is:

```text
outputs/cross_comparison/cross_comparison_<timestamp>.xlsx
```

A latest copy is also written:

```text
outputs/cross_comparison/cross_comparison_latest.xlsx
```

The workbook contains sheets with different scientific roles.

| Sheet | Meaning |
|---|---|
| `summary_pivot` | One row per model and experiment, with mean F1 for each condition. |
| `all_runs` | Raw result table with every model/experiment/condition/seed row. |
| `deltas_exp07` | Difference between Exp07 split variants and Exp07 baseline. |
| `deltas_exp08` | Difference between Exp08 augmented and baseline if Exp08 conditions are included. |
| `deltas_exp07_aug` | Difference between Exp07+Aug and the matching Exp07 split-only condition. |
| `paired_tests` | Paired t-test and Wilcoxon test results across seeds. |
| `model_comparison` | Head-to-head model comparison if exactly two models are selected. |
| `variant_summary` | Aggregate statistics per condition across models and experiments. |
| `experiment_details` | File paths, status, timing, and detailed metadata. |
| `documentation` | Built-in explanation of workbook interpretation. |

The JSON output contains the same information in machine-readable format:

```text
outputs/cross_comparison/cross_comparison_<timestamp>.json
```

A latest copy is also written:

```text
outputs/cross_comparison/cross_comparison_latest.json
```

---

## 18. Methodological Constraints and Assumptions

### 18.1 Same Evaluation Set Within a Condition

For a given condition and seed, all experiments and models should use the same evaluation split:

$$
D_{eval}^{(c,s)}.
$$

This makes model and method comparisons fair.

### 18.2 No Evaluation Augmentation

For `exp07+aug`, only training data changes:

$$
D_{train} \rightarrow D_{train} \cup G.
$$

The evaluation set remains unchanged:

$$
D_{eval}^{aug} = D_{eval}.
$$

### 18.3 Ready Experiments Depend on Base Outputs

`05_ready`, `06_ready`, and `06_svm_ready` are not independent training methods. They are downstream methods applied to Exp01 and/or Exp04 outputs.

Experiment **10** ready fusion methods depend on **`10_regular` + `10_cascade`** outputs (not on Exp01/Exp04).

The dependency graph is:

```text
Exp01 ─┐
       ├── Exp06_ready
Exp04 ─┘

Exp04 ─── Exp05_ready

Exp01 ─┐
       ├── Exp06_svm_ready
Exp04 ─┘

Exp10_regular ─┐
               ├── Exp10_fusion_ready
Exp10_cascade ─┤
               └── Exp10_svm_ready
```

### 18.4 Router evaluation protocols

| Protocol | Experiment IDs | Interpretation |
|---|---|---|
| **Primary** | `06_*_oof` | Nested stratified CV + OOF; report in main tables / abstract. |
| **Appendix** | `06_*_ready`, `10_*_ready` routers | Router trained and scored on the same eval tokens → **in-sample routing upper bound** only. |

Do not compare ready-router F1 directly to Exp01 held-out test F1 without a **Protocol** column.

### 18.5 F1 Is Strict Entity-Level F1

A token-level partially correct prediction is not enough. The entity boundary and entity type must both match.

For example, if the true entity is:

```text
B-PER I-PER
```

but the prediction is:

```text
B-PER O
```

then the full entity span is not correct.

---

## 19. Scientific Architecture in One Formal Diagram

The command implements the following mathematical mapping:

$$
(D, M, C, S, E)
\longrightarrow
\{F1_{m,e,c,s}, Precision_{m,e,c,s}, Recall_{m,e,c,s}\}.
$$

Where:

$$
D = \text{Hebrew BIO-labeled NER corpus},
$$

$$
M = \text{selected transformer models (Hebrew + multilingual registry keys)},
$$

$$
C = \text{split and augmentation conditions},
$$

$$
S = \text{paired seed set},
$$

$$
E = \text{selected NER architectures}.
$$

The result tensor is:

$$
R \in \mathbb{R}^{|M| \times |E| \times |C| \times |S| \times 3},
$$

where the final dimension stores:

$$
(F1, Precision, Recall).
$$

The analysis layer transforms $R$ into:

1. means,
2. standard deviations,
3. deltas,
4. paired statistical tests,
5. best-condition summaries,
6. model comparison tables.

---

## 20. Final Interpretation Guide

A reader should interpret the results as follows.

### If Exp01 performs best

The direct transformer token classifier is sufficient, and decomposition/fusion may not add value for that condition.

### If Exp04 performs best

The cascaded decomposition helps by separating the NER problem into easier subtasks:

$$
NER \approx EntityDetection + BIOPosition + EntityType.
$$

### If Exp05_ready improves over Exp04

Most improvement comes from repairing structurally inconsistent BIO/entity-type transitions.

### If Exp06_ready improves over both Exp01 and Exp04

Confidence-based fusion successfully exploits complementary strengths of the direct and cascaded systems.

### If Exp06_svm_ready improves over Exp06_ready

The disagreement pattern contains learnable information beyond raw confidence. The SVM router can identify when the regular model or cascaded model is more reliable.

### If Exp07+Aug improves over Exp07

LLM-generated training examples help the model generalize better, especially for rare or underrepresented entity labels.

### If Exp07+Aug does not improve

Possible explanations include:

1. generated examples are too noisy,
2. synthetic distribution differs from true evaluation distribution,
3. baseline training already has enough examples,
4. augmentation helps recall but hurts precision,
5. the model overfits generated patterns.

### If paired tests are significant

A significant paired test means the difference is consistent across seeds, not only caused by one lucky split.

The strongest evidence appears when both tests agree:

$$
p_{t-test} < 0.05
\quad\text{and}\quad
p_{Wilcoxon} < 0.05.
$$

---

## 21. Short Plain-English Summary

This pipeline is a rigorous experimental system for Hebrew NER. It tests **six** transformer encoders (four Hebrew-focused and two multilingual baselines) across multiple train/evaluation split strategies and augmented data variants. It compares a direct NER model, a cascaded three-step model, a structural consistency repair method, a confidence-based fusion method, and an SVM-based fusion method. Every comparison is repeated across 20 paired seeds so that improvements can

# Part II — Fair training hyperparameters

This document defines **one training profile** for comparing encoders in `run_cross_data_model_comparison.py` (DictaBERT, BEREL, HeRo, AlephBERT-Gimmel, XLM-RoBERTa, mT5, etc.). Every model in a study should see the **same** Exp01 and Exp04 hyperparameters.

The reference profile matches the **completed Colab run** for DictaBERT and BEREL on the 150-sentence pilot (`outputs/cross_comparison_150sent_dictabert_berel_noaug`, subset seed 42, 20 exp07 seeds, no augmentation).

---

## Quick reference (all models)

| Area | Setting | Value | Env override |
|------|---------|-------|----------------|
| **Exp01** | Epochs | **3** | `THESIS_NUM_EPOCHS` |
| **Exp01** | Learning rate | **5×10⁻⁵** | `THESIS_LEARNING_RATE` |
| **Exp01** | FP16 (Colab) | **on** | `THESIS_TRAINER_FP16=1` |
| **Exp01** | Weight decay | **0** | `THESIS_TRAINER_WEIGHT_DECAY` |
| **Exp01** | Class weights | **off** | `THESIS_BALANCED_CLASS_WEIGHTS=1` to enable |
| **Exp01** | Eval during train | **off** | `THESIS_TRAINER_BEST_F1_CHECKPOINT=1` for epoch eval + best F1 (slow) |
| **Exp01** | Batch / warmup | HF defaults (8 / 0) | — |
| **Exp04** | Epochs | **10** | `THESIS_EXP04_EPOCHS` |
| **Exp04** | Encoder LR | **2×10⁻⁵** | (fixed in `TRAINING_CONFIG`) |
| **Exp04** | Head LR | **1×10⁻³** | (fixed) |
| **Exp04** | Train batch | **16** | `THESIS_EXP04_TRAIN_BATCH` |
| **Exp04** | Grad accum | **2** | `THESIS_EXP04_GRAD_ACCUM` |
| **Exp04** | Loss weight $\lambda_{bio}$ | **10** (default) | `THESIS_EXP04_LAMBDA_BIO` |
| **Exp04** | Loss weight $\lambda_{type}$ | **5** (default) | `THESIS_EXP04_LAMBDA_TYPE` |

Exp05 / Exp06 “ready” and fusion steps **reuse** Exp01/Exp04 artifacts; they do not change base training hyperparameters.

**Calibrating $\lambda_{bio}$, $\lambda_{type}$:** Run `experiments/experiment_04_loss_weight_grid.py` once per split policy (default grid `{1,5,10}^2` on validation). Set `THESIS_EXP04_LAMBDA_BIO` and `THESIS_EXP04_LAMBDA_TYPE` from the `selection` sheet in `outputs/exp04_lambda_grid/` **before** cross-comparison. Exported automatically to `cross_comparison_*.xlsx` sheets **`journal_lambda_grid`** / **`journal_loss_config`** and consolidated **`loss_config_summary`**. Full paper workflow: [`thesis_overview_journal_paper_results.md`](thesis_overview_journal_paper_results.md).

---

## What the runner does automatically

When you start `run_cross_data_model_comparison.py`, it calls `core.training_defaults.apply_fair_comparison_training_defaults()`:

- Sets the env vars above with `setdefault` (your explicit exports **win**).
- Writes a **`training_hyperparameters`** block into `{output-dir}/run_manifest.json`.

Implementation: `core/training_defaults.py`, `core/th_functions.py` (Exp01), `core/auc_cascaded_pipeline.py` (Exp04).

---

## Colab: continue multilingual after DictaBERT / BEREL

Use the **same** output folder and `--resume`. Do **not** set different epochs or LR for XLM-R / mT5 only.

```python
%cd /content/drive/MyDrive/thesis_project/thesis/thesis  # your clone

!git pull origin main

import os
os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"
os.environ["THESIS_CSV_ENCODING"] = "utf-8"
os.environ["THESIS_FULL_NER_CSV"] = "/content/drive/MyDrive/thesis_project/thesis/data/ner_dataset.csv"

# Optional: force the reference profile (runner also applies defaults if unset)
os.environ.setdefault("THESIS_NUM_EPOCHS", "3")
os.environ.setdefault("THESIS_LEARNING_RATE", "5e-5")
os.environ.setdefault("THESIS_TRAINER_FP16", "1")
os.environ.setdefault("THESIS_TRAINER_WEIGHT_DECAY", "0")
os.environ.setdefault("THESIS_EXP04_EPOCHS", "10")

OUTPUT = "outputs/cross_comparison_150sent_dictabert_berel_noaug"

!python run_cross_data_model_comparison.py \
  --output-dir {OUTPUT} \
  --subset-sentences 150 \
  --subset-seed 42 \
  --models dictabert,berel,xlm_roberta,mt5 \
  --experiments 01,04,05_ready,06_ready,06_svm_ready \
  --skip-augmentation \
  --condition-sources exp07 \
  --exp07-source auto \
  --base-mode auto \
  --num-seeds 20 \
  --consolidated-error-analysis all \
  --resume
```

### If XLM-R / mT5 were trained with the wrong profile

1. Pull latest code (unified Exp01 path for all encoders).
2. Edit `{OUTPUT}/cross_comparison_progress_latest.json`: remove rows for `FacebookAI/xlm-roberta-base` and `google/mt5-base` (or failed runs).
3. Resume with the env block above so new runs match DictaBERT / BEREL.

Do **not** compare metrics from checkpoint rows trained with different env (e.g. 10 epochs + 5e-5 + class weights only on multilingual).

---

## Full 150-sent pilot (four models from scratch)

Same env as above; use `--exp07-source rerun` on first launch, omit `--resume` until you need it:

```bash
python run_cross_data_model_comparison.py \
  --output-dir outputs/cross_comparison_150sent_dictabert_berel_noaug \
  --subset-sentences 150 \
  --subset-seed 42 \
  --models dictabert,berel,xlm_roberta,mt5 \
  --experiments 01,04,05_ready,06_ready,06_svm_ready \
  --skip-augmentation \
  --condition-sources exp07 \
  --exp07-source rerun \
  --base-mode auto \
  --num-seeds 20 \
  --consolidated-error-analysis all
```

---

## Experiment-specific notes

### Exp01 — regular token classification (`experiment_01_regular_ner.py`)

- **Colab disk-minimal** (`THESIS_RUN_ENV=colab`, default `THESIS_DELETE_MODELS_AFTER_TRAIN=1`): checkpoints under `/tmp`, no mid-training eval unless `THESIS_TRAINER_BEST_F1_CHECKPOINT=1`.
- **Tokenization**: RoBERTa-style models (XLM-R) use `add_prefix_space=True` via `encode_words_for_ner()` in `core/model_backbone.py`. mT5 uses encoder + token head with `ignore_index=-100` for subword labels.
- **Models**: registry keys in `run_cross_data_model_comparison.py` → Hugging Face ids (`dicta-il/dictabert`, `dicta-il/BEREL_3.0`, `FacebookAI/xlm-roberta-base`, `google/mt5-base`, …).

### Exp04 — cascaded entity pipeline (`core/auc_cascaded_pipeline.py`)

- Uses `TRAINING_CONFIG` (not the HF Trainer). Epoch count comes from `THESIS_EXP04_EPOCHS` (default **10** after fair defaults).
- **Multi-task loss:** $\mathcal{L}_{entity} + \lambda_{bio}\mathcal{L}_{bio} + \lambda_{type}\mathcal{L}_{type}$ with masked B/I and type terms; defaults $\lambda_{bio}=10$, $\lambda_{type}=5$ unless env overrides are set after the validation grid.
- **Thresholds:** Post-training grid on validation for entity/BIO sigmoid thresholds (separate from $\lambda$ grid); see `theisis overview.md` §9.6.
- **Important:** Exp01 uses **3** epochs; Exp04 uses **10**. That is intentional (different experiment designs). Fair **cross-model** comparison means each experiment keeps the same settings for every encoder—not that Exp01 and Exp04 must share one epoch count.

### Ready experiments (05, 06_*)

- Consume cached metrics and probabilities from Exp01/Exp04; no separate encoder fine-tuning hyperparameter table.

---

## Overriding for a new study

Set env **before** launching the runner. Examples:

| Goal | Example |
|------|---------|
| Longer Exp01 for all models | `export THESIS_NUM_EPOCHS=5` |
| Standard BERT NER LR | `export THESIS_LEARNING_RATE=2e-5` |
| Imbalanced tags (all models) | `export THESIS_BALANCED_CLASS_WEIGHTS=1` |
| Shorter Exp04 | `export THESIS_EXP04_EPOCHS=5` |
| Custom loss weights (after grid) | `export THESIS_EXP04_LAMBDA_BIO=10` / `THESIS_EXP04_LAMBDA_TYPE=5` |
| Coarse validation grid only | `python experiments/experiment_04_loss_weight_grid.py` (+ optional `THESIS_EXP04_FAST=1`) |
| Disable fp16 | `export THESIS_TRAINER_FP16=0` |

If you change the profile mid-study, clear or retrain affected checkpoint rows and update the thesis methods section to match `run_manifest.json` → `training_hyperparameters`.

---

## Related files

| File | Role |
|------|------|
| `core/training_defaults.py` | Profile constants, `apply_fair_comparison_training_defaults()`, manifest snapshot |
| `core/th_functions.py` | Exp01 `TrainingArguments` |
| `core/auc_cascaded_pipeline.py` | Exp04 `TRAINING_CONFIG` + env |
| `experiments/experiment_04_loss_weight_grid.py` | Validation grid for $\lambda_{bio}$, $\lambda_{type}$ |
| `run_cross_data_model_comparison.py` | Applies defaults at run start, writes manifest |
| `theisis overview.md` §1.1, §9.4 | Pipeline commands; cascaded loss-weight protocol |

---

## Checklist before writing up results

- [ ] `run_manifest.json` contains `training_hyperparameters` for this output dir.
- [ ] All models in the comparison table were trained with the same manifest profile (or you document explicit overrides).
- [ ] No multilingual-only epoch / LR / class-weight shortcuts in the code revision used for the run (`git rev-parse HEAD` on Colab).
- [ ] Subset and seeds documented: `--subset-sentences`, `--subset-seed`, `--num-seeds`, `--condition-sources`.
- [ ] Exp04 $\lambda_{bio}$, $\lambda_{type}$ documented (validation grid output or explicit env overrides matching all Exp04 runs in the table).

# Part III — OOF router fusion

This document defines the **primary** learned-router evaluation used in the thesis for both corpus sizes:

| Profile | Sentences | Typical outer-test size (5 folds) | Output folder example |
|---------|-----------|-------------------------------------|------------------------|
| **Full labeled set** | ~300 | ~60 sentences / fold | `outputs/cross_comparison_full` |
| **150-sentence pilot** | 150 | ~30 sentences / fold | `outputs/cross_comparison_150sent` |

The **appendix / exploratory** protocol (`06_*_ready`) fits and scores the router on the **same** Exp07 eval tokens (in-sample routing upper bound). Do **not** compare ready-router F1 to Exp01 test F1 in main claims.

---

## 1. Experiment IDs (primary vs appendix)

| Role | Runner IDs | Module |
|------|------------|--------|
| **Primary (report in abstract / main tables)** | `06_svm_oof`, `06_svm_kernel_oof`, `06_nb_oof`, `06_lr_oof`, `06_rf_oof`, `06_mlp_oof` | `experiment_06_fusion_*_oof.py` → `fusion_router_oof_common.py` |
| **Appendix only** | `06_svm_ready`, … `06_mlp_ready` | `fusion_router_ready_common.py` |

Default cross-comparison experiments (if `--experiments` omitted): `01,04,05_ready,06_ready,06_svm_oof`.

---

## 2. Nested CV diagram

Corpus = all sentences in the current run (Exp07 **train + eval** JSON merged for that condition). Splits are **sentence-level** (no token leakage).

```text
[All sentences: N = 150 or ~300]
 ├── Outer fold 0 (test ≈ N/5) ──► outer-train ──► inner (K−1)-fold OOF ──► train router ──► eval outer-test
 ├── Outer fold 1 …
 └── Outer fold 4 …
```

**Per outer fold:**

1. **Outer test** (~20% of sentences): never used for NER training, inner OOF, or router fit in that fold.
2. **Inner loop** on outer-train (default **4** folds): train Exp01 + Exp04 on inner-train, predict inner-val → **OOF** predictions for every outer-train sentence.
3. **Router fit:** same targets and features as §12 in [`theisis overview.md`](theisis%20overview.md), on **OOF** disagreement rows only.
4. **Final NER:** retrain Exp01 + Exp04 on **all** outer-train sentences; predict outer-test; apply **frozen** router on disagreements; agreement passthrough unchanged.
5. **Pool** outer-test token predictions across all outer folds → one entity-level F1 over the full corpus (each sentence in exactly one outer test).

Fold assignment: **multilabel iterative stratification** (`multilabel_stratified_kfold_assignments` in `experiments/exp07_split_artifacts.py`) so rare entity types (e.g. pSEC, pSCHP) stay represented in each fold when possible.

---

## 3. Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `THESIS_ROUTER_OOF_OUTER_FOLDS` | `5` | Outer CV folds |
| `THESIS_ROUTER_OOF_INNER_FOLDS` | `4` | Inner OOF folds on outer-train |
| `THESIS_CROSS_OUTPUT_DIR` | set by `--output-dir` | OOF NER cache: `{output-dir}/oof_router_cache/` |
| `THESIS_PRESPLIT_TRAIN_JSON` / `THESIS_PRESPLIT_EVAL_JSON` | per condition | Corpus = union of both files |
| `{output-dir}/training_seeds.json` | 20 random seeds (persisted) | Stable `THESIS_SPLIT_SEED` across reruns |
| `THESIS_EXP04_LAMBDA_BIO` / `THESIS_EXP04_LAMBDA_TYPE` | **10** / **5** if unset | Same cascaded loss weights as standalone Exp04; set **before** OOF runs after validation grid (`theisis overview.md` §9.4) |

Router features, targets, classifiers, and §11.3 fallback: identical to ready routers (`fusion_ready_sources.py` constants).

Every inner/outer Exp04 retrain in OOF uses the same $\lambda_{bio}$, $\lambda_{type}$ as the main cross-comparison profile (env overrides read in `core/auc_cascaded_pipeline.py`).

---

## 4. Example commands

**150-sentence pilot (DictaBERT + BEREL, all OOF routers):**

```bash
python run_cross_data_model_comparison.py \
  --output-dir outputs/cross_comparison_150sent \
  --subset-sentences 150 --subset-seed 42 \
  --models dictabert,berel \
  --experiments 01,04,05_ready,06_ready,06_svm_oof,06_rf_oof \
  --skip-augmentation --condition-sources exp07 --exp07-source rerun \
  --base-mode auto --num-seeds 20 --resume
```

**Full corpus (~300 sentences):**

```bash
python run_cross_data_model_comparison.py \
  --output-dir outputs/cross_comparison_full \
  --models dictabert,berel,hero,alephbertgimmel,xlm_roberta,mt5 \
  --experiments 01,04,05_ready,06_ready,06_svm_oof,06_rf_oof \
  --skip-augmentation --condition-sources exp07 \
  --base-mode auto --num-seeds 20
```

**Appendix (in-sample ready routers, fast — reuse Exp01/Exp04 cache):**

```bash
python run_cross_data_model_comparison.py \
  --experiments 06_svm_ready,06_rf_ready \
  --base-mode reuse
```

---

## 5. Compute note

Each `(model, condition, seed, 06_*_oof)` run retrains Exp01 + Exp04 many times (inner folds × outer folds + outer finals). Cache under `oof_router_cache/` avoids repeating identical splits. Expect **much** longer wall time than `*_ready`.

---

## 6. Reporting checklist

- Main tables: **`06_*_oof`** F1 only for learned fusion.
- Label column: `protocol = nested_stratified_cv_oof`.
- Report per-fold metrics sheet (`fold_metrics`) when available.
- Appendix: `06_*_ready` labeled **in-sample routing upper bound**.
- **Score baselines inside the same folds.** Build `unified_protocol_table.xlsx` with
  `build_unified_protocol_table.py` so Exp01/Exp04/Exp05/confidence fusion are scored on the same
  pooled outer-test tokens as the routers. Never subtract a 70/30 holdout baseline F1 from an OOF
  fusion F1 — the fold models train on ~80% of the corpus vs. ~70% for the holdout, so the
  difference conflates protocol with method.
- Keep holdout results in a **separate** table with its own caption (seed-paired significance and
  the split-strategy comparison), never as extra columns beside OOF numbers.

# Part IV — Journal paper workflow & results

One command runs **(1) Exp04 loss-weight calibration** (resume-aware grid) and **(2) the full cross-comparison** (resume-aware checkpoint). See also [`theisis overview.md`](theisis%20overview.md) §9.4, [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md).

---

## 1. Single command (recommended)

**150-sentence pilot:**

```bash
python run_cross_data_model_comparison.py \
  --journal-paper \
  --resume \
  --output-dir outputs/cross_comparison_journal \
  --subset-sentences 150 \
  --subset-seed 42 \
  --exp07-source rerun \
  --models dictabert,berel \
  --base-mode auto
```

**Full corpus (~300 sentences):**

```bash
python run_cross_data_model_comparison.py \
  --journal-paper \
  --resume \
  --output-dir outputs/cross_comparison_full_journal \
  --exp07-source auto \
  --models dictabert,berel \
  --base-mode auto
```

### What `--journal-paper` does

| Step | Behavior | Resume cache |
|------|----------|--------------|
| **Loss weights** | Validation grid `{1,5,10}²` → sets `THESIS_EXP04_LAMBDA_BIO/TYPE` | `{output-dir}/exp04_lambda_grid_cache/exp04_lambda_selection.json` + `exp04_lambda_grid_progress.json` (per grid cell) |
| **Training seeds** | 3 random seeds persisted | `{output-dir}/training_seeds.json` |
| **Comparison** | `01,04,06_svm_oof,06_rf_oof`, exp07 only, no aug | `{output-dir}/cross_comparison_progress_latest.json` |
| **Export** | Journal sheets + consolidated error analysis (`all`) | Rebuild export skips re-training if checkpoint complete |

`--journal-paper` implies `--resume` unless you pass **`--no-resume`**.

### Manual two-step (optional)

```bash
python experiments/experiment_04_loss_weight_grid.py
set THESIS_EXP04_LAMBDA_BIO=...
set THESIS_EXP04_LAMBDA_TYPE=...
python run_cross_data_model_comparison.py --calibrate-exp04-loss-weights --resume ...
```

Prefer **`--journal-paper`** so calibration uses the same `--output-dir` cache.

### Flags

| Flag | Purpose |
|------|---------|
| `--resume` | Skip finished comparison jobs (always use with journal workflow) |
| `--no-resume` | Force re-run all pending comparison jobs |
| `--regenerate-exp04-loss-grid` | Ignore cached λ selection; re-run all 9 grid trainings |
| `--regenerate-training-seeds` | New `training_seeds.json` |
| `--skip-exp04-loss-calibration` | Skip grid; use env or defaults 10/5 |
| `--rebuild-from-checkpoint` | Excel/JSON only, no training |

---

## 2. Primary results workbook

**File:** `{output-dir}/cross_comparison_latest.xlsx`

| Sheet | Paper use |
|-------|-----------|
| **`journal_oof_fold_summary`** | Main OOF fusion F1 (5 folds); covers all 6 primary routers (linear SVM, RBF SVM, Naive Bayes, logistic regression, RF, MLP), not just SVM/RF |
| **`journal_main_table`** | Exp01/04 **and every OOF fusion method** (`JOURNAL_FOCUS_EXP_IDS` in `experiments/journal_results_export.py`), seeds mean±SD |
| **`journal_lambda_grid`** | Selected λ from calibration cache |
| **`journal_loss_config`** | λ recorded per Exp04 run |
| **`journal_paper_guide`** | Short index |

> **`journal_paired_fold_deltas` is not emitted** — see the correction in Part V §V.1. Significance
> tests come from `paired_fold_tests` in `unified_protocol_table.xlsx` instead.

### Paper-ready workbook

**File:** `{output-dir}/unified_protocol_table.xlsx` — built by `build_unified_protocol_table.py`
(Part V §V.1). **This is the single file to write from:** every method re-scored on the same
outer-test tokens, plus the `journal_*` sheets above copied in verbatim.

| Sheet | Paper use |
|-------|-----------|
| **`fold_summary`** | Main-table cells: mean ± SD F1 over all `(seed, fold)` units, every method |
| **`fusion_vs_base`** | Headline ΔF1, fusion minus each baseline on identical tokens |
| **`paired_fold_tests`** | Wilcoxon + t-test p-values, Holm-adjusted per `(model, split_condition)` |
| **`unified_main_table`** | Pooled outer-test F1/P/R, one protocol |

Do **not** mix the runner's holdout F1 with OOF F1 in one table: outer folds train on ~80% of the
corpus while the 70/30 holdout trains on ~70%, so a mixed table credits fusion with a training-data
advantage unrelated to fusion. Keep holdout results in a separate table with its own caption.

---

## 3. Consolidated error analysis

**File:** `{output-dir}/consolidated_error_analysis_all_*.xlsx`

| Sheet | Use |
|-------|-----|
| **`summary_non_crf_*`** | Error types, routing, confusions |
| **`loss_config_summary`** | λ per Exp04 metrics file |
| **`oof_fold_summary`** | Fold CV appendix |

---

## 4. Artifacts checklist

| File | Role |
|------|------|
| `exp04_lambda_grid_cache/exp04_lambda_selection.json` | Chosen λ for Methods |
| `exp04_lambda_grid_cache/loss_weight_grid_*.xlsx` | Heatmap figure |
| `training_seeds.json` | Reproducible 3 seeds |
| `cross_comparison_progress_latest.json` | Resume state |
| `run_manifest.json` | Hyperparameters + `exp04_lambda_calibration` |

---

## 5. Exp04 loss weights in code

- Overrides: `THESIS_EXP04_LAMBDA_BIO`, `THESIS_EXP04_LAMBDA_TYPE` (`core/auc_cascaded_pipeline.py`)
- Each run: sheet **`loss_config`** in `cascaded_pipeline_results.xlsx`
- Grid script: `experiments/experiment_04_loss_weight_grid.py` (also invoked via `exp04_loss_calibration.py`)

Implementation: `experiments/journal_results_export.py` builds journal sheets on export.

---

## 6. Local vs Google Colab (CPU or GPU)

`core/runtime_env.py` runs on import via `experiments/common.py`:

| Environment | Detection | Training device |
|-------------|-----------|-----------------|
| **Local** | Default (`THESIS_RUN_ENV` unset) | CUDA if available, else CPU |
| **Colab** | Auto (`google.colab`) or `THESIS_RUN_ENV=colab` | Same |
| **Colab CPU** | No CUDA | fp16 off; smaller Exp04 batches (4/4, grad accum 4) unless you override |

**Local (PowerShell / bash):** run from repo root; Intel proxy defaults apply when not on Colab. Override with `THESIS_HTTP_PROXY` / `THESIS_NO_PROXY` if needed.

```bash
python run_cross_data_model_comparison.py --journal-paper --resume --output-dir outputs/cross_comparison_journal --models dictabert,berel --base-mode auto
```

**Colab:** mount Drive, `%cd` to the clone, `pip install -r requirements.txt`. Colab is auto-detected; you may still set:

```python
import os
os.environ.setdefault("THESIS_RUN_ENV", "colab")  # optional; auto-detected
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("THESIS_CSV_ENCODING", "utf-8")
```

**GPU runtime (recommended):** default fp16 on when CUDA is available (`THESIS_TRAINER_FP16=1`).

**CPU runtime (slow but supported):** fp16 stays off; Exp04 uses smaller batches on Colab CPU. For a smoke test on CPU:

```python
os.environ["THESIS_EXP04_FAST"] = "1"   # fewer Exp04 epochs / threshold steps
os.environ["THESIS_NUM_EPOCHS"] = "1"   # Exp01 only — not for paper numbers
```

Then the same `--journal-paper` command as local (use a Drive-backed `--output-dir` and `--resume`).

# Part V — Statistical significance testing

This part applies to **both** corpus sizes (full ~300-sentence runs and the 150-sentence pilot). It defines the paired-design methodology behind every significance claim in the thesis: how many training seeds are needed, which tests to run, and where the results already live in the exported workbooks.

*Merged from `STATISTICAL_SIGNIFICANCE_GUIDE.md` (2026-10-04); that file is now a pointer to this section.* The 150-sentence pilot's concrete completed run (DictaBERT + BEREL, all 6 fusion methods, `--num-seeds 20`) is documented in [`thesis_overview_150_sentences.md`](thesis_overview_150_sentences.md) Part V, which mirrors this section with pilot-specific status notes.

## V.1 Primary path: `build_unified_protocol_table.py` (use this first)

> **Correction (2026-10-05).** Earlier revisions of this section claimed that
> `journal_paired_fold_deltas` in `cross_comparison_*.xlsx` already holds the fusion-vs-baseline
> tests. **It does not, and the sheet is not emitted at all.** `collect_oof_fold_long()` in
> `experiments/journal_results_export.py` skips every row whose `experiment_id` lacks `_oof`, so
> `exp01`, `exp04` and `exp06_ready` never enter `fold_long`. In `paired_fold_method_comparison()`
> both entries of `JOURNAL_BASELINE_EXP_IDS` are then absent from `available`, the pair-building
> loop `continue`s on both, `pairs` stays empty, and the function returns an empty frame — which the
> writer skips via `if not journal_paired_fold_df.empty`. Net effect: **no
> `journal_paired_fold_deltas` sheet exists in the workbook.**
>
> Root cause is structural, not a typo: baseline per-fold F1 was never computed, because Exp01/Exp04
> are holdout runs and only `06_*_oof` runs write a `fold_metrics` sheet. The fix is to recover the
> baselines from the OOF runs' own pooled predictions — which is what the builder below does.

Run the standalone builder instead. It re-scores Exp01, Exp04, Exp05 repair, confidence fusion and
every OOF router on the **same** pooled outer-test tokens, then runs the paired tests:

```bash
python build_unified_protocol_table.py \
  --oof-dir {output-dir} \
  --splits-dir {output-dir}/exp07/splits \
  --reference-xlsx {output-dir}/cross_comparison_latest.xlsx \
  --output {output-dir}/unified_protocol_table.xlsx
```

No retraining: each outer fold already retrained Exp01 + Exp04 and stored both predictions beside
gold in the router run's `detailed_results` sheet, so this is pure post-processing (seconds, CPU).

| Sheet in `unified_protocol_table.xlsx` | What it gives you |
|----------------------------------------|--------------------|
| `paired_fold_tests` | Paired **Wilcoxon signed-rank + t-test** p-values for every fusion method vs. Regular NER, Cascade NER, cascade+repair and confidence fusion, paired by `(training_seed, outer_fold)`, **Holm-adjusted** within each `(model, split_condition)` family. |
| `fold_summary` | Mean ± SD F1 over all `(seed, fold)` units per method — the main-table cells. |
| `fusion_vs_base` | ΔF1 of fusion minus each baseline on identical tokens. |
| `unified_main_table` | Pooled outer-test F1/P/R per method, one evaluation protocol. |
| `per_fold_f1` | Raw per-`(seed, fold)` F1 — the paired units behind the tests. |
| `paired_tests` *(carried over)* | Paired t-test + Wilcoxon across shared seeds (§14) for exp07-vs-exp07+aug and exp08 ablations. |

Because pairing is by `(training_seed, outer_fold)`, raising `--num-seeds` directly multiplies
`n_pairs` (seeds × 5) and therefore statistical power — the same mechanism documented for the
paired-seed design in §14. **Check `n_pairs` in the output.** At *n*=5 (a single seed) a two-sided
signed-rank test cannot fall below *p*=0.0625 regardless of effect size; at *n*=100 the floor is
≈4 × 10⁻¹⁸ and stops being a constraint.

**Caveat to state in Methods:** outer folds within one partition share training data, so paired CV
tests are *liberal* rather than conservative (Dietterich 1998; Bengio & Grandvalet 2004). Because
`THESIS_SPLIT_SEED` drives fold assignment (`split_seed + 1000` in `fusion_router_oof_common.py`),
each training seed yields a **different** 5-fold partition, so the seed dimension is genuine
repeated CV rather than reruns of a single partition — this is the main thing mitigating the
dependency.

**Use §V.2–§V.6 below only for:** (a) understanding *why* seed count matters, (b) ad-hoc comparisons
the builder doesn't cover (e.g. model-vs-model head-to-head on one condition — see `model_comparison`
sheet first), or (c) a >2-method omnibus test (Friedman).

## V.2 Why seed count matters (power analysis)

**Seeds are the primary source of variance** for paired significance testing (§14) — they control weight initialization, data shuffling, and training stochasticity. Exp07 split strategies are systematic, not random, conditions: useful for generalization claims but not a substitute for seed variance in a paired test.

| Term | Meaning | Effect |
|------|---------|--------|
| **Seeds** | Different random initializations | Creates paired observations for the same data condition |
| **Splits** | Different train/test partitions | Tests generalization across data conditions |
| **Runs** | Repeating the whole experiment | Same as seeds if you change the seed each time |

**Bottom line:** for significance testing, seeds = runs; each seed creates one paired observation.

With only 3 seeds (the `--journal-paper` default when `--num-seeds` is not explicit — §1.1), a paired t-test has very little power: p-values rarely reach < 0.05 unless the effect is huge.

### Sample-size requirements for α = 0.05

| Effect size | Required seeds | Power |
|-------------|-----------------|-------|
| Large (d=0.8) | 10 seeds | ~75% |
| Large (d=0.8) | 15 seeds | ~88% |
| Medium (d=0.5) | 20 seeds | ~75% |
| Medium (d=0.5) | 30 seeds | ~87% |
| Small (d=0.2) | 50+ seeds | ~50%+ |

**Practical recommendation:** 10–20 seeds for detecting meaningful differences; the full-thesis default in this document is already `--num-seeds 20` (§1).

### Run-configuration reference

| Goal | Seeds | Command |
|------|-------|---------|
| Quick sanity check | 3 | `--num-seeds 3` |
| Moderate confidence | 10 | `--num-seeds 10` |
| Publication-ready | 20 | `--num-seeds 20` (default for this profile) |
| High-confidence | 30 | `--num-seeds 30` |

## V.3 Statistical test formulas (reference)

**Paired t-test** (recommended primary test for F1 scores across matched seeds/folds; also derived with full math in §14):

```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(f1_method_a, f1_method_b)  # same seeds/folds, same order
```

**Wilcoxon signed-rank** (non-parametric alternative; less dependent on normality):

```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(f1_method_a, f1_method_b)
```

**Bootstrap confidence interval** (not computed automatically; useful supplementary evidence):

```python
import numpy as np

def bootstrap_ci(a, b, n_bootstrap=10000, ci=0.95):
    diffs = np.array(a) - np.array(b)
    boot_diffs = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True))
                  for _ in range(n_bootstrap)]
    lower = np.percentile(boot_diffs, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_diffs, (1 + ci) / 2 * 100)
    return np.mean(diffs), lower, upper
# If the returned interval excludes 0, the difference is significant at that CI level.
```

The strongest evidence appears when both the t-test and Wilcoxon agree ($p_{t} < 0.05$ and $p_{Wilcoxon} < 0.05$) — consistent with §20's interpretation guide: a result consistent across seeds/folds, not caused by one lucky split.

## V.4 Multi-method comparison (>2 methods)

`paired_fold_tests` only compares each fusion method against the baselines (pairwise). To test **all** fusion methods jointly (omnibus test before pairwise post-hoc), use a **Friedman test** with **Nemenyi post-hoc** or **Bonferroni correction**:

```python
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp  # pip install scikit-posthocs
import numpy as np

# rows = seeds (or seed x fold pairs), columns = methods
data = np.array([
    [0.72, 0.71, 0.74, 0.70, 0.73, 0.75],  # seed 1: [regular, cascade, svm, rbf_svm, nb, rf]
    # ... remaining seeds
])
stat, p = friedmanchisquare(*data.T)
if p < 0.05:
    print(sp.posthoc_nemenyi_friedman(data))
```

## V.5 Reporting template

```
DictaBERT achieved mean F1 = 0.742 ± 0.015 (SD over 20 seeds), compared to
BEREL's 0.731 ± 0.018. A paired t-test confirmed the difference was
statistically significant (t(19) = 2.84, p = 0.010 < 0.05), with DictaBERT
outperforming BEREL by an average of 1.1 F1 points.
```

For fusion-method claims, cite `paired_fold_tests` in `unified_protocol_table.xlsx` directly: report `mean_delta_f1`, `wilcoxon_p_holm` (and raw `wilcoxon_p`), `ttest_rel_p_holm`, and `n_pairs` for the method vs. baseline pair (so a reader can judge power), plus `holm_family_size` so the correction is auditable.

## V.6 Ad-hoc significance outside the journal pipeline (optional)

For comparisons `paired_fold_tests` doesn't cover out of the box — e.g. model-vs-model head-to-head on one specific condition (beyond what the `model_comparison` sheet already provides) — read directly from `cross_comparison_latest.json`:

```python
"""Compute pairwise statistical significance from cross_comparison results."""
import json
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

def load_results(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f).get("results", [])

def group_by_seed(results, model_id, experiment_id, condition_key):
    return [r["f1"] for r in results
            if r.get("model_id") == model_id and r.get("experiment_id") == experiment_id
            and r.get("condition_key") == condition_key and r.get("f1") is not None]

def pairwise_significance(f1_a, f1_b, alpha=0.05):
    if len(f1_a) != len(f1_b) or len(f1_a) < 3:
        return None
    t_stat, t_pval = ttest_rel(f1_a, f1_b)
    w_stat, w_pval = wilcoxon(f1_a, f1_b)
    return {
        "mean_diff": np.mean(f1_a) - np.mean(f1_b),
        "t_stat": t_stat, "t_pval": t_pval,
        "w_stat": w_stat, "w_pval": w_pval,
        "significant_ttest": t_pval < alpha, "significant_wilcoxon": w_pval < alpha,
        "n_pairs": len(f1_a),
    }

results = load_results(Path("outputs/cross_comparison/cross_comparison_latest.json"))
f1_dictabert = group_by_seed(results, "dicta-il/dictabert", "exp01", "exp07_baseline")
f1_berel = group_by_seed(results, "dicta-il/BEREL_3.0", "exp01", "exp07_baseline")
print(pairwise_significance(f1_dictabert, f1_berel))
```

## References

1. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*.
2. Dror, R., et al. (2018). The hitchhiker's guide to testing statistical significance in NLP. *ACL*.
3. Berg-Kirkpatrick, T., et al. (2012). An empirical investigation of statistical significance in NLP. *EMNLP*.
