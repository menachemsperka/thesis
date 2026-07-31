# Experiment 10 — BERT-CRF, Cascaded CRF, and Fusion (Teaching Guide)

This document is written for **students and instructors** who want to understand *why* Experiment 10 exists, *how* it connects to Experiments 01–06, and *where* to read the implementation in this repository.

Experiment 10 does **not** replace Experiments 01 or 04. It is an **additional research branch** that adds **Conditional Random Field (CRF)** decoding and **CRF-based fusion**, motivated by Lample et al. (2016), Souza et al. (2019), and Ben-Gigi et al. (2025).

---

## 1. Learning objectives

After working through this experiment, you should be able to explain:

1. Why per-token **softmax** NER (Exp01) can produce **invalid BIO sequences**, and how a **CRF layer** learns transition scores instead of relying only on post-hoc repair (Exp05).
2. How **emission scores** (from a linear head on BERT) and **transition scores** (CRF parameter matrix) combine in the **forward–backward** training objective and **Viterbi** inference.
3. How the **cascaded pipeline** (Exp04) is extended with a **full-tag CRF head** while keeping step-wise heads for diagnostics.
4. How **ready fusion** (Exp06) is reused for CRF outputs without retraining NER models.
5. How the **cross-comparison runner** caches Exp10 training artifacts and runs cheap fusion experiments on top.

---

## 2. Experiment IDs and files (map for navigation)

| Runner ID | Output folder | Main Python entry | Core implementation |
|-----------|---------------|-------------------|---------------------|
| `10_regular` | `outputs/exp10_regular/` | `experiments/experiment_10_regular_ner_crf.py` | `core/bert_crf_training.py`, `core/crf_layer.py` |
| `10_cascade` | `outputs/exp10_cascade/` | `experiments/experiment_10_cascaded_pipeline_crf.py` | `core/cascaded_crf_runtime.py` |
| `10_fusion_ready` | `outputs/exp10_fusion_ready/` | `experiments/experiment_10_fusion_crf_ready.py` | `experiments/fusion_crf_ready_sources.py` |
| `10_svm_ready` | `outputs/exp10_svm_ready/` | `experiments/experiment_10_fusion_svm_ready.py` | Same fusion loader + sklearn router |

**Orchestration:** `run_cross_data_model_comparison.py`  
**Separate base cache index:** `outputs/cross_comparison/cross_comparison_base_crf_ready_index.json`

---

## 3. Scientific motivation (plain language)

### 3.1 Softmax vs CRF on BIO tags

In Exp01, each token is classified **independently** (conditioned on context through BERT, but not on the *previous predicted tag*). That can yield illegal patterns, for example `B-PER` followed by `B-PER` for the same entity, or `I-PER` after `O`.

Exp05 fixes some inconsistencies **after** prediction. A **CRF** instead adds a learnable score for each **tag transition** $(y_{i-1} \rightarrow y_i)$ so the model prefers **globally coherent** sequences during training and decoding.

### 3.2 O-tag bias initialization (class imbalance)

Roughly **86%** of tokens are `O`. Souza et al. (2019) recommend initializing the **bias of the O class** to a positive constant (here **6.0**) so early training does not collapse everything to `O` before rare entity tags get a gradient signal.

Implementation: `BertCRFForTokenClassification` and `CascadedNERModelCRF` set `classifier.bias[o_id] = 6.0` when `O` is in the label set.

### 3.3 Cascaded + CRF

Exp04 teaches **modular** NER (entity? → B/I? → type?). Exp10 cascade **keeps** those three heads for step-wise metrics, and adds a **fourth head**: emissions over **full BIO-type tags** (e.g. `B-PER`, `I-LOC`, `O`) with a CRF loss. **Span-level pipeline F1** uses **Viterbi** on that joint head; step 3 **B/I type consistency** can still be applied post-decode (same idea as Exp05).

### 3.4 Fusion without retraining

Exp06 fuses Exp01 + Exp04 Excel outputs. Exp10 fusion does the same with **Exp10 regular CRF** + **Exp10 cascade CRF** workbooks—token alignment on `(sentence_id, token_idx)`.

---

## 4. Architecture diagrams

### 4.1 Regular BERT-CRF (`10_regular`)

```text
  WordPiece tokens
        │
        ▼
  ┌─────────────┐
  │  BERT encoder │
  └──────┬──────┘
         │ hidden states h_t
         ▼
  ┌─────────────┐
  │ Linear layer │  →  emission scores e_t(k)  for each tag k
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Linear-chain CRF │
  │  • train: -log P(y|x)  (forward–backward)
  │  • infer: Viterbi best path
  └─────────────┘
```

**Student exercise:** Compare `core/th_functions.py` (softmax head) with `core/bert_crf_training.py` (CRF head). What changes in `Trainer.predict`?

### 4.2 Cascaded CRF (`10_cascade`)

```text
  Shared BERT encoder
        │
        ├── entity head  (binary)     ─┐
        ├── bio head     (binary)     ├── original cascade losses (Exp04)
        ├── type head    (multi-class)─┘
        │
        └── tag emission head + CRF   ← joint BIO-type sequence (Exp10)
                    │
                    ▼
              Viterbi → pred_bio, pred_etype → span F1
                    │
                    optional THESIS_STEP3_BI_TYPE_RECONCILE (Exp05-style)
```

### 4.3 Ready fusion (`10_fusion_ready`, `10_svm_ready`)

```text
  exp10_regular.xlsx  ── token_predictions sheet
           │
           ├──── merge on (sentence_id, token_idx) ────► fusion rule
           │                                              │
  exp10_cascade.xlsx ── detailed_results (predicted)     ▼
                                              fused_pred_label → seqeval F1
```

---

## 5. Key equations (CRF)

For tag sequence $\mathbf{y} = (y_1,\ldots,y_T)$ and emissions $\mathbf{e}_t \in \mathbb{R}^K$:

**Score of a path:**

$$
s(\mathbf{y}) = \sum_{t=1}^{T} \left( A_{y_{t-1}, y_t} + e_t(y_t) \right)
$$

where $A$ is the learned transition matrix (implemented as `LinearChainCRF.transitions`, with extra START/STOP states).

**Training loss (negative log-likelihood):**

$$
\mathcal{L} = -\log \frac{\exp(s(\mathbf{y}^{*}))}{\sum_{\mathbf{y}'} \exp(s(\mathbf{y}'))}
$$

The denominator uses the **forward algorithm** (`_partition_function` in `core/crf_layer.py`).

**Inference:** **Viterbi** (`viterbi_decode`) finds $\arg\max_{\mathbf{y}} s(\mathbf{y})$.

---

## 6. Step-by-step: what happens when you run the cross-comparison

Command example:

```bash
python run_cross_data_model_comparison.py \
  --experiments 10_regular,10_cascade,10_fusion_ready,10_svm_ready \
  --models dictabert,berel \
  --base-mode auto
```

For each **(model, data condition, seed)**:

1. **`10_regular` / `10_cascade` (training)**  
   - Runner calls `_ensure_base_artifacts_crf` once per (model, condition): trains both if not cached.  
   - Individual experiment IDs then **load JSON results** from cache (same pattern as 01/04).

2. **`10_fusion_ready` / `10_svm_ready` (inference)**  
   - Runner sets `THESIS_READY_EXP10_REGULAR_XLSX` and `THESIS_READY_EXP10_CASCADE_XLSX` from cache.  
   - Fusion scripts read token-level sheets and write Excel + JSON under `outputs/exp10_fusion_*`.

3. **Disk cleanup**  
   - After training, `experiments/model_cleanup.py` removes heavy checkpoint folders (default `THESIS_DELETE_MODELS_AFTER_TRAIN=1`).  
   - **Excel and JSON metrics are kept.**

4. **Consolidated error analysis**  
   - If any Exp10 ID is selected, the runner merges error-analysis sheets into  
     `outputs/cross_comparison/consolidated_error_analysis_exp10_<timestamp>.xlsx`.

---

## 7. Environment variables (teaching checklist)

| Variable | Purpose |
|----------|---------|
| `THESIS_PRESPLIT_TRAIN_JSON` / `THESIS_PRESPLIT_EVAL_JSON` | Same splits as Exp07/Exp08 (set by cross-comparison runner). |
| `THESIS_MODEL_NAME` | Which Hebrew transformer to fine-tune. |
| `THESIS_SPLIT_SEED` | Reproducible split / run seed. |
| `THESIS_CURRENT_EXP_ID` | Used for checkpoint directory naming. |
| `THESIS_STEP3_BI_TYPE_RECONCILE=1` | Enables B/I entity-type reconciliation on cascade CRF outputs. |
| `THESIS_DELETE_MODELS_AFTER_TRAIN=1` | Delete checkpoints after Exp10 training (default on). |
| `THESIS_SAVE_TRAINED_MODELS=1` | Optionally save full models or SVM router artifacts. |
| `THESIS_READY_EXP10_REGULAR_XLSX` | Explicit path to regular CRF metrics Excel (fusion). |
| `THESIS_READY_EXP10_CASCADE_XLSX` | Explicit path to cascade CRF metrics Excel (fusion). |

---

## 8. Output workbook structure (what to show in class)

### Exp10 regular (`regular_ner_crf_results_*.xlsx`)

| Sheet | Content |
|-------|---------|
| `metrics` | One-row summary F1 / precision / recall. |
| `detailed_results` | Sentence-level true vs predicted label strings. |
| `token_predictions` | Per-token labels + emission-based confidence (for fusion). |
| Error-analysis sheets | From `experiments/error_analysis.py` (confusion, error types, examples). |

### Exp10 cascade (`cascaded_pipeline_crf_results_*.xlsx`)

Same layout as Exp04: `metrics` (per epoch / final_optimised), `detailed_results` with `eval_mode=predicted`.

### Exp10 fusion

Same sheets as Exp06 fusion (see `fusion_ready_sources.py` `_SHEET_DOCS`).

---

## 9. Suggested lab exercises

1. **Viterbi trace:** On a 5-token sentence, hand-compute Viterbi with a toy $3\times3$ tag set and compare to `LinearChainCRF.viterbi_decode` output.
2. **Ablation:** Run `10_regular` with O-bias 0 vs 6 and compare early-epoch F1 (requires small code change or env hook).
3. **Fusion comparison:** With cached artifacts, run only `10_fusion_ready` vs `10_svm_ready` and compare disagreement recovery rates in the `disagreement_analysis` sheet.
4. **Invalid BIO rate:** Count tokens where independent argmax on emissions violates BIO rules vs CRF Viterbi path.

---

## 10. References (as used in thesis planning)

- Lample et al. (2016) — neural architectures for NER; character LSTM + CRF motivation.  
- Souza et al. (2019) — BERT-CRF for NER; O-bias trick.  
- Ben-Gigi et al. (2025) — BERT-CRF strong baseline on Rabbinic Hebrew citation NER.

---

## 11. Relationship to Experiments 01–06 (summary table)

| Exp01–06 concept | Exp10 analogue |
|------------------|----------------|
| Regular softmax NER | `10_regular` BERT-CRF |
| Cascaded pipeline | `10_cascade` + joint tag CRF |
| Step3 consistency (05) | Reconciliation flag on cascade CRF decode |
| Confidence fusion (06_ready) | `10_fusion_ready` |
| SVM router (06_svm_ready) | `10_svm_ready` |

Experiments **01, 04, 05_ready, 06_*** are **unchanged**; Exp10 is an **additive** branch for thesis future-work evaluation.
