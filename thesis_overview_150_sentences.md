# Thesis Overview — 150-Sentence Training Pilot

Consolidated documentation for the **150-sentence subset** (fixed `--subset-seed 42`), DictaBERT / BEREL cross-comparison, OOF routers, loss-weight calibration, and journal exports.

*Merged on 2026-09-30 from `thesis_overview_150_sentences_pilot.md` and shared overview supplements.*

## Master table of contents

| Part | Section |
|------|---------|
| **I** | 150-sentence pilot — CLI, data, experiments, metrics |
| **II** | Fair training (DictaBERT / BEREL pilot profile) |
| **III** | OOF fusion on 150 sentences (5 folds × ~30 test sentences) |
| **IV** | Journal one-command run (`--journal-paper`) for 150-sentence subset |
| **V** | Statistical significance testing (seed-count power, paired t-test / Wilcoxon, reporting template) |

**Full-corpus math and global architecture:** [`thesis_overview.md`](thesis_overview.md) Part I.

---



# Part I — 150-sentence pilot

This document describes **exactly** what runs when you launch the cross-comparison below on the **150-sentence subset** with **DictaBERT** and **BEREL**, including software libraries, metrics, confidences, fusion parameters, and outputs.

It complements the full thesis pipeline doc: [`theisis overview.md`](theisis%20overview.md), **primary OOF router fusion:** [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md), and fair-training defaults: [`cross_comparison_fair_training_overview.md`](cross_comparison_fair_training_overview.md).

---

## 1. Canonical Colab / CLI command

Set an isolated output folder (recommended name shown; any path works):

```python
OUTPUT = "outputs/cross_comparison_150sent_dictabert_berel_ml_routers"
```

```bash
python run_cross_data_model_comparison.py \
  --output-dir {OUTPUT} \
  --subset-sentences 150 \
  --subset-seed 42 \
  --models dictabert,berel \
  --experiments 01,04,05_ready,06_ready,06_svm_oof,06_svm_kernel_oof,06_nb_oof,06_lr_oof,06_rf_oof,06_mlp_oof \
  --skip-augmentation \
  --condition-sources exp07 \
  --exp07-source rerun \
  --base-mode auto \
  --num-seeds 20 \
  --consolidated-error-analysis all \
  --resume
```

### 1.1 Flag reference (this run)

| Flag | Value | Effect |
|------|--------|--------|
| `--output-dir` | `{OUTPUT}` | All cross-comparison Excel/JSON, checkpoints, subset CSV, and **Exp07 splits for this pilot** live here (does not overwrite global `outputs/cross_comparison/` unless you point there). |
| `--subset-sentences` | `150` | Sample **150 sentence ids** from the full corpus CSV before building splits. |
| `--subset-seed` | `42` | RNG seed for **which** 150 sentences are chosen (reproducible). |
| `--models` | `dictabert,berel` | Two encoders only (see §2). |
| `--experiments` | see command | Eleven experiment IDs (§3). |
| `--skip-augmentation` | on | **No** Exp07+LLM-augmentation conditions (`exp07+aug` omitted). |
| `--condition-sources` | `exp07` | Only Exp07 sentence-split strategies (§4). |
| `--exp07-source` | `rerun` | Regenerate `{OUTPUT}/exp07/splits/` from the **150-sentence CSV** (required with subset; do not use `saved` from full corpus). |
| `--base-mode` | `auto` | Reuse cached Exp01+Exp04 artifacts when `(model, condition)` matches; otherwise train bases. |
| `--num-seeds` | `20` (thesis) / **`3` (journal paper default)** | Persisted random seeds in `{OUTPUT}/training_seeds.json` (§4.3). `--journal-paper` only substitutes its 3-seed default when `--num-seeds` is **not** explicitly passed (and `THESIS_CROSS_NUM_SEEDS` is unset) — an explicit `--num-seeds 20` together with `--journal-paper` now correctly stays at 20 (fixed; previously collided with the default and silently collapsed to 3). Journal: see [`thesis_overview_journal_paper_results.md`](thesis_overview_journal_paper_results.md). |
| `--consolidated-error-analysis` | `all` | One merged error-analysis workbook (higher RAM than `split`). Summary tabs compare **every fusion method actually run**: direct NER (`01`), cascade (`04`), confidence fusion (`06_ready`), and all ML routers (linear SVM, RBF SVM, Naive Bayes, logistic regression, RF, MLP) — see §13.2.1. |
| `--resume` | on | Skip completed jobs in `{OUTPUT}/cross_comparison_progress_latest.json`. |

**Colab:** set `os.environ["THESIS_RUN_ENV"] = "colab"` and `WANDB_DISABLED=true` before running (see [`COLAB_README.md`](COLAB_README.md)).

---

## 2. Models in this run

| CLI key | Hugging Face id | Role |
|---------|-----------------|------|
| `dictabert` | `dicta-il/dictabert` | General-purpose Hebrew BERT |
| `berel` | `dicta-il/BEREL_3.0` | Hebrew BERT with Biblical/Rabbinical orientation |

Registry: `run_cross_data_model_comparison.py` → `MODEL_REGISTRY`.

Each job is indexed by **(model, experiment, condition_key, seed)**. The runner sets `THESIS_MODEL_NAME`, `THESIS_CURRENT_CONDITION_KEY`, and `THESIS_SPLIT_SEED` per job.

---

## 3. Experiment pipeline (dependency graph)

```
                    ┌─────────────┐
                    │  Exp07 splits │
                    │ (150-sent CSV)│
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │  Exp01   │              │  Exp04   │
        │ Regular  │              │ Cascaded │
        │   NER    │              │ pipeline │
        └────┬─────┘              └────┬─────┘
             │                         │
             │    ┌────────────────────┤
             │    ▼                    │
             │  ┌──────────┐           │
             │  │ 05_ready │ (post-process Exp04 only)
             │  └──────────┘           │
             │                         │
             └──────────┬──────────────┘
                        ▼
              ┌─────────────────────┐
              │ 06_ready            │  confidence fusion (ready xlsx)
              │ 06_*_oof (ML)       │  primary learned routers (nested CV)
              └─────────────────────┘
```

**Important:** `06_ready` fuses cached Exp01 + Exp04 Excel outputs. **`06_*_oof`** retrains Exp01 + Exp04 inside **5-fold stratified CV + OOF** on all 150 sentences (train ∪ eval JSON per condition) — see [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md). Appendix **`06_*_ready`** (in-sample router bound) is optional and not in the canonical command above.

| Exp ID | Trains encoder? | Inputs | Script / core module |
|--------|-----------------|--------|----------------------|
| `01` | **Yes** | Exp07 train/eval JSON | `experiments/experiment_01_regular_ner.py`, `core/th_functions.py` |
| `04` | **Yes** | Exp07 train/eval JSON | `experiments/experiment_04_auc_cascaded_pipeline.py`, `core/auc_cascaded_pipeline.py` |
| `05_ready` | No | Exp04 metrics xlsx | `experiments/experiment_05_ready.py` |
| `06_ready` | No | Exp01 + Exp04 xlsx | `experiments/experiment_06_fusion_ready.py` |
| `06_svm_oof` | **Yes** (many fold trains) | Merged train+eval sentences | `experiment_06_fusion_svm_oof.py` → `fusion_router_oof_common.py` |
| `06_svm_kernel_oof` | **Yes** | same | `experiment_06_fusion_svm_kernel_oof.py` |
| `06_nb_oof` | **Yes** | same | `experiment_06_fusion_nb_oof.py` |
| `06_lr_oof` | **Yes** | same | `experiment_06_fusion_lr_oof.py` |
| `06_rf_oof` | **Yes** | same | `experiment_06_fusion_rf_oof.py` |
| `06_mlp_oof` | **Yes** | same | `experiment_06_fusion_mlp_oof.py` |

Shared loaders: `experiments/fusion_ready_sources.py` (`load_regular_from_exp01`, `load_cascade_from_exp04`, `merge_regular_cascade`, `compute_metrics`).

---

## 4. Data: 150-sentence subset and Exp07 conditions

### 4.1 Subset construction

1. Read full labeled CSV (`data/ner_dataset.csv` or `THESIS_NER_CSV`).
2. Draw **150 unique sentence ids** with RNG seed **42**.
3. Write `{OUTPUT}/data/ner_dataset_150_seed42.csv` (token rows for those sentences only).
4. Set `THESIS_NER_CSV` to that file for Exp07 regeneration and all training.

Implementation: `_build_sentence_subset` in `run_cross_data_model_comparison.py`.

### 4.2 Exp07 split strategies (this run)

With `--exp07-source rerun`, splits are rebuilt under `{OUTPUT}/exp07/splits/` via `experiments/exp07_split_artifacts.py`:

| Variant key | Label (short) | Method |
|-------------|---------------|--------|
| `before_exp01_baseline` | Baseline (simple random split) | Shuffle sentences; ~70% train / ~30% eval |
| `after_label_aware_split` | Label-aware greedy | Preserve non-`O` label coverage in train (~70%) |
| `after_multilabel_iterative_paper` | Multilabel stratified (paper-style) | Iterative assignment by rare labels |

Train/eval ratio: **≈ 0.7 / 0.3** at sentence level; no sentence appears in both splits.

### 4.3 Twenty paired seeds (`--num-seeds 20`)

On the **first** run for a given `--output-dir`, the runner draws **20 fully random** integer seeds (unique, 1…2³¹−1), saves them to:

`{OUTPUT}/training_seeds.json`

Every later run with the same output folder **reuses that file** (including `--resume`), so training stochasticity stays stable across reruns.

| Control | Effect |
|---------|--------|
| (default) | Load or create `{OUTPUT}/training_seeds.json` |
| `--training-seeds-file path.json` | Use a custom seed list file |
| `--regenerate-training-seeds` | Write a **brand-new, fully random** list, replacing every existing seed — orphans all already-completed runs for the old seeds. Only use for a from-scratch restart, never to "add more seeds" to a run in progress. |
| `THESIS_CROSS_SEEDS_MASTER` | Seeds the RNG when **creating** the file (optional) |

Legacy consecutive seeds **42…61** apply only if no `training_seeds.json` exists and you use old tooling; the cross-comparison runner always persists random seeds now.

**Extending seed count without losing completed runs (e.g. 10 → 20):** `--regenerate-training-seeds` regenerates **all** seeds randomly, so it cannot be used to grow an existing seed list — doing so would silently discard every run already completed under the original seeds. Instead, append new random seeds to the existing list in place (read `training_seeds.json`, keep all existing entries, draw additional unique random integers up to the new `--num-seeds` target, write back with the same `"seeds"` / `"num_seeds"` schema) and leave `--regenerate-training-seeds` off. The 150-sentence journal notebook (`colab_150_sentences_journal_10seeds.ipynb`, cell "7a2 — Extend training seeds") implements exactly this: it is idempotent (no-op once the file already has ≥ the target count) and a plain `--resume` afterward trains only the newly-added seeds, since the original seeds' checkpoint rows are untouched.

For each Exp07 **base** condition, the runner expands to **20 seeded conditions** (`exp07_<variant>__seed<seed>`). `THESIS_SPLIT_SEED` is set per job (model initialization / training stochasticity). Exp07 JSON paths for non-augmented runs are shared per variant; the **same** train/eval files are evaluated under each paired seed for fair cross-seed comparison (see `theisis overview.md` §14).

With `--skip-augmentation`, **no** `exp07+aug` conditions are included.

---

## 5. Python libraries (by role)

From [`requirements.txt`](requirements.txt) and experiment imports:

| Layer | Packages | Use in this run |
|-------|----------|-----------------|
| Deep learning | **PyTorch** (`torch`), **Hugging Face Transformers**, **datasets**, **accelerate**, **huggingface_hub** | Exp01 & Exp04 fine-tuning, inference, tokenization |
| Tabular / ML | **numpy**, **pandas**, **scipy** | Metrics tables, merge, feature matrices |
| Classical ML routers | **scikit-learn** (`sklearn`) | `LinearSVC`, `SVC`, `GaussianNB`, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer` |
| NER evaluation | **seqeval** | Entity-level precision, recall, F1 |
| Reporting | **openpyxl**, **pandas** | Result workbooks (multi-sheet Excel) |
| Progress | **tqdm** | Training loops |
| Encoding | **chardet** | CSV encoding detection |

Optional: **joblib** (via sklearn) when `THESIS_SAVE_TRAINED_MODELS=1` to export fitted routers under `outputs/trained_models/`.

---

## 6. Fair training hyperparameters (Exp01 & Exp04)

Applied at runner start by `core/training_defaults.apply_fair_comparison_training_defaults()` and recorded in `{OUTPUT}/run_manifest.json` → `training_hyperparameters`.

| Setting | Exp01 (regular NER) | Exp04 (cascaded) |
|---------|---------------------|------------------|
| Epochs | **3** (`THESIS_NUM_EPOCHS`) | **10** (`THESIS_EXP04_EPOCHS`) |
| Learning rate | **5×10⁻⁵** (`THESIS_LEARNING_RATE`) | Encoder **2×10⁻⁵**, heads **1×10⁻³** (fixed in `TRAINING_CONFIG`) |
| FP16 (Colab) | **on** (`THESIS_TRAINER_FP16=1`) | N/A (custom loop) |
| Weight decay | **0** | per Exp04 config |
| Class weights (Exp01) | **off** unless `THESIS_BALANCED_CLASS_WEIGHTS=1` | — |
| Exp04 batch / accum | — | **16** / **2** (env-overridable) |
| Exp04 loss weights $\lambda_{bio}$, $\lambda_{type}$ | — | Defaults **10** / **5**; override with `THESIS_EXP04_LAMBDA_BIO`, `THESIS_EXP04_LAMBDA_TYPE` after validation grid (**§8**) |

DictaBERT and BEREL use the **same** numeric profile in this pilot. See [`cross_comparison_fair_training_overview.md`](cross_comparison_fair_training_overview.md) for the full env table.

---

## 7. Exp01 — Regular NER (`01`)

**Architecture:** Hebrew transformer encoder + linear token classification head over full BIO-type labels (single pass).

**Training:** Hugging Face `Trainer` on Exp07 train split; evaluation metrics on eval split; checkpoints may be deleted after train when `THESIS_DELETE_MODELS_AFTER_TRAIN=1` (metrics kept).

**Per-token outputs** (sheet `token_predictions` in results xlsx):

| Column | Meaning |
|--------|---------|
| `true_label` | Gold BIO tag |
| `pred_label` | Predicted BIO tag |
| `prob` | **Confidence** for fusion (see §8.1) |
| `entropy` | Shannon entropy of tag softmax |
| `margin` | Top-1 minus top-2 softmax probability |

Code: `_build_token_predictions_with_probs` in `experiment_01_regular_ner.py`.

---

## 8. Exp04 — AUC cascaded pipeline (`04`)

**Architecture:** Three heads on shared encoder (see `core/auc_cascaded_pipeline.py`):

1. **Entity detection** — binary (entity vs non-entity), sigmoid.
2. **BIO position** — B vs I on entity tokens, sigmoid.
3. **Entity type** — multi-class on entity tokens.

**Training objective:** $\mathcal{L} = \mathcal{L}_{entity} + \lambda_{bio}\mathcal{L}_{bio} + \lambda_{type}\mathcal{L}_{type}$. Each term is a **mean** over its mask (entity on all valid tokens; B/I and type only on gold entity tokens). Binary heads use **focal loss** ($\alpha=0.25$, $\gamma=2$); type head uses cross-entropy.

**Loss-weight selection (validation grid):** Coefficients are not fixed by hand alone. Run `experiments/experiment_04_loss_weight_grid.py` on the Exp07 **validation** split (default grid $\lambda_{bio},\lambda_{type} \in \{1,5,10\}$, nine configs). Selection metric: **validation pipeline span F1** (`final_optimised`, `eval_mode=predicted`, after threshold sweep). Test / held-out reporting splits are not used for this search. Results: `outputs/exp04_lambda_grid/`. Export the winner before large cross-comparison jobs:

```bash
export THESIS_EXP04_LAMBDA_BIO=<selected>
export THESIS_EXP04_LAMBDA_TYPE=<selected>
```

Full math and protocol: `theisis overview.md` **§9.4–§9.6**.

**Inference:** After training, grid-search $\tau_e,\tau_b \in \{0.10,\ldots,0.90\}$ on validation (maximize $F1_{entity}+F1_{bio}$), then compose thresholds with type argmax into full BIO-type tags per token.

**Outputs** (Exp04 metrics xlsx):

| Sheet | Content |
|-------|---------|
| `loss_config` | `lambda_bio`, `lambda_type`, split seed |
| `metrics` | Epoch metrics + `final_optimised` row |
| `detailed_results` | Token-level predictions (`eval_mode=predicted` for fusion) |

Token columns in `detailed_results`:

| Column | Meaning |
|--------|---------|
| `pred_bio`, `pred_etype` | Cascade BIO + type |
| `entity_prob`, `bio_prob` | Sigmoid confidences for fusion composition |
| `true_bio`, `true_etype` | Gold parts |

---

## 9. Exp05_ready — B/I consistency (`05_ready`)

**No retraining.** Loads Exp04 `detailed_results`, applies **within-sentence** rule:

If token *i* is `B-X` and token *i+1* is `I-Y` with **X ≠ Y**, set both types to the side with higher `bio_prob` (tie-break: keep B side if equal).

Reconstructs full labels and recomputes entity-level F1 with **seqeval**.

Code: `_apply_bi_consistency` in `experiment_05_ready.py`.

---

## 10. Confidence definitions (Exp06 family)

All fusion methods align tokens with an **inner join** on `(sentence_id, token_idx)`. Only tokens present in **both** Exp01 and Exp04 exports are fused.

### 10.1 Regular confidence `regular_prob` (Exp01)

Softmax over all tag logits at token *i*:

$$
p_i^{reg} = \max_k P(y_i = k \mid x)
$$

$$
H_i^{reg} = -\sum_k P(y_i=k)\log(P(y_i=k)+\epsilon)
$$

$$
margin_i^{reg} = p_{i,(1)} - p_{i,(2)}
$$

### 10.2 Cascaded confidence `cascade_prob` (Exp04)

Let $p_i^{entity}$ = `entity_prob`, $p_i^{bio}$ = `bio_prob`, and `pred_bio` the cascade BIO decision.

If `pred_bio = O`:

$$
p_i^{cas} = 1 - p_i^{entity}
$$

Else (`B` or `I`):

$$
p_i^{cas} = p_i^{entity} \cdot p_i^{bio}
$$

**Derived (analysis / routers, not used in plain confidence fusion):**

$$
H_i^{cas} = -p_i^{cas}\log(p_i^{cas}+\epsilon) - (1-p_i^{cas})\log(1-p_i^{cas}+\epsilon)
$$

$$
margin_i^{cas} = 2|p_i^{cas} - 0.5|
$$

Loader: `load_cascade_from_exp04` in `fusion_ready_sources.py`.

### 10.3 Merged extras (routers)

$$
prob\_diff_i = p_i^{reg} - p_i^{cas},\quad |prob\_diff|_i,\quad max\_prob_i = \max(p_i^{reg}, p_i^{cas})
$$

Label parts: `regular_bio`, `regular_etype`, `cascade_bio`, `cascade_etype` from splitting predicted tags (`B-PER` → `B`, `PER`; `O` → `O`, `None`).

---

## 11. Exp06_ready — Confidence fusion

**Rule** (`experiment_06_fusion_ready.py`):

- If $\hat{y}_i^{reg} = \hat{y}_i^{cas}$: fused = agreed tag; `selected_source = agree`.
- If they **disagree**: pick label from side with **larger** $p_i^{reg}$ vs $p_i^{cas}$ (ties → regular).
- `selected_confidence` = confidence of the chosen side.

Uses **only** $p_i^{reg}$ and $p_i^{cas}$ (not entropy/margin).

---

## 12. ML router fusion — primary (`06_*_oof`)

**Full protocol:** [`thesis_overview_fusion_oof_cv.md`](thesis_overview_fusion_oof_cv.md)  
Shared implementation: `experiments/fusion_router_oof_common.py`.

On **150 sentences**, default **5** outer folds ⇒ **~30 sentences** per held-out outer test. Inner default **4** folds on ~120 outer-train sentences for OOF Exp01 + Exp04 predictions before router fit.

### 12.1 Agreement passthrough

If $\hat{y}_i^{reg} = \hat{y}_i^{cas}$, fused label = shared prediction (no router call).

### 12.2 Router training rows (disagreements only)

On disagree tokens, define:

$$
r_i = \mathbf{1}[\hat{y}_i^{reg} = y_i],\quad c_i = \mathbf{1}[\hat{y}_i^{cas} = y_i]
$$

**Training target** $z_i$:

| Case | Target |
|------|--------|
| $r_i=1$, $c_i=0$ | `regular` |
| $r_i=0$, $c_i=1$ | `cascade` |
| both correct or both wrong | **discarded** (ambiguous) |

If fewer than two classes remain, router training fails → **fallback to §11 confidence rule**.

**Primary OOF protocol:** router is trained on **inner OOF** predictions from outer-train only; **outer-test** sentences are never used for router fit. Pooled outer-test F1 covers all 150 sentences once.

### 12.A Appendix — in-sample ready routers (`06_*_ready`)

Optional fast path: `fusion_router_ready_common.py` — router fit and scored on the **same** Exp07 eval tokens (upper bound only). Not for main thesis tables.

### 12.3 Feature vector (all routers)

**Numeric (7)** → `StandardScaler`:

`regular_prob`, `cascade_prob`, `regular_margin`, `cascade_margin`, `prob_diff`, `abs_prob_diff`, `max_prob`

**Categorical (4)** → `OneHotEncoder(handle_unknown="ignore")`:

`regular_bio`, `regular_etype`, `cascade_bio`, `cascade_etype`

Constants: `FUSION_ROUTER_NUMERIC_FEATURES`, `FUSION_ROUTER_CATEGORICAL_FEATURES` in `fusion_ready_sources.py`.

**Not used as router inputs:** token text, sentence id, gold label, `regular_entropy` / `cascade_entropy`.

### 12.4 Classifiers and hyperparameters

All routers use sklearn `Pipeline`: `ColumnTransformer` → classifier.

| Experiment ID | sklearn class | Parameters (`fusion_ready_sources.py`) |
|---------------|---------------|----------------------------------------|
| `06_svm_oof` | `LinearSVC` | `C=1.0`, `class_weight="balanced"`, `random_state=42`, `max_iter=5000` |
| `06_svm_kernel_oof` | `SVC` | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, `random_state=42` |
| `06_nb_oof` | `GaussianNB` | defaults `{}` |
| `06_lr_oof` | `LogisticRegression` | `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, `random_state=42` |
| `06_rf_oof` | `RandomForestClassifier` | `n_estimators=100`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| `06_mlp_oof` | `MLPClassifier` | `hidden_layer_sizes=(64,32)`, `max_iter=1000`, `early_stopping=True`, `random_state=42` |

### 12.5 Inference on disagreements

Router predicts `regular` or `cascade` → copy that side’s **label** and set `selected_confidence` to that side’s $p_i^{reg}$ or $p_i^{cas}$.

### 12.6 Naive Bayes reservation (thesis wording)

**GaussianNB** assumes feature independence given the class. This pilot’s features include **correlated numerics** (e.g. `prob_diff` is a function of `regular_prob` and `cascade_prob`) and **one-hot categoricals** that co-vary with confidence. NB is included as an **empirical baseline**, not a theoretically ideal match for this feature design. Prefer linear/kernel SVM, logistic regression, random forest, or MLP when interpreting which meta-learner fits the routing task.

---

## 13. Evaluation metrics

### 13.1 Primary: entity-level F1 (seqeval)

**Library:** `seqeval.metrics.f1_score`, `precision_score`, `recall_score`.

**Unit of evaluation:** **entity spans**, not tokens. A span is $(start, end, type)$ with exclusive end index.

A prediction is correct only if **boundaries and type** match gold (strict CoNLL-style span matching used by seqeval).

$$
TP = |P \cap G|,\quad FP = |P \setminus G|,\quad FN = |G \setminus P|
$$

$$
Precision = \frac{TP}{TP+FP},\quad Recall = \frac{TP}{TP+FN},\quad F1 = \frac{2PR}{P+R}
$$

**Application:**

- Exp01 / Exp04: reported eval F1 from training scripts + token-level exports.
- Exp05 / Exp06*: `compute_metrics()` in `fusion_ready_sources.py` builds sentence lists via `to_seqeval_lists` and scores **`fused_pred_label`** (or 05’s corrected labels) against **`true_label`** from Exp01 alignment.

### 13.2 Secondary (workbook sheets)

Result xlsx files include token-level **confusion matrix**, **per-type metrics**, **error_type** taxonomy (`classify_error`), **confidence_analysis**, **disagreement_analysis**, **entity_length_analysis**. Token-level counts can differ from entity-level F1.

#### 13.2.1 Consolidated error analysis (all fusion methods)

`consolidate_error_analysis.py` (invoked via `--consolidated-error-analysis`) builds thesis summary tabs with **every** fusion/router variant actually trained, not just a Linear-SVM + RF subset (`METHOD_SPECS` / `FOCUS_THESIS_ERROR_ANALYSIS_EXP_IDS`):

| Column | Source experiment(s) | Role |
|--------|-------------------|------|
| Regular NER | `01` (or `10_regular`) | Direct encoder NER |
| Cascade NER | `04` (or `10_cascade`) | Three-step AUC cascade |
| Confidence Fusion | `06_ready` (or `10_fusion_ready`) | Scalar-confidence arbitration (§11) |
| Linear SVM Fusion | `06_svm_oof` (OOF primary; `06_svm_ready` / `10_svm_ready` appendix) | `LinearSVC` disagreement router |
| RBF SVM Fusion | `06_svm_kernel_oof` (`06_svm_kernel_ready` / `10_svm_kernel_ready` appendix) | RBF `SVC` disagreement router |
| Naive Bayes Fusion | `06_nb_oof` (`06_nb_ready` / `10_nb_ready` appendix) | `GaussianNB` disagreement router |
| Logistic Regression Fusion | `06_lr_oof` (`06_lr_ready` / `10_lr_ready` appendix) | `LogisticRegression` disagreement router |
| RF Fusion | `06_rf_oof` (`06_rf_ready` / `10_rf_ready` appendix) | `RandomForestClassifier` disagreement router |
| MLP Fusion | `06_mlp_oof` (`06_mlp_ready` / `10_mlp_ready` appendix) | `MLPClassifier` disagreement router (recognized if run; not in the default `EXPERIMENTS` list) |

Every experiment ID selected via `--experiments` that matches one of the rows above is folded into its column automatically; **Confidence Fusion has no trained classifier**, so it is included in the FP/FN/Type/Boundary error-type table but has no routing-decision section (no `selected_source` classifier to analyze). Re-consolidate after training; no need to re-run `01`/`04` if metrics workbooks already exist.

Per-router sections in the consolidated workbook: **every** ML router (Linear SVM, RBF SVM, Naive Bayes, Logistic Regression, RF, MLP — whichever were actually run) gets its own Routing Decision / Disagreement / Error-Types-by-Route breakdown (not pooled).

### 13.3 Fusion-specific counters

Metrics sheet records: `tokens_aligned`, `disagreements`, `agreements`, `selected_regular` / `selected_cascade` counts, `truth_label_mismatch_between_sources` (should be 0 when gold is consistent).

---

## 14. Caching, resume, and artifacts

### 14.1 Base cache (`--base-mode auto`)

Exp01 + Exp04 are trained once per `(model_id, condition train/eval paths)` and indexed in `{OUTPUT}/cross_comparison_base_ready_index.json`. Ready experiments (`05_ready`, `06_*`) read cached xlsx paths from that index.

### 14.2 Resume

`--resume` loads `{OUTPUT}/cross_comparison_progress_latest.json`, skips successful `(model, experiment, condition)` keys, retries prior errors.

### 14.3 Main outputs

| Path | Content |
|------|---------|
| `{OUTPUT}/data/ner_dataset_150_seed42.csv` | 150-sentence subset |
| `{OUTPUT}/exp07/splits/` | Train/eval JSON + `split_meta.json` |
| `{OUTPUT}/run_manifest.json` | CLI args, subset info, training hyperparameters |
| `{OUTPUT}/cross_comparison_*.xlsx/json` | Aggregated F1 table |
| `{OUTPUT}/consolidated_error_analysis_all_*.xlsx` | Merged error-analysis (this flag) |
| `outputs/exp01/`, `outputs/exp04/`, `outputs/exp06_*_ready/` | Per-experiment metrics workbooks (paths referenced in index) |
| `outputs/exp04_lambda_grid/` | Validation grid for $\lambda_{bio}$, $\lambda_{type}$ (heatmap + selection; run before locking weights) |

---

## 15. Scale of this run (order of magnitude)

Let:

- $M = 2$ models (dictabert, berel),
- $V \approx 3$ Exp07 variants (after exclusions),
- $S = 20$ seeds,
- $E = 11$ experiments ($2$ train + $9$ ready).

**Condition count** per model: $V \times S = 60$. **Total jobs** $\approx M \times E \times V \times S = 2 \times 11 \times 3 \times 20 = 1320$ (exact $V$ from `split_meta.json` after Exp07 rerun).

Training cost dominates Exp01 + Exp04; ready fusion steps are CPU-only and fast once bases exist.

---

## 16. Related source files (quick index)

| Topic | File |
|-------|------|
| Runner / CLI | `run_cross_data_model_comparison.py` |
| Fusion loaders & metrics | `experiments/fusion_ready_sources.py` |
| ML routers | `experiments/fusion_router_ready_common.py` |
| Exp07 splits | `experiments/exp07_split_artifacts.py` |
| Training defaults | `core/training_defaults.py` |
| Exp04 loss-weight grid | `experiments/experiment_04_loss_weight_grid.py` |
| Full thesis math | `theisis overview.md` §9–§13, §12B |

---

*Document version: includes Exp04 validation grid for cascaded loss weights (`experiment_04_loss_weight_grid.py`, `THESIS_EXP04_LAMBDA_*`).*

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

# Part III — OOF router fusion (150 sentences)

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
  fusion F1 — the fold models train on ~120 sentences vs. ~105 for the holdout, so the difference
  conflates protocol with method.
- Keep holdout results in a **separate** table with its own caption (seed-paired significance and
  the random-vs-label-aware split comparison), never as extra columns beside OOF numbers.

# Part IV — Journal paper workflow (150-sentence example)

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

**File:** `{output-dir}/unified_protocol_table.xlsx` — built by `build_unified_protocol_table.py`
(see Part V §V.1). **This is the single file to write the paper from:** it re-scores every method on
the same outer-test tokens *and* copies the runner's `journal_*` sheets in verbatim.

| Sheet | Paper use |
|-------|-----------|
| **`fold_summary`** | Main-table cells: mean ± SD F1 over all `(seed, fold)` units, every method |
| **`fusion_vs_base`** | Headline ΔF1, fusion minus each baseline on identical tokens |
| **`paired_fold_tests`** | Wilcoxon + t-test p-values, Holm-adjusted per `(model, split_condition)` |
| **`unified_main_table`** | Pooled outer-test F1/P/R, one protocol |
| **`journal_lambda_grid`** *(carried)* | Selected λ from calibration cache — Methods section |
| **`journal_loss_config`** *(carried)* | λ recorded per Exp04 run |
| **`journal_main_table`** *(carried)* | Seed-based holdout numbers — separate table only |

**Runner output:** `{output-dir}/cross_comparison_latest.xlsx` is still written by cell 8 / the
runner and remains the source for the carried sheets above. Do **not** mix its holdout F1 with the
OOF F1 in one table: outer folds train on ~120 of 150 sentences while the 70/30 holdout trains on
~105, so a mixed table credits fusion with a training-data advantage unrelated to fusion.

> `journal_paired_fold_deltas` is **not** emitted — see the correction in Part V §V.1.

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

# Part V — Statistical significance testing

**Status for this pilot:** the 150-sentence cross-comparison referenced throughout Parts I–IV (DictaBERT + BEREL, all 6 primary fusion methods) was run with **`--num-seeds 20`** (original 10 seeds + 10 extended in place per §4.3, no runs discarded). This satisfies the "publication-ready" seed-count guidance in §V.2 below, and gives `paired_fold_tests` (§V.1) up to `20 × 5 = 100` paired observations per method comparison.

*Merged from `STATISTICAL_SIGNIFICANCE_GUIDE.md` (2026-10-04); that file is now a pointer to this section.*

## V.1 Primary path: `build_unified_protocol_table.py` (use this first)

> **Correction (2026-10-05).** Earlier revisions of this section claimed that
> `journal_paired_fold_deltas` in `cross_comparison_*.xlsx` already holds the fusion-vs-baseline
> tests. **It does not, and the sheet is not emitted at all.** `collect_oof_fold_long()` in
> `experiments/journal_results_export.py` skips every row whose `experiment_id` lacks `_oof`:
>
> ```python
> if OOF_EXPERIMENT_SUFFIX not in exp_id:
>     continue
> ```
>
> so `exp01`, `exp04` and `exp06_ready` never enter `fold_long`. In
> `paired_fold_method_comparison()` both entries of `JOURNAL_BASELINE_EXP_IDS` are then absent
> from `available`, the pair-building loop `continue`s on both, `pairs` stays empty, and the
> function returns an empty frame — which the writer skips via `if not journal_paired_fold_df.empty`.
> Net effect: **no `journal_paired_fold_deltas` sheet exists in the workbook.**

Run the standalone builder instead. It re-scores Exp01, Exp04, Exp05 repair, confidence fusion and
every OOF router on the **same** pooled outer-test tokens — supplying the per-fold base-model F1
that `fold_long` is missing — and then runs the paired tests:

```bash
python build_unified_protocol_table.py \
  --oof-dir {output-dir} \
  --splits-dir {output-dir}/exp07/splits \
  --reference-xlsx {output-dir}/cross_comparison_latest.xlsx \
  --output {output-dir}/unified_protocol_table.xlsx
```

| Sheet in `unified_protocol_table.xlsx` | What it gives you |
|----------------------------------------|--------------------|
| `paired_fold_tests` | Paired **Wilcoxon signed-rank + t-test** p-values for every fusion method vs. Regular NER, Cascade NER, cascade+repair and confidence fusion, paired by `(training_seed, outer_fold)`, **Holm-adjusted** within each `(model, split_condition)` family. |
| `fold_summary` | Mean ± SD F1 over all `(seed, fold)` units per method — the main-table cells. |
| `fusion_vs_base` | ΔF1 of fusion minus each baseline on identical tokens. |
| `unified_main_table` | Pooled outer-test F1/P/R per method, one evaluation protocol. |
| `per_fold_f1` | Raw per-`(seed, fold)` F1 — the paired units behind the tests. |
| `paired_tests` *(carried over)* | Paired t-test + Wilcoxon across shared seeds for exp07-vs-exp07+aug and exp08 ablations (§13.1 of Part I). |

Because pairing is by `(training_seed, outer_fold)`, raising `--num-seeds` directly multiplies
`n_pairs` (seeds × 5) and therefore statistical power — this is exactly why extending from 10 to 20
seeds (§4.3) was worth doing before finalizing significance claims. **Check `n_pairs` in the output:**
it should read 100 for this pilot. If it reads 5, only one seed's OOF workbooks were discovered, and
at *n*=5 a two-sided signed-rank test cannot fall below *p*=0.0625 regardless of effect size.

**Caveat to state in Methods:** outer folds within one partition share training data, so paired CV
tests are *liberal* rather than conservative (Dietterich 1998; Bengio & Grandvalet 2004). Because
`THESIS_SPLIT_SEED` drives fold assignment (`split_seed + 1000`), each training seed yields a
**different** 5-fold partition, so the seed dimension is genuine repeated CV rather than reruns of a
single partition — this is the main thing mitigating the dependency.

**Use §V.2–§V.6 below only for:** (a) understanding *why* seed count matters, (b) ad-hoc comparisons
the builder doesn't cover (e.g. DictaBERT vs. BEREL head-to-head on one condition), or (c) a
>2-method omnibus test (Friedman).

## V.2 Why seed count matters (power analysis)

**Seeds are the primary source of variance** for paired significance testing — they control weight initialization, data shuffling, and training stochasticity. Data-split strategies (Exp07 variants) are systematic, not random, conditions: useful for generalization claims but not a substitute for seed variance in a paired test.

| Term | Meaning | Effect |
|------|---------|--------|
| **Seeds** | Different random initializations | Creates paired observations for the same data condition |
| **Splits** | Different train/test partitions | Tests generalization across data conditions |
| **Runs** | Repeating the whole experiment | Same as seeds if you change the seed each time |

**Bottom line:** for significance testing, seeds = runs; each seed creates one paired observation.

With only 3 seeds (the `--journal-paper` default when `--num-seeds` is not explicit — see Part I §1.1), a paired t-test has very little power: p-values rarely reach < 0.05 unless the effect is huge.

### Sample-size requirements for α = 0.05

| Effect size | Required seeds | Power |
|-------------|-----------------|-------|
| Large (d=0.8) | 10 seeds | ~75% |
| Large (d=0.8) | 15 seeds | ~88% |
| Medium (d=0.5) | 20 seeds | ~75% |
| Medium (d=0.5) | 30 seeds | ~87% |
| Small (d=0.2) | 50+ seeds | ~50%+ |

**Practical recommendation:** 10–20 seeds for detecting meaningful differences; this pilot uses 20.

### Run-configuration reference

| Goal | Seeds | Command |
|------|-------|---------|
| Quick sanity check | 3 | `--num-seeds 3` |
| Moderate confidence | 10 | `--num-seeds 10` |
| Publication-ready | 20 | `--num-seeds 20` (used here) |
| High-confidence | 30 | `--num-seeds 30` |

## V.3 Statistical test formulas (reference)

**Paired t-test** (recommended primary test for F1 scores across matched seeds/folds):

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

The strongest evidence appears when both the t-test and Wilcoxon agree ($p_{t} < 0.05$ and $p_{Wilcoxon} < 0.05$) — a result consistent across seeds/folds, not caused by one lucky split.

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

For comparisons `paired_fold_tests` doesn't cover out of the box — e.g. DictaBERT vs. BEREL head-to-head on one specific condition, read directly from `cross_comparison_latest.json`:

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
