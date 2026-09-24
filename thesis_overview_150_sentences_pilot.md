# 150-Sentence Pilot — Methods Reference (DictaBERT & BEREL)

This document describes **exactly** what runs when you launch the cross-comparison below on the **150-sentence subset** with **DictaBERT** and **BEREL**, including software libraries, metrics, confidences, fusion parameters, and outputs.

It complements the full thesis pipeline doc: [`theisis overview.md`](theisis%20overview.md) and fair-training defaults: [`cross_comparison_fair_training_overview.md`](cross_comparison_fair_training_overview.md).

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
  --experiments 01,04,05_ready,06_ready,06_svm_ready,06_svm_kernel_ready,06_nb_ready,06_lr_ready,06_rf_ready,06_mlp_ready \
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
| `--num-seeds` | `20` | Paired training seeds **42…61** (`DEFAULT_BASE_SEED=42`) for every model/condition (§4.3). |
| `--consolidated-error-analysis` | `all` | One merged error-analysis workbook (higher RAM than `split`). Summary tabs compare **four methods only**: direct NER (`01`), cascade (`04`), linear SVM router (`06_svm_ready`), RF router (`06_rf_ready`) — see §13.2.1. |
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
              │ 06_ready            │  confidence fusion
              │ 06_*_ready (ML)      │  learned routers
              └─────────────────────┘
```

**Important:** `06_ready` and all `06_*` ML routers fuse **Exp01 + Exp04** ready Excel outputs. They do **not** automatically use `05_ready` (B/I consistency) as the cascade side. `05_ready` is a **separate reported method** applied to Exp04 outputs.

| Exp ID | Trains encoder? | Inputs | Script / core module |
|--------|-----------------|--------|----------------------|
| `01` | **Yes** | Exp07 train/eval JSON | `experiments/experiment_01_regular_ner.py`, `core/th_functions.py` |
| `04` | **Yes** | Exp07 train/eval JSON | `experiments/experiment_04_auc_cascaded_pipeline.py`, `core/auc_cascaded_pipeline.py` |
| `05_ready` | No | Exp04 metrics xlsx | `experiments/experiment_05_ready.py` |
| `06_ready` | No | Exp01 + Exp04 xlsx | `experiments/experiment_06_fusion_ready.py` |
| `06_svm_ready` | No (fits sklearn router) | Exp01 + Exp04 | `experiments/experiment_06_fusion_svm_ready.py` → `fusion_router_ready_common.py` |
| `06_svm_kernel_ready` | No | same | `experiment_06_fusion_svm_kernel_ready.py` |
| `06_nb_ready` | No | same | `experiment_06_fusion_nb_ready.py` |
| `06_lr_ready` | No | same | `experiment_06_fusion_lr_ready.py` |
| `06_rf_ready` | No | same | `experiment_06_fusion_rf_ready.py` |
| `06_mlp_ready` | No | same | `experiment_06_fusion_mlp_ready.py` |

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

**Excluded** from cross-comparison (not in this run): `exp07_after_multilabel_stratified` (legacy non-paper variant).

Train/eval ratio: **≈ 0.7 / 0.3** at sentence level; no sentence appears in both splits.

### 4.3 Twenty paired seeds (`--num-seeds 20`)

Seed list: **42, 43, …, 61** (`DEFAULT_BASE_SEED + i`).

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

DictaBERT and BEREL use the **same** numeric profile in this pilot.

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

**Inference:** Thresholds on entity/BIO heads; composed into full BIO-type tags per token.

**Outputs** (sheet `detailed_results`, rows with `eval_mode=predicted`):

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

## 12. ML router fusion (`06_svm_ready`, `06_svm_kernel_ready`, `06_nb_ready`, `06_lr_ready`, `06_rf_ready`, `06_mlp_ready`)

Shared implementation: `experiments/fusion_router_ready_common.py`.

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

**Ready-protocol limitation:** router is **fit and evaluated on the same eval tokens** (exploratory upper bound). Strict generalization requires a held-out router split (`experiment_06_fusion_svm.py` full path).

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
| `06_svm_ready` | `LinearSVC` | `C=1.0`, `class_weight="balanced"`, `random_state=42`, `max_iter=5000` |
| `06_svm_kernel_ready` | `SVC` | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, `random_state=42` |
| `06_nb_ready` | `GaussianNB` | defaults `{}` |
| `06_lr_ready` | `LogisticRegression` | `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, `random_state=42` |
| `06_rf_ready` | `RandomForestClassifier` | `n_estimators=100`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |
| `06_mlp_ready` | `MLPClassifier` | `hidden_layer_sizes=(64,32)`, `max_iter=1000`, `early_stopping=True`, `random_state=42` |

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

#### 13.2.1 Consolidated error analysis (four-way comparison)

`consolidate_error_analysis.py` (invoked via `--consolidated-error-analysis`) builds thesis summary tabs with **exactly these pipelines** (non-CRF):

| Column | Source experiment | Role |
|--------|-------------------|------|
| Regular NER | `01` | Direct encoder NER |
| Cascade NER | `04` | Three-step AUC cascade |
| Linear SVM Fusion | `06_svm_ready` | `LinearSVC` disagreement router |
| RF Fusion | `06_rf_ready` | `RandomForestClassifier` disagreement router |

Other runs in the same cross-comparison (e.g. `05_ready`, `06_ready`, kernel SVM, NB, LR, MLP) stay in `cross_comparison_*.xlsx` but are **excluded** from `consolidated_error_analysis_*.xlsx`. Re-consolidate after training; no need to re-run `01`/`04` if metrics workbooks already exist.

Per-router sections in the consolidated workbook: **Linear SVM** and **RF** each get their own routing / disagreement breakdown (not pooled).

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
| Full thesis math | `theisis overview.md` §9–§13, §12B |

---

*Document version: aligned with git `main` after ready ML router fusion (`06_svm_kernel_ready`, `06_nb_ready`, `06_lr_ready`, `06_rf_ready`, `06_mlp_ready`).*
