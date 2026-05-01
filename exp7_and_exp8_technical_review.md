# Exp07 and Exp08 Technical Review

## Purpose

This document gives a concise but implementation-faithful, step-by-step technical review of:

- Exp01 baseline training flow
- Exp07 split methods (all 9)
- Exp08 augmentation flow
- How everything is orchestrated in `run_cross_data_model_comparison.py`

Code basis:

- `experiments/experiment_01_regular_ner.py`
- `experiments/experiment_07_sentence_split_strategy.py`
- `experiments/experiment_08_llm_augmentation.py`
- `run_cross_data_model_comparison.py`

---

## 1) Exp01 Baseline (Reference Condition)

Exp01 is the base NER workflow used as the regular baseline.

### Step-by-step

1. Resolve dataset path (`THESIS_NER_CSV` or default `data/ner_dataset.csv`).
2. Resolve model via `configure_model_environment()`.
3. Resolve split seed from `THESIS_SPLIT_SEED` (default 42).
4. Check whether pre-split JSON files were injected by environment:
   - `THESIS_PRESPLIT_TRAIN_JSON`
   - `THESIS_PRESPLIT_EVAL_JSON`
5. Load and prepare dataframe with `PrepDataSetNERTraining().load_and_prepare_data(...)`.
6. Branch:
   - If pre-split exists: load sentence lists and run `run_training_with_presplit(...)`.
   - Otherwise: run default split/training pipeline `run_training_steps(...)`.
7. Compute evaluation outputs and aggregate global metrics (F1, precision, recall).
8. Write Excel + JSON artifacts under `outputs/exp01`.

### Important technical role

- In cross-comparison, Exp01 is executed repeatedly with external precomputed splits, so it acts as a common training/eval engine on top of each split condition.

---

## 2) Exp07 Split Strategy Experiment

Exp07 compares 9 sentence-level splitting methods. Each method is trained/evaluated across multiple seeds, then split artifacts are saved for reuse.

### Shared pipeline for all split methods

1. Load dataset and convert to sentence objects (`tf.train_data_fit(...)`).
2. For each seed in `base_seed ... base_seed + num_seeds - 1`:
   - For each split variant:
     - Build train/eval sentence sets at 70/30.
     - Train/evaluate NER with that split.
     - Record metrics and label-distribution diagnostics.
3. Aggregate per-variant mean/std metrics.
4. Choose best non-baseline variant by highest mean F1.
5. Save first-seed train/eval JSON for all variants into `outputs/exp07/splits`.
6. Save `split_meta.json` with baseline key, best key, files, and summary metadata.

### Exp07 Method 1: Baseline (Simple Random)

Implementation: `_simple_random_split(...)`

1. Copy sentence list.
2. Shuffle with deterministic RNG(seed).
3. Compute split index = `int(N * 0.7)` and clamp to keep both sides non-empty.
4. First segment -> train, remainder -> eval.

### Exp07 Method 2: Label-Aware Greedy

Implementation: `_label_aware_split(...)` wrapper over `tf.split_list(..., ensure_label_coverage=True)`.

1. Delegate split to core helper with label-coverage logic enabled.
2. Use 70/30 sentence split target.
3. Internally prioritize preserving non-O label distribution and maintaining train coverage.

### Exp07 Method 3: Rare-Label Boosted

Implementation: `_rare_label_boosted_split(...)`

1. Compute non-O global label token counts.
2. Define rare labels as labels with count <= median label frequency.
3. Mark all sentences containing any rare label.
4. Start train set from these rare-containing sentences.
5. If train is too large, downsample rare set to target size.
6. If train is too small, fill remaining slots from non-rare sentences (randomized).
7. Remaining sentences become eval.

### Exp07 Method 4: Inverse-Frequency Weighted (Presence-based)

Implementation: `_inverse_freq_weighted_split(...)`

1. Compute global non-O label token counts and `max_freq`.
2. For each sentence, get unique non-O labels (presence, not token multiplicity).
3. Score sentence as `sum(max_freq / count(label))` over unique labels.
4. Add tiny random tie-break noise.
5. Sort descending by score.
6. Take top `target_train_count` as train; rest as eval.

### Exp07 Method 5: Min-Max Equalized

Implementation: `_minmax_equalized_split(...)`

1. Shuffle input sentences (seeded).
2. Compute per-sentence non-O label token counts.
3. Build global counts and target train counts per label (`total * 0.7`, min 1.0).
4. Initialize empty train and zero current label counts.
5. Greedy loop until train target size:
   - For each candidate sentence not selected, simulate adding it.
   - Compute minimum label coverage ratio across all labels.
   - Select sentence that maximizes this minimum ratio.
6. Selected sentences -> train; remainder -> eval.

### Exp07 Method 6: Inverse-Frequency Token-Weighted

Implementation: `_inverse_freq_token_weighted_split(...)`

1. Compute global non-O counts and `max_freq`.
2. For each sentence, count non-O labels with multiplicity.
3. Score sentence as `sum(token_count_in_sentence(label) * max_freq / count(label))`.
4. Add tiny random tie-break noise.
5. Rank descending, take top `target_train_count` for train.
6. Remaining sentences -> eval.

### Exp07 Method 7: Inverse-Frequency Eval-Guaranteed

Implementation: `_inverse_freq_eval_guaranteed_split(...)`

Phase 1 (eval reservation):
1. Compute global non-O counts.
2. Sort labels by rarity (ascending count).
3. For each rare-to-common label, reserve one sentence into eval that contains it (if available).
4. Expand eval-covered label set from each reserved sentence.

Phase 2 (train fill):
5. Score non-reserved sentences by inverse-frequency presence score.
6. Select top-scoring sentences into train until train target size.
7. Eval is all non-train sentences (reserved + leftovers).

### Exp07 Method 8: Inverse-Frequency Log-Scaled

Implementation: `_inverse_freq_log_scaled_split(...)`

1. Compute global non-O counts and `max_freq`.
2. For each sentence, get unique non-O labels.
3. Score sentence as `sum(log(1 + max_freq / count(label)))`.
4. Add tiny random tie-break noise.
5. Rank descending and select top `target_train_count` for train.
6. Remaining sentences -> eval.

### Exp07 Method 9: Multilabel Stratified (Iterative Stratification)

Implementation: `_multilabel_stratified_split(...)`

1. Treat each sentence as a multilabel instance using the set of unique non-O labels in that sentence.
2. Build per-label sentence index lists and desired train/eval counts from the split ratio (70/30).
3. Process labels from rarest to most common (fewest unassigned examples first).
4. For each unassigned sentence containing the current label, compute each fold's remaining need across all labels present in that sentence.
5. Assign the sentence to the fold (train or eval) with greater remaining need (with tiny random tie-break noise).
6. Update current per-label counts after each assignment.
7. After label-driven assignment, distribute any remaining unassigned sentences (typically O-only) to satisfy overall train size.
8. Return final train/eval sentence lists.

Why this matters: unlike simple rarity heuristics, this method explicitly targets proportional label representation in both folds and is closer to true multilabel stratification (Sechidis et al., 2011).

### Exp07 output artifacts used downstream

- `outputs/exp07/splits/<variant>_train.json`
- `outputs/exp07/splits/<variant>_eval.json`
- `outputs/exp07/splits/split_meta.json`

---

## 3) Exp08 LLM Augmentation Experiment

Exp08 compares baseline vs augmented training while keeping eval fixed.

### Global run flow (per seed)

1. Build label-aware split (`tf.split_list(..., ensure_label_coverage=True)`) at 70/30.
2. Train baseline model on original train, evaluate on eval.
3. Generate augmented training sentences from train only.
4. Train augmented model on (original + generated) train, same eval.
5. Record metrics for both variants.
6. On first seed, save richer diagnostics and split artifacts.

### Step-by-step augmentation logic

Implementation core: `_augment_training_data(...)`

1. Resolve augmentation MLM model:
   - Prefer explicit `THESIS_AUGMENTATION_MODEL_NAME`.
   - Else infer from NER model family (dictabert/berel/etc.).
   - Prefer local MLM checkpoints/snapshots; fallback to Hub id.
2. Build fill-mask pipeline (local-only if local path or local-only env set).
3. Compute label stats on current train (`tf.generate_label_df(...)`), including deficits (`Delta to Max`).
4. Compute Q1 sentence-frequency threshold for "extremely rare" labels.
5. Compute inverse-frequency rarity scores over labels from current train.
6. For each label with positive deficit:
   - Gather candidate train sentences containing that label.
   - Score and rank candidates by rarity (`log(1 + max_count/count(label))` sum over labels in sentence).
   - Set generation target `target_total = deficit * multiplier`.
   - Extract known entity token targets from full dataset for that label.
7. Allocate generation budget across candidate sentences proportionally to rarity score.
8. For each candidate sentence:
   - Strategy A: Multi-position masking.
     - Find all positions of B-/I- tag for target label.
     - Mask one position at a time.
     - Run fill-mask with constrained `targets=known_entity_tokens`.
     - Keep valid substitutions, preserve label sequence.
   - Strategy B: Context-preserving duplication (for extremely rare labels).
     - If generation shortfall remains, duplicate original sentence copies.
9. Append generated samples, then recompute label stats and rarity counts as augmentation proceeds.
10. Shuffle generated sentences (seeded) to avoid order bias.
11. Return generated list + generation log dataframe.

### Exp08 output artifacts used downstream

- `outputs/exp08/splits/baseline_train.json`
- `outputs/exp08/splits/baseline_eval.json`
- `outputs/exp08/splits/augmented_train.json`
- `outputs/exp08/splits/augmented_eval.json` (same eval as baseline)
- `outputs/exp08/splits/split_meta.json`

---

## 4) Advanced Modeling Pipelines Used Downstream

In addition to the standard end-to-end NER baseline (Exp01), the project evaluates four more model-centric pipelines on the same data conditions. In practice, each of these methods is run on representative split conditions from Exp07 (baseline vs. improved split) and on Exp08-style train sets (original vs. augmented), so the comparison isolates how much of the performance change comes from the modeling pipeline itself versus the data preparation strategy.

### Exp03: AUC-2T (AUC-oriented Two-Task Training)

This method reformulates NER as two binary token-level tasks designed for highly imbalanced data. Instead of predicting the full BIO-plus-type label set, it predicts whether each token belongs to any entity and whether that entity token marks the beginning of a span. These two outputs are then combined into generic BIO labels, yielding `O`, `B-ENT`, or `I-ENT` rather than specific entity types such as PER, LOC, or ORG. The training objective uses an AUC-oriented margin loss to better separate the sparse entity tokens from the dominant `O` class. In this sense, Exp03 is best understood as a generic entity-span detector. Conceptually, it follows the general direction of two-task AUC-based approaches for imbalanced NER, including Nguyen et al. (2023), but in this repository it is implemented as a simplified generic-entity variant rather than a full typed NER reproduction.

### Exp04: Cascaded NER Pipeline

This pipeline extends the decomposition idea into a full typed NER system. First, it predicts whether each token belongs to any entity. Second, it assigns boundary labels, distinguishing beginning from inside positions for the detected entity tokens. Third, it predicts the specific entity type for each entity token. The final label is reconstructed from these three stages, allowing the model to move from generic entity detection to full typed NER. This staged design can make each subproblem easier to learn, although errors in earlier stages may propagate to later ones.

### Exp05: Cascaded Pipeline with Step-3 Consistency Check

This variant keeps the same three-stage cascaded model as Exp04 and adds a post-processing reconciliation step. After the BIO and entity-type predictions are produced, the pipeline checks for structurally inconsistent spans, such as incompatible type assignments within what should be a single entity. These conflicts are then resolved using a consistency rule based on the stronger probability signal. Thus, Exp05 is best described as a small corrective refinement of Exp04 rather than a new model architecture.

### Exp06: Fusion of Regular and Cascaded Models

This method combines the outputs of the standard end-to-end NER model and the cascaded pipeline. When both models predict the same label, that label is kept. When they disagree, the final label is chosen from the model with the higher confidence score. This makes Exp06 a lightweight ensemble layer built on top of Exp01 and Exp04, intended to exploit complementary strengths without introducing a more complex joint model.

---

## 5) How It Comes Together in run_cross_data_model_comparison.py

This script is the orchestrator that combines models, split conditions, and downstream experiments.

### Stage A: Prepare artifacts

1. Prepare Exp07 split artifacts (`saved`, `rerun`, or `auto`).
2. Prepare Exp08 augmentation artifacts (reuse or rerun).
3. Optionally prepare Exp07+Aug artifacts:
   - For each Exp07 variant train split, call Exp08 augmentation core.
   - Save `<variant>_augmented_train.json` + unchanged eval JSON in `outputs/exp07_augmented/splits`.

### Stage B: Build data conditions

Build one unified condition list from:

1. Exp07 variants (9 conditions typically).
2. Exp08 baseline and augmented (2 conditions).
3. Exp07+Aug variants (9 conditions typically, if generated).

Each condition defines:

- source tag (`exp07`, `exp08`, `exp07+aug`)
- condition key/label
- train JSON path
- eval JSON path
- baseline flag

### Stage C: Execute the run matrix

For each selected model, experiment, and condition:

1. Resolve model env (`THESIS_MODEL_NAME`, offline flags for local checkpoints).
2. Inject pre-split env:
   - `THESIS_PRESPLIT_TRAIN_JSON`
   - `THESIS_PRESPLIT_EVAL_JSON`
3. Import and run experiment module (`exp01`, `exp03`, `exp04`, `exp05`, `exp06`).
4. Collect returned metrics (F1, precision, recall, status).
5. Clear pre-split env.
6. Append run row and write progress checkpoint JSON.

### Stage D: Post-processing and outputs

1. Build summary pivot (best condition per model/experiment).
2. Build Exp07 deltas (variant - Exp07 baseline).
3. Build Exp08 deltas (augmented - Exp08 baseline).
4. Build Exp07+Aug deltas (split+aug - split only).
5. Build model-vs-model comparison table.
6. Build variant aggregate summary.
7. Save Excel + JSON under `outputs/cross_comparison` and update latest copies.

---

## 6) Reviewer Notes (Concise)

1. Exp01 is the common baseline engine and also the execution target for pre-split comparison runs.
2. Exp07 contributes deterministic split artifacts for eight sentence allocation policies.
3. Exp08 contributes controlled train-only augmentation while preserving eval integrity.
4. Cross comparison standardizes evaluation by forcing all downstream experiments to consume explicit train/eval JSON pairs.
5. The pipeline supports resumable long runs via progress checkpointing.
