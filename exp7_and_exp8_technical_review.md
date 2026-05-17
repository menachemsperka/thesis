# Exp07 and Exp08 Technical Review

## Purpose

This document gives a concise but implementation-faithful, step-by-step technical review of:

- Exp01 baseline training flow
- Exp07 split methods (all 4)
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

Exp07 compares 4 sentence-level splitting methods. Each method is trained/evaluated across multiple seeds, then split artifacts are saved for reuse.

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

Step-by-step (what it does internally at a high level):

1. Start from sentence-level units (not token-level rows), where each sentence carries a set/list of non-O labels.
2. Compute the global target split size (70/30 by sentence count).
3. Compute label occurrence statistics from the full sentence pool.
4. Enable `ensure_label_coverage=True`, which activates a label-aware assignment path instead of plain random slicing.
5. Greedily allocate sentences while trying to:
   - keep rare labels represented in train,
   - avoid obvious label dropout,
   - and stay close to the target fold sizes.
6. If a sentence contains labels that are currently underrepresented in train, the helper tends to place it in train first.
7. Continue assignment until all sentences are placed, then return train/eval lists.

How Method 2 preserves label representation:

1. It uses label-presence heuristics during assignment, rather than pure random placement.
2. It prioritizes train coverage for scarce labels so the model does not miss them during training.
3. It reduces (but does not mathematically eliminate) cases where a label is absent from one fold.

Limitations of Method 2 (important for comparison):

1. It is still a generic greedy heuristic, not a full multilabel stratification algorithm.
2. It does not explicitly optimize per-label deficits in both folds at each assignment step.
3. It can preserve train coverage well but still drift on eval proportionality for some minority labels.

### Exp07 Method 3: Multilabel Stratified (Iterative Stratification)

Implementation: `_multilabel_stratified_split(...)`

1. Treat each sentence as a multilabel instance using the set of unique non-O labels in that sentence.
2. Build per-label sentence index lists and desired train/eval counts from the split ratio (70/30).
3. Process labels from rarest to most common (fewest unassigned examples first).
4. For each unassigned sentence containing the current label, compute each fold's remaining need across all labels present in that sentence.
5. Assign the sentence to the fold (train or eval) with greater remaining need (with tiny random tie-break noise).
6. Update current per-label counts after each assignment.
7. After label-driven assignment, distribute any remaining unassigned sentences (typically O-only) to satisfy overall train size.
8. Return final train/eval sentence lists.

Method 2 vs Method 3 (direct comparison):

1. Optimization target:
   - Method 2: heuristic train-coverage-oriented label-aware greedy assignment.
   - Method 3: explicit per-label train/eval need balancing during assignment.
2. Label granularity:
   - Method 2: label-aware helper logic, but not explicit multilabel deficit tracking for every step.
   - Method 3: each sentence is treated as a multilabel instance and scored against fold deficits of all its labels.
3. Rare-label handling:
   - Method 2: tends to protect rare labels in train.
   - Method 3: processes rare labels first and tries to preserve proportional presence across both folds.
4. Expected fold behavior:
   - Method 2: usually better than random for coverage, but proportionality can still be uneven.
   - Method 3: generally closer to true multilabel stratification and balanced representation in train and eval.

### What Method 2 Is Missing (and Method 3 Adds)

Method 2 (`_label_aware_split`) is useful, but it is still a generic greedy helper. It does **not** explicitly optimize the full multilabel allocation problem sentence-by-sentence with per-label fold deficits.

What Method 3 adds on top of Method 2:

1. **True multilabel view per sentence**: each sentence is treated as a set of labels and allocated by considering all labels it carries simultaneously.
2. **Rarest-label-first control loop**: assignment order is driven by the rarest still-unassigned labels, which gives minority labels priority during allocation.
3. **Per-label remaining-need objective**: each assignment compares train/eval demand gaps for the sentence's labels and chooses the fold with greater total need.
4. **Closer proportional matching by label**: the algorithm directly targets per-label train/eval proportions, rather than only broad label-coverage heuristics.
5. **More principled behavior for co-occurring labels**: when labels co-appear in the same sentence, Method 3 accounts for their joint effect during assignment.

Why this matters: unlike simple rarity heuristics, this method explicitly targets proportional label representation in both folds and is closer to true multilabel stratification (Sechidis et al., 2011).

### Exp07 Method 4: Multilabel Stratified (Paper-Style Tie-Breaking)

Implementation: `_multilabel_iterative_paper_split(...)`

1. Treat each sentence as a multilabel instance using the set of unique non-O labels in that sentence.
2. Build per-label sentence index lists and desired train/eval counts from the split ratio (70/30).
3. Repeatedly select the rarest still-unassigned label.
4. For candidate sentences containing that label, decide train vs eval with a paper-style priority order:
   - remaining need for the currently selected rare label,
   - then remaining fold capacity,
   - then random tie-break if still tied.
5. Assign the sentence and immediately update fold counts and per-label counts.
6. Continue until label-driven assignment is exhausted.
7. Place any leftover sentences afterward while preserving the overall 70/30 sentence target.
8. Return final train/eval sentence lists.

What is different from Method 3:

1. Method 3 uses the summed remaining need across all labels in the candidate sentence.
2. Method 4 gives first priority to the currently selected rare label instead of the sentence's aggregate cross-label deficit.
3. Method 4 also makes fold capacity an explicit tie-break stage before randomness.
4. So Method 3 is a more blended aggregate-deficit heuristic, while Method 4 is a more paper-faithful rare-label-first assignment rule.

Why this matters: both methods are multilabel-aware, but they resolve ambiguous assignments differently. Method 4 tests whether the more paper-style tie-breaking policy leads to different downstream behavior than Method 3.

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

1. Exp07 variants (4 conditions typically).
2. Exp08 baseline and augmented (2 conditions).
3. Exp07+Aug variants (4 conditions typically, if generated).

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
2. Exp07 contributes deterministic split artifacts for four sentence allocation policies.
3. Exp08 contributes controlled train-only augmentation while preserving eval integrity.
4. Cross comparison standardizes evaluation by forcing all downstream experiments to consume explicit train/eval JSON pairs.
5. The pipeline supports resumable long runs via progress checkpointing.
