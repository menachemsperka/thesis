# Experiment 08 — LLM Mask-Filling Augmentation for Rare Labels

## What This Experiment Does

This experiment tackles a common problem in NER (Named Entity Recognition):
**label imbalance**.  Some entity types (e.g. "Person", "Location") appear in
many sentences, while others (e.g. "Book", "Ceremony") appear in very few.
A model trained on such data may never learn to recognise the rare types.

**Solution:** Generate *new* training sentences for the rare entity types using
the same DictaBERT language model that we already use for NER — specifically
its **masked language model** (fill-mask) capability.

### Key Principles

| Principle | Detail |
|-----------|--------|
| **Baseline split** | Same label-aware 70 / 30 split as Experiment 01 (`tf.split_list` with `ensure_label_coverage=True`). |
| **Test data untouched** | The 30 % test set is *never* modified — all augmentation happens only in the training set. |
| **Same model for NER & augmentation** | DictaBERT serves double duty: fine-tuned for NER *and* used for fill-mask generation. No extra model needed. |
| **Multi-seed evaluation** | Results are averaged over 5 random seeds to ensure statistical robustness. |
| **Datasets saved for reuse** | The augmented training set is saved as JSON so experiments 03–06 can be re-run with it. |

---

## How It Works — Step by Step

### 1. Data Loading & Splitting

The Hebrew NER dataset (`ner_dataset.csv`) is loaded and converted to
sentence-level dicts (`{"text": "...", "labels": [...]}`).  The data is split
70 / 30 using the label-aware greedy algorithm from experiment 01 that tries
to preserve the distribution of non-O labels across both splits.

### 2. Identifying Rare Labels

For the training portion, we compute statistics per entity type:

- **Instance count** — how many tokens carry that label.
- **Sentence count** — how many distinct sentences contain at least one token
  of that label.
- **Delta to max** — difference between this label's sentence count and the
  most frequent label's sentence count.

Labels with a positive delta (fewer sentences than the most frequent label) are
candidates for augmentation.

### 3. Advanced Sentence Scoring (Inspired by Experiment 07)

Not all source sentences are equally useful for augmentation.  We use an
**inverse-frequency log-dampened scoring** approach borrowed from the split
strategies in experiment 07:

```
score(sentence) = sum over each non-O label L:
    log(1 + max_label_count / count(L))
```

This means:
- Sentences containing **very rare** labels score highest.
- Sentences containing **only common** labels score lowest.
- The **log dampening** prevents a single extremely rare label from
  dominating all generation budget.

Higher-scored sentences get a proportionally larger share of the generation
budget, so we produce more variants where they matter most.

### 4. Generation Multiplier (×3 default)

The raw "delta to max" tells us the minimum deficit.  We multiply it by a
configurable factor (default **×3**) to generate substantially more variants.
This ensures the model sees enough diverse examples of rare entities.

### 5. Three Complementary Generation Strategies

For each (sentence, budget) pair, we apply up to three methods:

#### Strategy A: Multi-Position Mask-Filling

Unlike v1 (which only masked the first entity token), we now mask **each
entity position independently**:

1. Take sentence: `"בספר תורה_ומצוה נאמר כי"`
   with labels: `["O", "B-BOK", "O", "O"]`
2. For each position tagged B-BOK or I-BOK:
   - Replace that token with `[MASK]`
   - Run DictaBERT fill-mask, constrained to known entity tokens
   - Each prediction creates a new sentence variant
3. Deduplicate across positions to avoid exact repeated variants.

This generates more diverse variants because each entity position has
different surrounding context, leading to different predictions.

#### Strategy B: Context-Preserving Duplication

For **extremely rare** labels (sentence count ≤ Q1 threshold, bottom 25 %):
- If fill-mask could not produce enough variants to meet the budget,
  duplicate the original sentence to fill the shortfall.
- This ensures the model gets enough exposure to these labels even when
  the entity vocabulary is too small for diverse mask-filling.

#### Strategy C: Budget Overflow Handling

After processing, we shuffle all generated sentences randomly to prevent
ordering bias during training.

### 6. Training & Evaluation

Two models are trained for each seed:

| Condition | Training Data |
|-----------|--------------|
| **Baseline** | Original training sentences only |
| **Augmented** | Original + all generated sentences |

Both are evaluated on the **same unmodified test set**.

### 7. Saving Datasets for Reuse

After training, the first seed's splits are saved to `outputs/exp08/splits/`:

```
outputs/exp08/splits/
  ├── baseline_train.json      (original 181 sentences)
  ├── baseline_eval.json       (78 sentences — never modified)
  ├── augmented_train.json     (181 + ~1800 generated sentences)
  ├── augmented_eval.json      (same 78 sentences)
  └── split_meta.json          (metadata for downstream use)
```

These files can then be consumed by experiments 03–06 via the
`run_experiments_with_exp08_data.py` runner script.

---

## Why This Approach Works Well

### Advantages over v1

| Improvement | v1 (old) | v2 (current) |
|-------------|----------|--------------|
| **Generation volume** | 1× deficit | 3× deficit (configurable multiplier) |
| **Masking positions** | First entity token only | Every entity position independently |
| **Sentence selection** | Balance score (tf function) | Inverse-freq log-dampened scoring |
| **Extremely rare labels** | No special handling | Context-preserving duplication as fallback |
| **Dataset persistence** | Not saved | Saved as JSON for exp03–06 reuse |

### Why Mask-Filling Is a Good Approach

1. **Contextual realism** — The fill-mask model considers surrounding words
   when predicting the masked entity.  Generated sentences are grammatically
   and contextually appropriate.

2. **No extra model** — DictaBERT is already downloaded and used for NER.
   Re-using it for augmentation avoids dependency on external services.

3. **Constrained vocabulary** — Predictions are restricted to tokens actually
   seen in the training data, preventing hallucinated entities.

4. **Label preservation** — Only entity tokens change; BIO structure stays valid.

5. **Targeted augmentation** — Only rare labels are augmented; common labels
   are left alone.

### Alternative Approaches Considered

| Method | Pros | Cons |
|--------|------|------|
| **Simple duplication** | Fast, trivial | No diversity — risk of overfitting |
| **Synonym replacement** | Fast | Requires Hebrew synonym dictionary for entities |
| **Back-translation** | Diverse paraphrases | Expensive; may break BIO alignment |
| **Generative LLM** (GPT) | Novel sentences | Requires API; expensive; hard to control labels |
| **Entity substitution from KB** | High diversity | Requires Hebrew entity KB per type |
| **Mask-filling (chosen)** | Contextual, uses existing model, label-safe | Bounded by entity vocabulary size |

---

## Running Experiments 03–06 with Augmented Data

Run:

```bash
python run_experiments_with_exp08_data.py
```

The runner now behaves as follows:

- If `outputs/exp08/splits/` already contains valid split files, it uses them directly.
- If split files are missing or invalid, it automatically runs
   `experiments/experiment_08_llm_augmentation.py` first, then runs exp03–06.

This runs each of experiments 03, 04, 05, 06 twice:
1. With **baseline** training data (no augmentation)
2. With **augmented** training data (original + generated)

Both use the same eval set.  Results are saved to `outputs/exp08_comparison/`.

### What It Produces

| File | Contents |
|------|----------|
| `exp08_comparison_*.xlsx` | Excel with summary, all runs, and delta sheets |
| `exp08_comparison_*.json` | Machine-readable full results |

### Customization

```bash
# Run only specific experiments
set THESIS_EXP08_RUN_EXPERIMENTS=03,04
python run_experiments_with_exp08_data.py

# Force rerun experiment 08 before exp03-06 (CLI option)
python run_experiments_with_exp08_data.py --force-exp08

# Force rerun experiment 08 before exp03-06 (environment variable)
set THESIS_EXP08_FORCE_RERUN=1
python run_experiments_with_exp08_data.py
```

---

## Output Files

All outputs are saved in `outputs/exp08/`:

| File | Contents |
|------|----------|
| `llm_augmentation_*.xlsx` | Excel workbook (see sheets below) |
| `llm_augmentation_*.json` | Machine-readable results dict |
| `llm_augmentation_per_seed_*.csv` | Per-seed metrics for each condition |
| `llm_augmentation_metric_stats_*.csv` | Aggregated mean/std per condition |
| `llm_augmentation_thesis_summary_*.csv` | Thesis-ready summary table |
| `llm_augmentation_label_count_*.csv` | Label distribution before/after augmentation |
| `splits/` | Saved train/eval JSON files for reuse |

### Excel Sheets

| Sheet | What it shows |
|-------|--------------|
| **metrics** | Thesis summary: Condition, F1 mean±std, Precision, Recall, Accuracy |
| **detailed_results** | Per-seed scores for baseline and augmented conditions |
| **metric_stats** | Numeric mean, std, and delta for every metric |
| **score_summary_numeric** | Same as metric_stats plus 95 % CI for F1 |
| **label_count_comparison** | Side-by-side: token count and sentence count per label, before and after augmentation |
| **generation_log** | What was generated: label, source, rarity score, budget, mask-filled count, duplication count |
| **documentation** | Key–value metadata for academic citation |

---

## How to Run

```bash
# Default: 5 seeds, multiplier ×3
python experiments/experiment_08_llm_augmentation.py

# Then run exp03-06 with the augmented data
python run_experiments_with_exp08_data.py

# Force rerun exp08 before running exp03-06
python run_experiments_with_exp08_data.py --force-exp08

# Custom seed, count, and multiplier
set THESIS_SPLIT_SEED=100
set THESIS_EXP08_NUM_SEEDS=3
set THESIS_EXP08_MULTIPLIER=5
python experiments/experiment_08_llm_augmentation.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THESIS_SPLIT_SEED` | `42` | Base random seed for reproducible splits |
| `THESIS_EXP08_NUM_SEEDS` | `5` | Number of seeds to evaluate (minimum 2) |
| `THESIS_EXP08_MULTIPLIER` | `3` | Generation multiplier (how many × the deficit) |
| `THESIS_NER_CSV` | `data/ner_dataset.csv` | Override dataset path |
| `THESIS_DEBUG` | `0` | Set to `1` for verbose console output |
| `THESIS_MODEL_NAME` | auto-detected | Force a specific model path or HuggingFace ID |
| `THESIS_EXP08_RUN_EXPERIMENTS` | `03,04,05,06` | Which experiments to run with augmented data |
| `THESIS_EXP08_FORCE_RERUN` | `0` | If `1/true/yes/on`, rerun experiment 08 before running exp03–06 even when saved splits exist |
