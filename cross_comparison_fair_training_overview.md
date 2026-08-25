# Cross-model fair training overview

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

Exp05 / Exp06 “ready” and fusion steps **reuse** Exp01/Exp04 artifacts; they do not change base training hyperparameters.

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
| Disable fp16 | `export THESIS_TRAINER_FP16=0` |

If you change the profile mid-study, clear or retrain affected checkpoint rows and update the thesis methods section to match `run_manifest.json` → `training_hyperparameters`.

---

## Related files

| File | Role |
|------|------|
| `core/training_defaults.py` | Profile constants, `apply_fair_comparison_training_defaults()`, manifest snapshot |
| `core/th_functions.py` | Exp01 `TrainingArguments` |
| `core/auc_cascaded_pipeline.py` | Exp04 `TRAINING_CONFIG` + env |
| `run_cross_data_model_comparison.py` | Applies defaults at run start, writes manifest |
| `theisis overview.md` §1.1 | Pipeline commands and pointer to this doc |

---

## Checklist before writing up results

- [ ] `run_manifest.json` contains `training_hyperparameters` for this output dir.
- [ ] All models in the comparison table were trained with the same manifest profile (or you document explicit overrides).
- [ ] No multilingual-only epoch / LR / class-weight shortcuts in the code revision used for the run (`git rev-parse HEAD` on Colab).
- [ ] Subset and seeds documented: `--subset-sentences`, `--subset-seed`, `--num-seeds`, `--condition-sources`.
