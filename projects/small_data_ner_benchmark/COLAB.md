# Google Colab — Cross-benchmark NER comparison (Exp10)

Runs **`run_cross_benchmark_comparison.py`**: regular BERT-CRF vs cascaded + consistency vs SVM fusion on CoNLL-2003 / NEMO / BC5CDR, with **random** and **paper-style multilabel stratified** splits.

**Regimes:** **`small_300` only** (300-sentence pool, 70/30 per seed × 2 split variants). Omit `full` unless you explicitly add `--regimes small_300,full`.

General clone/Drive setup: [COLAB_README.md](../../COLAB_README.md).

---

## 0a. Sync code from GitHub (run if files are missing on Drive)

Your working copy on Drive is often **`/content/drive/MyDrive/thesis_project/thesis`** (GitHub repo `menachemsperka/thesis`).  
The benchmark needs **`projects/small_data_ner_benchmark/corpus_loaders.py`** — that file is on **`main`** on GitHub; if `ls` says missing, **`git pull` did not update that folder** (wrong directory, not a git repo, or pull failed).

**Security:** never paste a GitHub PAT into a notebook or chat. Use [Colab Secrets](https://colab.research.google.com/) (`userdata.get('GITHUB_TOKEN')`) or `getpass`, and revoke any token that was exposed.

```python
import os
from pathlib import Path

REPO = "/content/drive/MyDrive/thesis_project/thesis"  # your Drive clone
assert Path(REPO).is_dir(), f"Missing folder: {REPO}"

%cd {REPO}

# Must be a git checkout (should print a path under REPO)
!git rev-parse --show-toplevel

!git fetch origin
!git status -sb
!git pull origin main

# Benchmark package (all of these should exist after a successful pull)
!ls -la projects/small_data_ner_benchmark/corpus_loaders.py
!ls -la projects/small_data_ner_benchmark/split_stats.py
!git log -1 --oneline -- projects/small_data_ner_benchmark/corpus_loaders.py
```

If **`git rev-parse` fails**, Drive is not a git clone — clone fresh next to it or replace the folder:

```python
# Optional: fresh clone (uses secret GITHUB_TOKEN — add in Colab Secrets, not in code)
import os
from google.colab import userdata

PARENT = "/content/drive/MyDrive/thesis_project"
%cd {PARENT}
token = userdata.get("GITHUB_TOKEN")  # create secret in Colab
!git clone https://{token}@github.com/menachemsperka/thesis.git thesis_fresh
# Then set REPO = f"{PARENT}/thesis_fresh" in section 0
```

If **`git pull` fails** because of local edits on Drive:

```python
%cd /content/drive/MyDrive/thesis_project/thesis
!git stash push -m "colab-drive-stash"
!git pull origin main
```

On your **PC** (folder `thesis_github`), after you commit benchmark changes, run **`git push origin main`** so Colab can pull them. Check push status:

```bash
git status -sb
git push origin main
```

---

## 0. Session setup (run once)

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import os

REPO = "/content/drive/MyDrive/thesis_project/thesis"  # or .../thesis_github if that is your clone
CACHE = "/content/drive/MyDrive/thesis_project/benchmark_hf_cache"
RUNNER = "projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py"

os.makedirs(CACHE, exist_ok=True)
os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
# Required for English/Latin benchmark CSVs (runner also sets this on import).
os.environ["THESIS_SKIP_HEBREW_TEXT_VALIDATION"] = "1"

%cd {REPO}
```

```python
!pip install -q -r requirements.txt
```

Enable **Runtime → Change runtime type → T4 GPU** before **Run**. If you see `Could not find cuda drivers`, the runtime is still CPU-only — change runtime and restart, then re-run setup + prepare/run.

Optional NEMO dataset override (run **before** prepare **1b — NEMO** only):

```python
import os
os.environ["THESIS_BENCHMARK_NEMO_DATASET"] = "onlplab/nemo"
```

Optional faster cascaded epochs while debugging:

```python
import os
os.environ["THESIS_EXP04_EPOCHS"] = "3"
```

---

## 1. Prepare-only (one cell per benchmark)

**small_300 + full**, **20 seeds** (42–61). Steps **1a–1c** download corpora + baseline splits (CPU). Step **1d** runs **exp08 LLM mask-fill** on each train split (GPU strongly recommended; slow).

### 1a. CoNLL-2003 + bert-base-uncased

```python
!python $RUNNER \
  --prepare-only \
  --skip-augmentation \
  --benchmarks conll2003_bert \
  --regimes small_300 \
  --num-seeds 20 \
  --cache-dir $CACHE
```

### 1b. NEMO + DictaBERT

Set `THESIS_BENCHMARK_NEMO_DATASET` above if needed, then:

```python
!python $RUNNER \
  --prepare-only \
  --skip-augmentation \
  --benchmarks nemo_dictabert \
  --regimes small_300 \
  --num-seeds 20 \
  --cache-dir $CACHE
```

### 1c. BC5CDR + PubMedBERT

```python
!python $RUNNER \
  --prepare-only \
  --skip-augmentation \
  --benchmarks bc5cdr_pubmedbert \
  --regimes small_300 \
  --num-seeds 20 \
  --cache-dir $CACHE
```

Check baseline splits:

```python
!ls -la projects/small_data_ner_benchmark/data/*/split_meta.json
```

### 1d. LLM augmentation (exp08, per benchmark)

Uses each benchmark’s encoder for fill-mask (`bert-base-uncased`, DictaBERT, PubMedBERT). Eval splits are **unchanged**; only train JSON is extended. Optional: `os.environ["THESIS_EXP08_MULTIPLIER"] = "3"`.

```python
!python $RUNNER \
  --prepare-augmentation-only \
  --benchmarks conll2003_bert \
  --regimes small_300 \
  --num-seeds 20 \
  --cache-dir $CACHE
```

Repeat for `nemo_dictabert` and `bc5cdr_pubmedbert`. Augmented files: `data/<benchmark>/splits/<regime>/*_augmented_train.json`.

```python
!ls projects/small_data_ner_benchmark/data/conll2003_bert/splits/small_300/*augmented*
```

---

## 2. Run (train or resume)

Default **`--train-modes baseline,augmented`**: trains each split twice (original train vs exp08-augmented train, same eval). Baseline-only: add **`--skip-augmentation`**. Requires **1d** completed if using augmented mode.

```python
!python $RUNNER \
  --benchmarks conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert \
  --regimes small_300 \
  --num-seeds 20 \
  --train-modes baseline,augmented \
  --experiments 10_regular,10_cascade,10_svm_ready \
  --base-mode auto \
  --cache-dir $CACHE \
  --resume
```

Colab expands **`$RUNNER`** and **`$CACHE`** from the setup cell.

### Restart training

**Continue** after a disconnect (skip runs already in the checkpoint):

```python
!python $RUNNER \
  --benchmarks conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert \
  --regimes small_300 \
  --num-seeds 20 \
  --train-modes baseline,augmented \
  --experiments 10_regular,10_cascade,10_svm_ready \
  --base-mode auto \
  --cache-dir $CACHE \
  --resume
``` (keep splits on Drive; drop progress only):

```python
!rm -f projects/small_data_ner_benchmark/outputs/cross_comparison/benchmark_cross_comparison_checkpoint.json
```

Then run the same **`--resume`** command again (first run after delete trains from run 1).

**Change seed count (e.g. 5 → 20)** — delete prepared splits for each benchmark (or remove `split_meta.json`), re-run prepare **1a–1c** with `--num-seeds 20`, delete the checkpoint, then run §2.

```python
# Example: wipe one benchmark's splits to re-prepare
!rm -f projects/small_data_ner_benchmark/data/conll2003_bert/split_meta.json
!rm -rf projects/small_data_ner_benchmark/data/conll2003_bert/splits
```

### Do you need `--prepare-only` again?

**No**, if everything below is still on Drive and you use the **same** benchmarks, **`--regimes small_300`**, **`--num-seeds 20`**, and `--pool-seed` (default 42):

| What | Where (on Drive, under repo) | Re-run prepare? |
|------|------------------------------|-----------------|
| Splits + `corpus.csv` | `projects/small_data_ner_benchmark/data/<benchmark>/` (`split_meta.json`, `splits/`, `pool_300.json`, …) | Only if you **delete** that folder or change seeds/regimes/benchmark list |
| Hugging Face download cache | `$CACHE` (e.g. `benchmark_hf_cache/`) | Only if cache was cleared; training **does not** re-download if cache exists |

On **Run**, the script skips prepare when `split_meta.json` already lists the **same seeds and regimes** you pass on the CLI. If you increase `--num-seeds` (e.g. 5 → 20), re-run **prepare 1a–1c** (the runner will **re-prepare automatically** on the next full run if you skip prepare-only, but prepare-only cells skip only when seeds match — run one prepare cell or delete `split_meta.json` first).

### Prepare-only: resume?

There is **no** mid-download checkpoint (no `--resume` for prepare). If a run **crashes** while preparing one benchmark, that benchmark may be incomplete until you re-run prepare.

**Re-run the same prepare cell** (1a, 1b, or 1c) after a failure — only that benchmark runs again; the others stay skipped if already complete.

`--resume` applies only to **training** (section 2), not to prepare.

### How long does prepare-only take?

Rough Colab order-of-magnitude (first time, decent network; **CPU runtime is fine**):

| Phase | Time |
|-------|------|
| CoNLL-2003 | ~1–5 min download + &lt;1 min splits |
| BC5CDR | ~2–10 min download + &lt;1 min splits |
| NEMO | ~5–30 min (corpus size / Hub speed varies) |
| **All 3 benchmarks** | ~**10–45 min** first time; often **&lt;5 min** if `$CACHE` + `data/*/` already on Drive |

Split generation itself (2 variants × 20 seeds × 2 regimes per benchmark) is seconds to a couple of minutes per benchmark on CPU.

**Re-run `--prepare-only`** when you change `--num-seeds`, `--regimes`, `--benchmarks`, or `--pool-seed`, or after deleting `data/` or the HF cache.

Keep **`%cd`** pointed at the Drive copy of the repo (`REPO` above). A clone only under `/content/` is wiped when the runtime disconnects.

---

## Results on Drive

| Artifact | Path (under repo root) |
|----------|-------------------------|
| Main Excel | `projects/small_data_ner_benchmark/outputs/cross_comparison/cross_comparison_latest.xlsx` (sheet **`dataset_details`**: per seed train/eval sentences, tokens, entities) |
| Dataset details JSON | `projects/small_data_ner_benchmark/outputs/cross_comparison/dataset_details_latest.json` |
| Exp10 error analysis | `projects/small_data_ner_benchmark/outputs/cross_comparison/consolidated_error_analysis_exp10_latest.xlsx` |
| Checkpoint | `projects/small_data_ner_benchmark/outputs/cross_comparison/benchmark_cross_comparison_checkpoint.json` |

```python
!ls -la projects/small_data_ner_benchmark/outputs/cross_comparison/
```

```python
from google.colab import files
files.download("projects/small_data_ner_benchmark/outputs/cross_comparison/cross_comparison_latest.xlsx")
```

---

## CLI flags (reference)

| Flag | Meaning |
|------|---------|
| `--prepare-only` | Baseline corpora + splits (use `--skip-augmentation` for 1a–1c only) |
| `--prepare-augmentation-only` | Exp08 LLM mask-fill on existing train splits (§1d) |
| `--train-modes` | `baseline,augmented` (default) or `baseline` with `--skip-augmentation` |
| `--force-augmentation` | Rebuild `*_augmented_train.json` files |
| `--benchmarks` | `conll2003_bert`, `nemo_dictabert`, `bc5cdr_pubmedbert` |
| `--regimes` | Default in this doc: `small_300` only (add `,full` if needed) |
| `--num-seeds` | `20` → seeds 42–61 (default `--seed-start` 42) |
| `--experiments` | Default `10_regular,10_cascade,10_svm_ready` |
| `--base-mode` | `auto` \| `reuse` \| `retrain` |
| `--cache-dir` | Hugging Face cache on Drive (`$CACHE`) |
| `--resume` | Train if no checkpoint; otherwise skip finished runs |

Equivalent entry point: `projects/small_data_ner_benchmark/run_benchmark.py`.

---

## Troubleshooting

**Resume: 360 completed, then only re-exports Excel (no `Run 361/720 … +aug`)**  
The training loop must use the same **`--train-modes`** as the run plan (fixed in current repo). **Git pull** on Drive, then confirm augmented files exist and the startup line shows augmented conditions:

```python
!ls projects/small_data_ner_benchmark/data/*/splits/small_300/*augmented* | head
```

After pull, `--resume` should log e.g. `Run plan: … (120 baseline, 120 augmented) × 3 = 720 runs` and train **`+aug`** rows. If you see `0 augmented`, run **`--prepare-augmentation-only`** first.

**`prepared seeds [42…46] do not include all requested [42…61]`**  
Splits were built with **`--num-seeds 5`**; training uses **20**. Re-run prepare for each benchmark (1a–1c) with `--num-seeds 20`, or run §2 once — it will **re-prepare** benchmarks whose `split_meta.json` seed list does not match. Then **`--resume`** training.

```python
!python $RUNNER --prepare-only --benchmarks conll2003_bert --regimes small_300,full --num-seeds 20 --cache-dir $CACHE
# repeat for nemo_dictabert, bc5cdr_pubmedbert
```

**`ModuleNotFoundError: No module named 'corpus_loaders'`**  
Sync the full `projects/small_data_ner_benchmark/` folder to Drive (must include `corpus_loaders.py`). Then re-run §0 and your prepare/run cell. Quick check:

```python
!ls projects/small_data_ner_benchmark/corpus_loaders.py
```

**`HebrewCorpusEncodingError` … Sample tokens: `['EU', 'rejects', …]`**  
The main thesis loader assumes a Hebrew corpus. Benchmark runs must use an updated repo where `core/hebrew_text_io.py` honors `THESIS_SKIP_HEBREW_TEXT_VALIDATION`, and Colab setup sets that variable (see §0). Sync your Drive copy from git, re-run §0, then `--resume`.

**Run 61+ (NEMO) starts training then hangs or is slow on seed**  
Use a **GPU** runtime (§0). The stack trace through `torch.xpu.manual_seed_all` is often Colab CPU/XPU noise; real training needs CUDA.

**180 runs all show `F1=N/A` in ~2s**  
Usually the Hebrew CSV error above; fix that before expecting checkpoints or meaningful `--resume` progress.
