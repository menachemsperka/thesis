# Google Colab — Cross-benchmark NER comparison (Exp10)

Runs **`run_cross_benchmark_comparison.py`**: regular BERT-CRF vs cascaded + consistency vs SVM fusion on CoNLL-2003 / NEMO / BC5CDR, with **random** and **paper-style multilabel stratified** splits.

**Regimes (always both):** `small_300` (300-sentence pool) and `full` (official train pool), 70/30 per split variant and seed.

General clone/Drive setup: [COLAB_README.md](../../COLAB_README.md).

---

## 0. Session setup (run once)

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import os

REPO = "/content/drive/MyDrive/thesis_project/thesis_github"  # change if needed
CACHE = "/content/drive/MyDrive/thesis_project/benchmark_hf_cache"
RUNNER = "projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py"

os.makedirs(CACHE, exist_ok=True)
os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

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

**small_300 + full**, 5 seeds (42–46). CPU is fine. Run **0 → setup**, then **1a → 1b → 1c** in order (or retry only the cell that failed).

If a benchmark already has `data/<benchmark>/split_meta.json` on Drive, that cell is **skipped** automatically.

### 1a. CoNLL-2003 + bert-base-uncased

```python
!python $RUNNER \
  --prepare-only \
  --benchmarks conll2003_bert \
  --regimes small_300,full \
  --num-seeds 5 \
  --cache-dir $CACHE
```

### 1b. NEMO + DictaBERT

Set `THESIS_BENCHMARK_NEMO_DATASET` above if needed, then:

```python
!python $RUNNER \
  --prepare-only \
  --benchmarks nemo_dictabert \
  --regimes small_300,full \
  --num-seeds 5 \
  --cache-dir $CACHE
```

### 1c. BC5CDR + PubMedBERT

```python
!python $RUNNER \
  --prepare-only \
  --benchmarks bc5cdr_pubmedbert \
  --regimes small_300,full \
  --num-seeds 5 \
  --cache-dir $CACHE
```

Check all three are ready (each folder should contain `split_meta.json`):

```python
!ls -la projects/small_data_ner_benchmark/data/*/split_meta.json
```

---

## 2. Run (train or resume)

Same scope as prepare. Always use **`--resume`**: first execution trains; re-run this cell after a disconnect to continue.

```python
!python $RUNNER \
  --benchmarks conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert \
  --regimes small_300,full \
  --num-seeds 5 \
  --experiments 10_regular,10_cascade,10_svm_ready \
  --base-mode auto \
  --cache-dir $CACHE \
  --resume
```

Colab expands **`$RUNNER`** and **`$CACHE`** from the setup cell.

### Do you need `--prepare-only` again?

**No**, if everything below is still on Drive and you use the **same** benchmarks, `--regimes small_300,full`, `--num-seeds 5`, and `--pool-seed` (default 42):

| What | Where (on Drive, under repo) | Re-run prepare? |
|------|------------------------------|-----------------|
| Splits + `corpus.csv` | `projects/small_data_ner_benchmark/data/<benchmark>/` (`split_meta.json`, `splits/`, `pool_300.json`, …) | Only if you **delete** that folder or change seeds/regimes/benchmark list |
| Hugging Face download cache | `$CACHE` (e.g. `benchmark_hf_cache/`) | Only if cache was cleared; training **does not** re-download if cache exists |

On **Run**, the script skips prepare when `data/<benchmark>/split_meta.json` already exists.

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

Split generation itself (2 variants × 5 seeds × 2 regimes per benchmark) is seconds to a couple of minutes per benchmark on CPU.

**Re-run `--prepare-only`** when you change `--num-seeds`, `--regimes`, `--benchmarks`, or `--pool-seed`, or after deleting `data/` or the HF cache.

Keep **`%cd`** pointed at the Drive copy of the repo (`REPO` above). A clone only under `/content/` is wiped when the runtime disconnects.

---

## Results on Drive

| Artifact | Path (under repo root) |
|----------|-------------------------|
| Main Excel | `projects/small_data_ner_benchmark/outputs/cross_comparison/cross_comparison_latest.xlsx` |
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
| `--prepare-only` | Datasets + splits only |
| `--benchmarks` | `conll2003_bert`, `nemo_dictabert`, `bc5cdr_pubmedbert` |
| `--regimes` | Use `small_300,full` |
| `--num-seeds` | `5` → seeds 42–46 (default `--seed-start` 42) |
| `--experiments` | Default `10_regular,10_cascade,10_svm_ready` |
| `--base-mode` | `auto` \| `reuse` \| `retrain` |
| `--cache-dir` | Hugging Face cache on Drive (`$CACHE`) |
| `--resume` | Train if no checkpoint; otherwise skip finished runs |

Equivalent entry point: `projects/small_data_ner_benchmark/run_benchmark.py`.
