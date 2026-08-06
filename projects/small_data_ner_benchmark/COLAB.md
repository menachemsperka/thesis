# Google Colab — Cross-benchmark NER comparison (Exp10)

This guide runs **`run_cross_benchmark_comparison.py`**: regular BERT-CRF vs cascaded + consistency vs SVM fusion on CoNLL-2003 / NEMO / BC5CDR, with **random** and **paper-style multilabel stratified** splits (`small_300` and `full` regimes).

For general Colab setup (clone, Drive, proxies), see also [COLAB_README.md](../../COLAB_README.md) at the repo root.

---

## 1. Mount Drive and open the repo

Run once per Colab session:

```python
from google.colab import drive
drive.mount("/content/drive")
```

If the repo is already on Drive (recommended):

```python
import os

REPO = "/content/drive/MyDrive/thesis_project/thesis_github"  # adjust if needed
os.makedirs(os.path.dirname(REPO), exist_ok=True)
%cd {REPO}
```

If you still need to clone, use a PAT with `getpass` as in [COLAB_README.md](../../COLAB_README.md), then `%cd thesis_github`.

---

## 2. Install dependencies

```python
%cd /content/drive/MyDrive/thesis_project/thesis_github  # same REPO path as above
!pip install -q -r requirements.txt
```

---

## 3. Colab environment (run before any benchmark command)

Disables corporate proxies and enables Colab-friendly Trainer checkpoint behavior:

```python
import os

os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: faster cascaded training while debugging
# os.environ["THESIS_EXP04_EPOCHS"] = "3"
```

Optional — persist Hugging Face dataset cache on Drive (speeds up `--prepare-only` reruns):

```python
CACHE = "/content/drive/MyDrive/thesis_project/benchmark_hf_cache"
os.makedirs(CACHE, exist_ok=True)
```

---

## 4. Prepare datasets and splits (no GPU training)

Downloads corpora and writes `projects/small_data_ner_benchmark/data/*/split_meta.json` and split JSON files.

**All benchmarks:**

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --prepare-only \
  --cache-dir /content/drive/MyDrive/thesis_project/benchmark_hf_cache
```

**Single benchmark (faster smoke test):**

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --prepare-only \
  --benchmarks conll2003_bert \
  --cache-dir /content/drive/MyDrive/thesis_project/benchmark_hf_cache
```

Override NEMO Hub id if needed:

```python
import os
os.environ["THESIS_BENCHMARK_NEMO_DATASET"] = "onlplab/nemo"
```

---

## 5. Dry-run (plan jobs, no training)

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --dry-run \
  --benchmarks conll2003_bert \
  --regimes small_300 \
  --num-seeds 2
```

---

## 6. Run the comparison (GPU)

Enable **Runtime → Change runtime type → GPU** before this cell.

**Recommended smoke test** (one corpus, small regime, 2 seeds):

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --benchmarks conll2003_bert \
  --regimes small_300 \
  --num-seeds 2 \
  --base-mode auto \
  --cache-dir /content/drive/MyDrive/thesis_project/benchmark_hf_cache
```

**Full matrix** (3 benchmarks × 2 regimes × 2 split variants × 5 default seeds × 3 experiments — long):

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --base-mode auto \
  --cache-dir /content/drive/MyDrive/thesis_project/benchmark_hf_cache
```

**Subset examples:**

```python
# CoNLL + PubMed only, both regimes, 5 seeds
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --benchmarks conll2003_bert,bc5cdr_pubmedbert \
  --regimes small_300,full \
  --num-seeds 5 \
  --base-mode auto

# NEMO + DictaBERT, small data only
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --benchmarks nemo_dictabert \
  --regimes small_300 \
  --num-seeds 5 \
  --base-mode auto
```

---

## 7. Resume after disconnect

Checkpoint and outputs live under the repo (on Drive if you `%cd` there):

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --resume \
  --base-mode auto \
  --cache-dir /content/drive/MyDrive/thesis_project/benchmark_hf_cache
```

Reuse trained Exp10 regular+cascade pairs without retraining (fusion-only or re-export):

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --resume \
  --base-mode reuse \
  --experiments 10_svm_ready
```

Force retrain all CRF bases:

```python
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --base-mode retrain \
  --benchmarks conll2003_bert \
  --regimes small_300 \
  --num-seeds 2
```

---

## 8. Where to find results

Under the repo (paths relative to repo root):

| Artifact | Location |
|----------|----------|
| Main Excel / JSON | `projects/small_data_ner_benchmark/outputs/cross_comparison/cross_comparison_latest.xlsx` |
| Exp10 error analysis | `projects/small_data_ner_benchmark/outputs/cross_comparison/consolidated_error_analysis_exp10_latest.xlsx` |
| Resume checkpoint | `projects/small_data_ner_benchmark/outputs/cross_comparison/benchmark_cross_comparison_checkpoint.json` |
| Per-run metrics (thesis layout) | `outputs/exp10_regular/`, `outputs/exp10_cascade/`, `outputs/exp10_svm_ready/` |

List outputs from a notebook cell:

```python
!ls -la projects/small_data_ner_benchmark/outputs/cross_comparison/
```

Download the latest workbook in Colab:

```python
from google.colab import files
files.download("projects/small_data_ner_benchmark/outputs/cross_comparison/cross_comparison_latest.xlsx")
```

---

## 9. One-shot notebook sequence (copy-paste)

Minimal end-to-end after the repo is on Drive:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"

REPO = "/content/drive/MyDrive/thesis_project/thesis_github"
CACHE = "/content/drive/MyDrive/thesis_project/benchmark_hf_cache"
%cd {REPO}

!pip install -q -r requirements.txt

!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --prepare-only --benchmarks conll2003_bert --cache-dir {CACHE}

!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --benchmarks conll2003_bert --regimes small_300 --num-seeds 2 \
  --base-mode auto --cache-dir {CACHE}
```

---

## CLI reference (same runner as local)

| Flag | Example |
|------|---------|
| `--benchmarks` | `conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert` |
| `--regimes` | `small_300,full` |
| `--experiments` | `10_regular,10_cascade,10_svm_ready` (default) |
| `--num-seeds` | `5` (seeds 42–46 by default; use `--seed-start`) |
| `--base-mode` | `auto` \| `reuse` \| `retrain` |
| `--prepare-only` | Download + splits only |
| `--dry-run` | Print planned conditions |
| `--resume` | Continue from checkpoint |

Entry point (equivalent):

```python
!python projects/small_data_ner_benchmark/run_benchmark.py --help
```
