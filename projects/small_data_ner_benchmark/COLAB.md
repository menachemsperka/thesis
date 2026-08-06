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

Optional NEMO dataset override (set **before** prepare/run):

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

## 1. Prepare-only

All three benchmarks, **small_300 + full**, 5 seeds (42–46). Downloads corpora and writes split JSON (no training).

```python
!python $RUNNER \
  --prepare-only \
  --benchmarks conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert \
  --regimes small_300,full \
  --num-seeds 5 \
  --cache-dir $CACHE
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
