# Small-data / cross-domain NER benchmark

Compares **Exp10** methods on public corpora using the **same split logic as the thesis paper** (Experiment 07):

| Split variant | exp07 key | Description |
|---------------|-----------|-------------|
| Simple random | `before_exp01_baseline` | Uniform shuffle, 70/30 |
| Multilabel stratified (paper-style) | `after_multilabel_iterative_paper` | Rarest-label-first iterative stratification |

**Regimes**

- `small_300` — fixed pool of **300** sentences sampled from official train (`pool_seed=42`); each seed applies both split variants on that pool (≈210 train / 90 eval).
- `full` — pool = **all official train** sentences; same 70/30 + variants per seed.

**Experiments:** `10_regular`, `10_cascade`, `10_svm_ready` (same as `run_cross_data_model_comparison.py` Exp10 set).

## Entry point

From repo root:

```bash
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --prepare-only
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --dry-run
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --resume --base-mode auto
python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py --resume --base-mode reuse
```

`run_benchmark.py` is an alias for the same runner.

### CLI (aligned with main cross-comparison)

| Flag | Purpose |
|------|---------|
| `--benchmarks` | `conll2003_bert,nemo_dictabert,bc5cdr_pubmedbert` (default: all) |
| `--regimes` | `small_300,full` |
| `--num-seeds` / `--seeds` | Paired seeds (default **20**, seeds 42–61) |
| `--base-mode` | `auto` / `reuse` / `retrain` for Exp10 CRF base cache |
| `--resume` | Train from scratch if no checkpoint; otherwise continue (same flags every run) |
| `--prepare-only` | Download HF data + write splits only |

## Outputs (same layout as main runner)

Under `projects/small_data_ner_benchmark/outputs/cross_comparison/`:

- `cross_comparison_<timestamp>.xlsx` + `cross_comparison_latest.xlsx`
- `cross_comparison_<timestamp>.json` + `cross_comparison_latest.json`
- `consolidated_error_analysis_exp10_<timestamp>.xlsx` (+ `_latest`) — merged token-level error sheets from every successful Exp10 run
- `benchmark_cross_comparison_checkpoint.json` — resume state
- `cross_comparison_base_crf_ready_index.json` — Exp10 regular+cascade cache index

Training artifacts still land under repo `outputs/exp10_*` (same as thesis); row `metrics_file` paths feed consolidation.

Excel sheets: `summary_pivot`, `all_runs`, `deltas_split_variants` (paper − random per seed), `paired_summary`, `documentation`, `exp10_error_analysis` (pointer sheet).

## Google Colab

**Step-by-step notebook cells and commands:** [COLAB.md](COLAB.md)

Set environment **before** importing/running (same as [COLAB_README.md](../../COLAB_README.md)):

```python
import os
os.environ["THESIS_RUN_ENV"] = "colab"
os.environ["WANDB_DISABLED"] = "true"
```

Then from the cloned repo root:

```python
%cd thesis_github
!python projects/small_data_ner_benchmark/run_cross_benchmark_comparison.py \
  --benchmarks conll2003_bert --regimes small_300 --num-seeds 2
```

Optional: mount Drive and set `THESIS_BENCHMARK_CACHE` via `--cache-dir /content/drive/MyDrive/thesis/hf_cache`.

## Benchmarks

| Key | Dataset | Model |
|-----|---------|--------|
| `conll2003_bert` | CoNLL-2003 | `bert-base-uncased` |
| `nemo_dictabert` | `onlplab/nemo` (fallback `imvladikon/nemo_corpus`) | DictaBERT |
| `bc5cdr_pubmedbert` | BC5CDR | PubMedBERT |

`THESIS_SKIP_HEBREW_TEXT_VALIDATION=1` is set automatically for non-Hebrew corpora.

## Layout

```
projects/small_data_ner_benchmark/
  run_cross_benchmark_comparison.py   # main runner
  run_benchmark.py                    # alias
  configs.py / corpus_loaders.py / splits.py
  data/<benchmark>/corpus.csv, split_meta.json, splits/<regime>/...
  outputs/cross_comparison/
```
