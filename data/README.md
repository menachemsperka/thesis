Place the required training datasets in this folder:

| File | Use |
|------|-----|
| **`ner_dataset.pkl`** | Default **full** corpus (~300 sentences, tracked in git) |
| **`ner_dataset_full.pkl`** | Same content as `ner_dataset.pkl` — use for explicit full-corpus runs |
| **`ner_dataset_150_seed42.pkl`** | Fixed **150-sentence** subset (`random_state=42`, tracked in git) |
| `ner_dataset.xlsx` | Source export for rebuilding pickles |

Rebuild all canonical pickles from the current source:

```bash
python scripts/build_ner_dataset_corpus_pkls.py
```

Legacy `ner_dataset.csv` is optional; loaders prefer `.pkl` when present.

**Colab / runner:** point at a fixed pickle (supports `.pkl` paths despite the env name):

```python
os.environ["THESIS_NER_CSV"] = "/path/to/repo/data/ner_dataset_150_seed42.pkl"
```

For the 150-sentence pilot, use the prebuilt PKL above instead of `--subset-sentences 150` so every machine sees the same sentences.

Other optional files:

- `ner_training_generated.csv`
- `ner_training_duplicated.csv`
