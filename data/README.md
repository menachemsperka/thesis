Place the required training datasets in this folder:

- `ner_dataset.pkl` — **default** labeled corpus (built from `ner_dataset.xlsx`; avoids CSV encoding issues)
- `ner_dataset.xlsx` — source export for rebuilding the pickle (`python scripts/build_ner_dataset_pkl.py`)
- `ner_training_generated.csv`
- `ner_training_duplicated.csv`

Legacy `ner_dataset.csv` is optional; loaders prefer `ner_dataset.pkl` when present.
