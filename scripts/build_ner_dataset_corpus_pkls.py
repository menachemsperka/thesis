#!/usr/bin/env python3
"""Build canonical NER corpus pickles: full (~300 sentences) and 150-sentence subset.

Outputs (under ``data/``):

* ``ner_dataset.pkl`` — default full corpus (same bytes as ``ner_dataset_full.pkl``)
* ``ner_dataset_full.pkl`` — explicit full-corpus alias for full-dataset runs
* ``ner_dataset_150_seed42.pkl`` — fixed 150 sentence ids (``random_state=42``)

Source priority: existing ``ner_dataset.pkl`` → ``ner_dataset.xlsx`` → ``ner_dataset.csv``.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CORE_DIR = PROJECT_ROOT / "core"

DEFAULT_PKL = DATA_DIR / "ner_dataset.pkl"
FULL_PKL = DATA_DIR / "ner_dataset_full.pkl"
SUBSET_PKL = DATA_DIR / "ner_dataset_150_seed42.pkl"

SUBSET_SENTENCES = 150
SUBSET_SEED = 42


def _load_source_df() -> pd.DataFrame:
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    from hebrew_text_io import read_ner_dataset

    for path in (
        DATA_DIR / "ner_dataset.pkl",
        DATA_DIR / "ner_dataset.xlsx",
        DATA_DIR / "ner_dataset.csv",
    ):
        if path.exists():
            df, _enc = read_ner_dataset(path)
            return df
    raise FileNotFoundError(
        f"No source corpus under {DATA_DIR} (need pkl, xlsx, or csv)."
    )


def _write_pkl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    n_sent = int(df["id"].nunique()) if "id" in df.columns else -1
    print(f"Wrote {path.name}: {len(df)} token rows, {n_sent} sentences, {path.stat().st_size} bytes")


def _subset_150(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Expected column 'id' for sentence grouping.")
    sentence_ids = df["id"].dropna().drop_duplicates().astype(str).tolist()
    take = min(SUBSET_SENTENCES, len(sentence_ids))
    sampled = (
        pd.Series(sentence_ids)
        .sample(n=take, random_state=SUBSET_SEED, replace=False)
        .tolist()
    )
    out = df[df["id"].astype(str).isin(sampled)].copy()
    return out


def main() -> int:
    df = _load_source_df()
    n_full = int(df["id"].nunique()) if "id" in df.columns else 0
    print(f"Source corpus: {len(df)} rows, {n_full} sentences")

    _write_pkl(df, FULL_PKL)
    _write_pkl(df, DEFAULT_PKL)

    sub = _subset_150(df)
    _write_pkl(sub, SUBSET_PKL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
