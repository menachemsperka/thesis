"""
split_io.py — Shared I/O utilities for saving and loading pre-computed
train/eval sentence splits produced by experiment 07.

Saved Format
------------
Each JSON file stores a list of sentence dicts in the format used by
``th_functions.train_data_fit``:

    [
        {"text": "token1 token2 ...", "labels": ["B-PER", "I-PER", "O", ...]},
        ...
    ]

Conversion helpers translate this format into the structures expected by
the AUC cascaded pipeline (``tokens`` / ``bio_tags`` / ``entity_types``)
and into DataFrame rows for experiments that operate on raw DataFrames
(experiment 03).

Public API
----------
``save_split``          — persist a sentence list to JSON.
``load_split``          — read a sentence list from JSON.
``sentences_to_cascaded`` — convert sentence dicts to cascaded-pipeline format.
``sentences_to_dataframe`` — convert sentence dicts to a token-level DataFrame.
``get_splits_dir``      — canonical path for saved splits.
``build_thesis_documentation_df`` — generate a documentation DataFrame for Excel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "outputs" / "exp07" / "splits"


# ============================================================================
# Save / Load
# ============================================================================

def get_splits_dir() -> Path:
    """Return (and create) the canonical directory for saved splits."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    return SPLITS_DIR


def save_split(sentences: list[dict], path: Path) -> Path:
    """Save a list of sentence dicts to *path* as JSON (UTF-8).

    Each sentence dict must have ``text`` (str) and ``labels`` (list[str]).
    Returns the path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for sent in sentences:
        payload.append({
            "text": sent.get("text", ""),
            "labels": list(sent.get("labels", [])),
        })
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split(path: Path) -> list[dict]:
    """Load a list of sentence dicts from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    sentences = []
    for item in raw:
        sentences.append({
            "text": str(item.get("text", "")),
            "labels": list(item.get("labels", [])),
        })
    return sentences


# ============================================================================
# Format Conversion — sentence dicts ↔ cascaded pipeline format
# ============================================================================

def _parse_bio_label(label: str) -> tuple[str, str | None]:
    """Split a BIO tag like 'B-PER' into ('B', 'PER').  'O' → ('O', None)."""
    label = str(label).strip()
    if label == "O" or not label:
        return "O", None
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    return "O", None


def sentences_to_cascaded(sentences: list[dict]) -> list[dict]:
    """Convert exp07 sentence dicts to cascaded-pipeline format.

    Input format:  ``{"text": "tok1 tok2 ...", "labels": ["B-PER", "O", ...]}``
    Output format: ``{"tokens": [...], "bio_tags": [...], "entity_types": [...]}``
    """
    result = []
    for sent in sentences:
        tokens = str(sent.get("text", "")).split()
        labels = list(sent.get("labels", []))
        bio_tags = []
        entity_types = []
        for label in labels:
            bio, etype = _parse_bio_label(label)
            bio_tags.append(bio)
            entity_types.append(etype)
        result.append({
            "tokens": tokens,
            "bio_tags": bio_tags,
            "entity_types": entity_types,
        })
    return result


def sentences_to_dataframe(sentences: list[dict], start_id: int = 1) -> pd.DataFrame:
    """Convert sentence dicts to a token-level DataFrame suitable for exp03.

    Columns: ``id``, ``token``, ``raw_tags`` — matching the schema of
    ``ner_dataset.csv``.
    """
    rows = []
    for idx, sent in enumerate(sentences, start=start_id):
        tokens = str(sent.get("text", "")).split()
        labels = list(sent.get("labels", []))
        for tok, label in zip(tokens, labels):
            rows.append({"id": idx, "token": tok, "raw_tags": str(label)})
    return pd.DataFrame(rows)


# ============================================================================
# Documentation helper for Excel thesis tab
# ============================================================================

def build_thesis_documentation_df(
    experiment_id: str,
    experiment_name: str,
    extra_rows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame suitable for an Excel 'documentation' worksheet.

    Contains key–value pairs describing the experiment and how results should
    be interpreted for academic citation.
    """
    rows: list[dict[str, str]] = [
        {"Section": "Experiment", "Key": "ID", "Value": experiment_id},
        {"Section": "Experiment", "Key": "Name", "Value": experiment_name},
        {"Section": "Dataset", "Key": "Source", "Value": "ner_dataset.csv (Hebrew NER corpus)"},
        {"Section": "Dataset", "Key": "Format", "Value": "Token-level BIO tags (B-TYPE, I-TYPE, O)"},
        {"Section": "Split Protocol", "Key": "Train / Eval Ratio", "Value": "70% / 30%"},
        {"Section": "Split Protocol", "Key": "Split Granularity", "Value": "Sentence-level (no token leakage)"},
        {"Section": "Split Protocol", "Key": "Seeds", "Value": "5 independent random seeds (42–46)"},
        {"Section": "Split Protocol", "Key": "Stratification", "Value": "Non-O label distribution preserved via greedy optimisation"},
        {"Section": "Model", "Key": "Base Model", "Value": "DictaBERT (dicta-il/dictabert)"},
        {"Section": "Model", "Key": "Language", "Value": "Hebrew"},
        {"Section": "Evaluation", "Key": "Primary Metric", "Value": "Entity-level F1 (seqeval strict)"},
        {"Section": "Evaluation", "Key": "Secondary Metrics", "Value": "Precision, Recall, Accuracy"},
        {"Section": "Evaluation", "Key": "Aggregation", "Value": "Mean ± standard deviation across seeds"},
        {"Section": "Reporting", "Key": "Significance", "Value": "Paired seed-by-seed comparison (same train/eval per seed)"},
        {"Section": "Reporting", "Key": "Delta", "Value": "Improvement = split_variant_metric − baseline_metric"},
        {"Section": "Citation", "Key": "Condition Labels",
         "Value": "Baseline = simple random split; others = label-aware split strategies from Experiment 07"},
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return pd.DataFrame(rows)
