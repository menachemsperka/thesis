"""Load public NER corpora and convert to thesis sentence + CSV formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Non-interactive Hub loads on Colab/CI (conll2003 and similar script datasets).
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

CONLL_TAG_NAMES = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def _conll_id_to_bio(tag_id: int) -> str:
    if 0 <= int(tag_id) < len(CONLL_TAG_NAMES):
        return CONLL_TAG_NAMES[int(tag_id)]
    return "O"


def _hf_row_to_sentence(tokens: list[str], labels: list[Any], label_names: list[str] | None) -> dict:
    bio_labels: list[str] = []
    for lab in labels:
        if isinstance(lab, int):
            if label_names and 0 <= lab < len(label_names):
                bio_labels.append(str(label_names[lab]))
            else:
                bio_labels.append(_conll_id_to_bio(lab))
        else:
            bio_labels.append(str(lab).strip() or "O")
    text = " ".join(str(t) for t in tokens)
    if len(bio_labels) != len(tokens):
        raise ValueError(f"Token/label length mismatch ({len(tokens)} vs {len(bio_labels)})")
    return {"text": text, "labels": bio_labels}


def _split_to_sentences(split, token_col: str, label_col: str, label_names: list[str] | None) -> list[dict]:
    sentences: list[dict] = []
    for row in split:
        tokens = list(row[token_col])
        labels = list(row[label_col])
        sentences.append(_hf_row_to_sentence(tokens, labels, label_names))
    return sentences


def _guess_columns(column_names: list[str]) -> tuple[str, str]:
    token_candidates = ["tokens", "words", "text_tokens", "sentence_tokens"]
    label_candidates = ["ner_tags", "labels", "tags", "bio_tags", "entity_tags"]
    token_col = next((n for n in token_candidates if n in column_names), None)
    label_col = next((n for n in label_candidates if n in column_names), None)
    if not token_col or not label_col:
        raise ValueError(f"Cannot infer token/label columns from {column_names}")
    return token_col, label_col


def _label_names_from_feature(split, label_col: str) -> list[str] | None:
    feature = split.features.get(label_col)
    if feature is None:
        return None
    inner = getattr(feature, "feature", None)
    names = getattr(inner, "names", None)
    if names:
        return list(names)
    return None


def _load_dataset_from_snapshot(snapshot_dir: Path):
    from datasets import Dataset, DatasetDict, load_dataset

    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    jsonl_files = sorted(snapshot_dir.rglob("*.jsonl"))
    json_files = sorted(snapshot_dir.rglob("*.json"))

    def by_split(paths: list[Path]) -> dict[str, str]:
        split_map: dict[str, str] = {}
        for path in paths:
            name = path.name.lower()
            if "train" in name:
                split_map["train"] = str(path)
            elif "validation" in name or "valid" in name or "dev" in name:
                split_map["validation"] = str(path)
            elif "test" in name:
                split_map["test"] = str(path)
        return split_map

    if parquet_files:
        data_files = by_split(parquet_files)
        if data_files:
            return DatasetDict(load_dataset("parquet", data_files=data_files))
    candidate_json = [p for p in [*jsonl_files, *json_files] if p.name.lower() != "dataset_infos.json"]
    if candidate_json:
        data_files = by_split(candidate_json)
        if data_files:
            return DatasetDict(load_dataset("json", data_files=data_files))

    raise RuntimeError(f"No parquet/json splits found under snapshot: {snapshot_dir}")


def _trust_remote_code() -> bool:
    raw = (os.environ.get("THESIS_HF_TRUST_REMOTE_CODE") or os.environ.get("HF_DATASETS_TRUST_REMOTE_CODE") or "1")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _snapshot_fallback(repo_id: str, cache_dir: Path):
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )
    return _load_dataset_from_snapshot(snapshot_dir)


def _load_hf_dataset(repo_id: str, cache_dir: Path, config_name: str | None = None):
    from datasets import load_dataset

    trust = _trust_remote_code()
    load_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir), "trust_remote_code": trust}

    def _try_load(target_id: str, config: str | None) -> Any:
        if config:
            return load_dataset(target_id, config, **load_kwargs)
        return load_dataset(target_id, **load_kwargs)

    last_err: Exception | None = None
    try:
        return _try_load(repo_id, config_name)
    except ValueError as exc:
        last_err = exc
        if "trust_remote_code" in str(exc).lower() or "custom code" in str(exc).lower():
            load_kwargs["trust_remote_code"] = True
            return _try_load(repo_id, config_name)
        raise
    except RuntimeError as exc:
        last_err = exc
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
    except Exception as exc:
        last_err = exc

    try:
        return _snapshot_fallback(repo_id, cache_dir)
    except Exception as snap_err:
        raise RuntimeError(f"Failed to load dataset {repo_id!r} (config={config_name!r})") from (snap_err or last_err)


def load_conll2003(cache_dir: Path) -> dict[str, list[dict]]:
    last_err: Exception | None = None
    ds = None
    for repo_id in ("conll2003", "tner/conll2003", "eriktks/conll2003"):
        try:
            ds = _load_hf_dataset(repo_id, cache_dir)
            break
        except Exception as exc:
            last_err = exc
    if ds is None:
        raise RuntimeError("Failed to load CoNLL-2003 from Hub mirrors") from last_err
    token_col, label_col = _guess_columns(ds["train"].column_names)
    label_names = _label_names_from_feature(ds["train"], label_col)
    return {
        "train": _split_to_sentences(ds["train"], token_col, label_col, label_names),
        "validation": _split_to_sentences(ds["validation"], token_col, label_col, label_names),
        "test": _split_to_sentences(ds["test"], token_col, label_col, label_names),
    }


def _load_dataset_with_fallbacks(dataset_ids: list[str], cache_dir: Path, config_name: str | None = None):
    last_err: Exception | None = None
    for ds_id in dataset_ids:
        if not ds_id:
            continue
        try:
            return _load_hf_dataset(ds_id, cache_dir, config_name=config_name if ds_id.startswith("bigbio/") else None)
        except Exception as exc:
            last_err = exc
            config_name = None
    raise RuntimeError(f"Failed to load dataset from {dataset_ids}") from last_err


def load_nemo(cache_dir: Path) -> dict[str, list[dict]]:
    override = (os.environ.get("THESIS_BENCHMARK_NEMO_DATASET") or "").strip()
    candidates = [override] if override else []
    candidates.extend(["onlplab/nemo", "imvladikon/nemo_corpus"])
    ds = _load_dataset_with_fallbacks(candidates, cache_dir)
    if not isinstance(ds, dict) or "train" not in ds:
        raise ValueError("NEMO dataset must expose train/validation/test splits")

    token_col, label_col = _guess_columns(ds["train"].column_names)
    label_names = _label_names_from_feature(ds["train"], label_col)
    splits: dict[str, list[dict]] = {}
    for name in ds.keys():
        splits[name] = _split_to_sentences(ds[name], token_col, label_col, label_names)

    if "validation" not in splits and "dev" in splits:
        splits["validation"] = splits.pop("dev")
    if "test" not in splits:
        if "validation" in splits:
            splits["test"] = splits["validation"]
        else:
            raise ValueError("NEMO dataset has no test or validation split")
    if "train" not in splits:
        raise ValueError("NEMO dataset has no train split")
    return splits


def load_bc5cdr(cache_dir: Path) -> dict[str, list[dict]]:
    last_err: Exception | None = None
    ds = None
    for repo_id, config in (("bigbio/bc5cdr", "bc5cdr_bigbio_kb"), ("tner/bc5cdr", None)):
        try:
            ds = _load_hf_dataset(repo_id, cache_dir, config_name=config)
            break
        except Exception as exc:
            last_err = exc
    if ds is None:
        raise RuntimeError("Failed to load BC5CDR from bigbio/bc5cdr or tner/bc5cdr") from last_err
    if not isinstance(ds, dict):
        raise ValueError("BC5CDR loader returned unexpected structure")

    token_col, label_col = _guess_columns(ds["train"].column_names)
    label_names = _label_names_from_feature(ds["train"], label_col)
    out: dict[str, list[dict]] = {}
    for name in ds.keys():
        out[name] = _split_to_sentences(ds[name], token_col, label_col, label_names)

    if "validation" not in out and "dev" in out:
        out["validation"] = out.pop("dev")
    if "test" not in out:
        out["test"] = out.get("validation", out["train"])
    return out


def load_benchmark_splits(dataset_key: str, cache_dir: Path) -> dict[str, list[dict]]:
    if dataset_key == "conll2003":
        return load_conll2003(cache_dir)
    if dataset_key == "nemo":
        return load_nemo(cache_dir)
    if dataset_key == "bc5cdr":
        return load_bc5cdr(cache_dir)
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def sentences_to_corpus_csv(sentences: list[dict], start_id: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, sent in enumerate(sentences, start=start_id):
        tokens = str(sent.get("text", "")).split()
        labels = list(sent.get("labels", []))
        for tok, lab in zip(tokens, labels):
            rows.append({"id": idx, "token": tok, "raw_tags": str(lab)})
    return pd.DataFrame(rows)


def write_corpus_csv(all_sentences: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = sentences_to_corpus_csv(all_sentences)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def sample_sentence_pool(train_sentences: list[dict], pool_size: int, pool_seed: int) -> list[dict]:
    if len(train_sentences) < pool_size:
        raise ValueError(
            f"Train split has {len(train_sentences)} sentences; need at least {pool_size} for the small-data pool."
        )
    rng = np.random.default_rng(pool_seed)
    indices = rng.choice(len(train_sentences), size=pool_size, replace=False)
    return [train_sentences[int(i)] for i in indices]
