from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Proxy setup (Windows + Linux):
# - Uses explicit project defaults for Intel network access
# - Still allows override via THESIS_HTTP_PROXY / THESIS_HTTPS_PROXY / THESIS_NO_PROXY
# - Mirrors values to both uppercase/lowercase names for compatibility
http_proxy = (
    os.environ.get("THESIS_HTTP_PROXY")
    or os.environ.get("HTTP_PROXY")
    or os.environ.get("http_proxy")
    or "http://proxy-dmz.intel.com:912"
)
https_proxy = (
    os.environ.get("THESIS_HTTPS_PROXY")
    or os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
    or "http://proxy-dmz.intel.com:912"
)
no_proxy = (
    os.environ.get("THESIS_NO_PROXY")
    or os.environ.get("NO_PROXY")
    or os.environ.get("no_proxy")
    or "localhost,intel.com,127.0.0.1"
)

os.environ["HTTP_PROXY"] = http_proxy
os.environ["http_proxy"] = http_proxy
os.environ["HTTPS_PROXY"] = https_proxy
os.environ["https_proxy"] = https_proxy
os.environ["NO_PROXY"] = no_proxy
os.environ["no_proxy"] = no_proxy

# TLS setup for corporate SSL interception:
# - THESIS_CA_BUNDLE points to PEM file with corporate root/intermediate certs
# - THESIS_DISABLE_SSL_VERIFY=1 disables certificate verification (last resort)
ca_bundle = (
    os.environ.get("THESIS_CA_BUNDLE")
    or os.environ.get("REQUESTS_CA_BUNDLE")
    or os.environ.get("SSL_CERT_FILE")
    or ""
).strip()
disable_ssl_verify = (os.environ.get("THESIS_DISABLE_SSL_VERIFY") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if ca_bundle:
    os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
    os.environ["CURL_CA_BUNDLE"] = ca_bundle
    os.environ["SSL_CERT_FILE"] = ca_bundle

if disable_ssl_verify:
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prep"
MODELS_DIR = PROJECT_ROOT / "models"
HF_MODELS_CACHE_DIR = MODELS_DIR / "hf_models"
HF_DATASETS_CACHE_DIR = MODELS_DIR / "hf_datasets"
HF_NEMO_DATASET_SAVED_DIR = HF_DATASETS_CACHE_DIR / "imvladikon__nemo_corpus_saved"
DATASET_ID = "imvladikon/nemo_corpus"


@dataclass
class ModelRunSpec:
    model_id: str
    train: bool


@dataclass
class PrepRunConfig:
    # Core split/training params
    seed: int
    validation_size: float
    learning_rate: float
    epochs: int
    batch_size: int
    # Speed controls
    max_train_steps: int
    max_train_samples: int
    max_eval_samples: int
    max_test_samples: int
    model_limit: int


# Central place for quick iteration vs fuller runs.
# You can switch profile with THESIS_PREP_PROFILE=ultra_fast|smoke|full.
RUN_CONFIGS: dict[str, PrepRunConfig] = {
    "ultra_fast": PrepRunConfig(
        seed=42,
        validation_size=0.1,
        learning_rate=2e-5,
        epochs=1,
        batch_size=4,
        max_train_steps=20,
        max_train_samples=128,
        max_eval_samples=64,
        max_test_samples=64,
        model_limit=1,
    ),
    "smoke": PrepRunConfig(
        seed=42,
        validation_size=0.1,
        learning_rate=3e-5,
        epochs=2,
        batch_size=8,
        max_train_steps=120,
        max_train_samples=1200,
        max_eval_samples=256,
        max_test_samples=256,
        model_limit=1,
    ),
    "full": PrepRunConfig(
        seed=42,
        validation_size=0.1,
        learning_rate=2e-5,
        epochs=3,
        batch_size=8,
        max_train_steps=-1,
        max_train_samples=0,
        max_eval_samples=0,
        max_test_samples=0,
        model_limit=0,
    ),
}


RUN_SPECS: list[ModelRunSpec] = [
    ModelRunSpec("dicta-il/alephbertgimmel-base", True),
    ModelRunSpec("HeNLP/HeRo", True),
    ModelRunSpec("dicta-il/dictabert", True),
    ModelRunSpec("dicta-il/BEREL_3.0", True),
    ModelRunSpec("dicta-il/dictabert-ner", False),
]


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_splits(ds: DatasetDict, validation_size: float, seed: int) -> DatasetDict:
    has_train = "train" in ds
    has_validation = "validation" in ds
    has_test = "test" in ds

    if has_train and has_validation and has_test:
        return ds

    if has_train and has_test and not has_validation:
        split = ds["train"].train_test_split(test_size=validation_size, seed=seed)
        return DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                "test": ds["test"],
            }
        )

    if has_train and not has_validation and not has_test:
        split_once = ds["train"].train_test_split(test_size=0.2, seed=seed)
        split_twice = split_once["test"].train_test_split(test_size=0.5, seed=seed)
        return DatasetDict(
            {
                "train": split_once["train"],
                "validation": split_twice["train"],
                "test": split_twice["test"],
            }
        )

    raise ValueError(f"Unsupported dataset split configuration: {list(ds.keys())}")


def _repo_to_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    )


def _get_or_download_model_dir(model_id: str) -> Path:
    local_dir = HF_MODELS_CACHE_DIR / _repo_to_local_name(model_id)
    built_in_dir = MODELS_DIR / model_id.split("/")[-1]

    for candidate in (local_dir, built_in_dir):
        if _is_model_dir(candidate):
            return candidate

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(HF_MODELS_CACHE_DIR),
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            cache_dir=str(HF_MODELS_CACHE_DIR),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model '{model_id}' from Hugging Face. "
            "Check proxy/SSL settings or provide a local model copy in models/."
        ) from exc

    tokenizer.save_pretrained(str(local_dir))
    model.save_pretrained(str(local_dir))
    return local_dir


def _prepare_models() -> dict[str, Path]:
    prepared: dict[str, Path] = {}
    print("Preparing models (local-first)...")
    for spec in RUN_SPECS:
        try:
            model_dir = _get_or_download_model_dir(spec.model_id)
        except Exception as exc:
            print(f"Skipping model {spec.model_id}: {exc}")
            continue
        prepared[spec.model_id] = model_dir
        print(f"Model ready: {spec.model_id} -> {model_dir}")

    if not prepared:
        raise RuntimeError(
            "No model is available locally and downloads failed. "
            "Place at least one model under models/ or fix proxy/SSL access."
        )

    skipped = len(RUN_SPECS) - len(prepared)
    if skipped:
        print(f"Continuing with {len(prepared)} model(s); skipped {skipped} unavailable model(s).")

    return prepared


def _load_dataset_from_snapshot(snapshot_dir: Path) -> DatasetDict:
    # Prefer the canonical NEMO flat-token files when present.
    preferred_bmes = {
        "train": snapshot_dir / "data" / "spmrl" / "gold" / "token-single_gold_train.bmes",
        "validation": snapshot_dir / "data" / "spmrl" / "gold" / "token-single_gold_dev.bmes",
        "test": snapshot_dir / "data" / "spmrl" / "gold" / "token-single_gold_test.bmes",
    }
    if all(path.exists() for path in preferred_bmes.values()):
        def parse_bmes(path: Path) -> tuple[list[list[str]], list[list[str]]]:
            sentences_tokens: list[list[str]] = []
            sentences_labels: list[list[str]] = []
            current_tokens: list[str] = []
            current_labels: list[str] = []

            with path.open("r", encoding="utf-8") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line:
                        if current_tokens:
                            sentences_tokens.append(current_tokens)
                            sentences_labels.append(current_labels)
                            current_tokens = []
                            current_labels = []
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    token = " ".join(parts[:-1])
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)

            if current_tokens:
                sentences_tokens.append(current_tokens)
                sentences_labels.append(current_labels)

            return sentences_tokens, sentences_labels

        datasets_by_split: dict[str, Dataset] = {}
        for split_name, split_file in preferred_bmes.items():
            tokens, tags = parse_bmes(split_file)
            datasets_by_split[split_name] = Dataset.from_dict({"tokens": tokens, "ner_tags": tags})
        return DatasetDict(datasets_by_split)

    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    jsonl_files = sorted(snapshot_dir.rglob("*.jsonl"))
    json_files = sorted(snapshot_dir.rglob("*.json"))
    csv_files = sorted(snapshot_dir.rglob("*.csv"))

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
        return DatasetDict(load_dataset("parquet", data_files=[str(path) for path in parquet_files]))

    candidate_json = [path for path in [*jsonl_files, *json_files] if path.name.lower() != "dataset_infos.json"]
    if candidate_json:
        data_files = by_split(candidate_json)
        if data_files:
            return DatasetDict(load_dataset("json", data_files=data_files))
        return DatasetDict(load_dataset("json", data_files=[str(path) for path in candidate_json]))

    if csv_files:
        data_files = by_split(csv_files)
        if data_files:
            return DatasetDict(load_dataset("csv", data_files=data_files))
        return DatasetDict(load_dataset("csv", data_files=[str(path) for path in csv_files]))

    split_files: dict[str, Path] = {}
    bmes_candidates = sorted(snapshot_dir.rglob("*.bmes"))
    for path in bmes_candidates:
        lower_name = path.name.lower()
        if "token-single" not in lower_name:
            continue
        if "train" in lower_name and "train" not in split_files:
            split_files["train"] = path
        elif ("validation" in lower_name or "valid" in lower_name or "dev" in lower_name) and "validation" not in split_files:
            split_files["validation"] = path
        elif "test" in lower_name and "test" not in split_files:
            split_files["test"] = path

    if split_files:
        def parse_bmes(path: Path) -> tuple[list[list[str]], list[list[str]]]:
            sentences_tokens: list[list[str]] = []
            sentences_labels: list[list[str]] = []
            current_tokens: list[str] = []
            current_labels: list[str] = []

            with path.open("r", encoding="utf-8") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line:
                        if current_tokens:
                            sentences_tokens.append(current_tokens)
                            sentences_labels.append(current_labels)
                            current_tokens = []
                            current_labels = []
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    token = " ".join(parts[:-1])
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)

            if current_tokens:
                sentences_tokens.append(current_tokens)
                sentences_labels.append(current_labels)

            return sentences_tokens, sentences_labels

        datasets_by_split: dict[str, Dataset] = {}
        for split_name, split_file in split_files.items():
            tokens, tags = parse_bmes(split_file)
            datasets_by_split[split_name] = Dataset.from_dict({"tokens": tokens, "ner_tags": tags})

        return DatasetDict(datasets_by_split)

    raise RuntimeError(
        "No supported dataset files were found in the snapshot (parquet/json/csv/bmes). "
        f"Snapshot dir: {snapshot_dir}"
    )


def _is_outside_label(value: Any, label_list: list[str] | None) -> bool:
    text = str(value).strip()
    if text.upper() == "O":
        return True
    if label_list is not None:
        try:
            idx = int(value)
            if 0 <= idx < len(label_list):
                return str(label_list[idx]).strip().upper() == "O"
        except (TypeError, ValueError):
            pass
    return False


def _split_non_o_count(split_ds: Dataset, label_col: str, label_list: list[str] | None) -> int:
    count = 0
    for example in split_ds:
        for value in example[label_col]:
            if not _is_outside_label(value, label_list):
                count += 1
    return count


def _dataset_has_reasonable_ner_coverage(ds: DatasetDict) -> bool:
    if "train" not in ds:
        return False

    token_col, label_col = _guess_columns(ds)
    label_list = _get_label_list(ds, label_col)

    if "validation" not in ds or "test" not in ds:
        return _split_non_o_count(ds["train"], label_col, label_list) > 0

    return (
        _split_non_o_count(ds["train"], label_col, label_list) > 0
        and _split_non_o_count(ds["validation"], label_col, label_list) > 0
        and _split_non_o_count(ds["test"], label_col, label_list) > 0
    )


def _load_remote_dataset_with_fallback(dataset_id: str) -> DatasetDict:
    try:
        remote_ds = load_dataset(dataset_id, cache_dir=str(HF_DATASETS_CACHE_DIR))
        return DatasetDict(remote_ds)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        snapshot_dir = Path(
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                cache_dir=str(HF_DATASETS_CACHE_DIR),
            )
        )
        return _load_dataset_from_snapshot(snapshot_dir)


def _load_nemo_dataset() -> DatasetDict:
    local_file = (os.environ.get("THESIS_PREP_DATASET_LOCAL") or "").strip()
    force_rebuild = (os.environ.get("THESIS_PREP_REBUILD_DATASET_CACHE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not local_file:
        if HF_NEMO_DATASET_SAVED_DIR.exists() and not force_rebuild:
            loaded = DatasetDict(load_from_disk(str(HF_NEMO_DATASET_SAVED_DIR)))
            if _dataset_has_reasonable_ner_coverage(loaded):
                return loaded
            print(
                "Cached NEMO dataset appears degenerate for NER (missing non-O labels in one or more splits). "
                "Rebuilding cache from snapshot files."
            )

        dataset_dict = _load_remote_dataset_with_fallback(DATASET_ID)
        if not _dataset_has_reasonable_ner_coverage(dataset_dict):
            raise RuntimeError(
                "Loaded NEMO dataset has no non-O labels in one or more splits. "
                "This indicates an invalid source/cache state."
            )

        if HF_NEMO_DATASET_SAVED_DIR.exists():
            shutil.rmtree(HF_NEMO_DATASET_SAVED_DIR, ignore_errors=True)

        try:
            dataset_dict.save_to_disk(str(HF_NEMO_DATASET_SAVED_DIR))
        except Exception as exc:
            print(f"Warning: could not persist NEMO dataset cache to disk: {exc}")
        return dataset_dict

    local_path = Path(local_file)
    if not local_path.exists():
        raise FileNotFoundError(f"THESIS_PREP_DATASET_LOCAL does not exist: {local_path}")

    suffix = local_path.suffix.lower()
    if suffix == ".csv":
        ds = load_dataset("csv", data_files=str(local_path))
    elif suffix in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files=str(local_path))
    elif local_path.is_dir():
        ds = load_dataset(str(local_path))
    else:
        raise ValueError(
            "Unsupported local dataset format. "
            "Use .csv, .json/.jsonl, or a dataset script directory."
        )

    return DatasetDict(ds)


def _guess_columns(ds: DatasetDict) -> tuple[str, str]:
    train = ds["train"]
    columns = train.column_names

    token_candidates = ["tokens", "words", "text_tokens", "sentence_tokens"]
    label_candidates = ["ner_tags", "labels", "tags", "bio_tags", "entity_tags"]

    token_col = next((name for name in token_candidates if name in columns), None)
    label_col = next((name for name in label_candidates if name in columns), None)

    if token_col and label_col:
        return token_col, label_col

    raise ValueError(
        "Could not infer token/label columns for NER dataset. "
        f"Available columns: {columns}. "
        f"Expected token columns: {token_candidates}, label columns: {label_candidates}"
    )


def _get_label_list(ds: DatasetDict, label_col: str) -> list[str]:
    train = ds["train"]
    feature = train.features[label_col]

    if hasattr(feature, "feature") and hasattr(feature.feature, "names") and feature.feature.names:
        return list(feature.feature.names)

    labels_set: set[str] = set()
    for example in train:
        for value in example[label_col]:
            labels_set.add(str(value))

    ordered = sorted(labels_set)
    if "O" in ordered:
        ordered.remove("O")
        ordered = ["O", *ordered]
    return ordered


def _ensure_numeric_labels(ds: DatasetDict, label_col: str, label_list: list[str]) -> DatasetDict:
    train_split = ds["train"]
    first_non_empty: list[Any] | None = None
    for example in train_split:
        values = example.get(label_col, [])
        if values:
            first_non_empty = values
            break

    if not first_non_empty:
        return ds

    if not isinstance(first_non_empty[0], str):
        return ds

    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    def encode_labels(example: dict[str, Any]) -> dict[str, Any]:
        encoded = [label_to_id.get(str(value), 0) for value in example[label_col]]
        return {label_col: encoded}

    return DatasetDict(
        {
            split_name: split_ds.map(encode_labels)
            for split_name, split_ds in ds.items()
        }
    )


def _build_align_fn(tokenizer, token_col: str, label_col: str):
    def tokenize_and_align_labels(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples[token_col],
            truncation=True,
            is_split_into_words=True,
        )

        aligned_labels = []
        for idx, labels in enumerate(examples[label_col]):
            word_ids = tokenized.word_ids(batch_index=idx)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
                prev_word_id = word_id
            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return tokenize_and_align_labels


def _take_first_n(split_ds: Dataset, limit: int) -> Dataset:
    if limit <= 0:
        return split_ds
    if len(split_ds) <= limit:
        return split_ds
    return split_ds.select(range(limit))


def _to_label_strings(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: list[str],
    id2label_override: dict[int, str] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    pred_ids = np.argmax(predictions, axis=2)

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []

    for pred_row, label_row in zip(pred_ids, labels):
        sentence_true: list[str] = []
        sentence_pred: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue

            if 0 <= int(label_id) < len(label_list):
                true_label = label_list[int(label_id)]
            else:
                true_label = str(label_id)

            if id2label_override is not None:
                pred_label = id2label_override.get(int(pred_id), str(int(pred_id)))
            elif 0 <= int(pred_id) < len(label_list):
                pred_label = label_list[int(pred_id)]
            else:
                pred_label = str(pred_id)

            sentence_true.append(true_label)
            sentence_pred.append(pred_label)

        true_labels.append(sentence_true)
        pred_labels.append(sentence_pred)

    return true_labels, pred_labels


def _compute_metrics(true_labels: list[list[str]], pred_labels: list[list[str]]) -> dict[str, float]:
    true_entity_count = sum(1 for sentence in true_labels for tag in sentence if tag != "O")
    pred_entity_count = sum(1 for sentence in pred_labels for tag in sentence if tag != "O")

    if true_entity_count == 0:
        # seqeval emits undefined-metric warnings when no true entities exist in the split.
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_entity_tokens": float(true_entity_count),
            "pred_entity_tokens": float(pred_entity_count),
        }

    return {
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
        "true_entity_tokens": float(true_entity_count),
        "pred_entity_tokens": float(pred_entity_count),
    }


def _train_and_evaluate(
    model_id: str,
    model_dir: Path,
    train_enabled: bool,
    ds: DatasetDict,
    token_col: str,
    label_col: str,
    label_list: list[str],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    max_train_steps: int,
    max_train_samples: int,
    max_eval_samples: int,
    max_test_samples: int,
    seed: int,
) -> dict[str, Any]:
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    align_fn = _build_align_fn(tokenizer, token_col, label_col)

    encoded = ds.map(
        align_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    train_ds = _take_first_n(encoded["train"], max_train_samples)
    eval_ds = _take_first_n(encoded["validation"], max_eval_samples)
    test_ds = _take_first_n(encoded["test"], max_test_samples)

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    run_name = model_id.replace("/", "__")
    run_output = OUTPUT_DIR / f"tmp_{run_name}_{_now()}"

    args = TrainingArguments(
        output_dir=str(run_output),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_train_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    if train_enabled:
        trainer.train()

    pred_output = trainer.predict(test_ds)

    model_id2label = getattr(trainer.model.config, "id2label", None)
    normalized_id2label = None
    if isinstance(model_id2label, dict):
        normalized_id2label = {}
        for key, value in model_id2label.items():
            try:
                normalized_id2label[int(key)] = str(value)
            except (TypeError, ValueError):
                continue

    true_labels, pred_labels = _to_label_strings(
        pred_output.predictions,
        pred_output.label_ids,
        label_list=label_list,
        id2label_override=normalized_id2label,
    )
    metrics = _compute_metrics(true_labels, pred_labels)

    return {
        "model": model_id,
        "train_enabled": train_enabled,
        "epochs": epochs if train_enabled else 0,
        "learning_rate": learning_rate if train_enabled else None,
        "max_train_steps": max_train_steps if train_enabled else 0,
        "samples_train": len(train_ds),
        "samples_validation": len(eval_ds),
        "samples_test": len(test_ds),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "true_entity_tokens": int(metrics["true_entity_tokens"]),
        "pred_entity_tokens": int(metrics["pred_entity_tokens"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    profile_name = (os.environ.get("THESIS_PREP_PROFILE") or "smoke").strip().lower()
    if profile_name not in RUN_CONFIGS:
        raise ValueError(
            f"Unknown THESIS_PREP_PROFILE='{profile_name}'. Valid: {sorted(RUN_CONFIGS)}"
        )

    cfg = RUN_CONFIGS[profile_name]

    seed = int(os.environ.get("THESIS_PREP_SEED", str(cfg.seed)))
    validation_size = float(os.environ.get("THESIS_PREP_VAL_SIZE", str(cfg.validation_size)))
    learning_rate = float(os.environ.get("THESIS_PREP_LR", str(cfg.learning_rate)))
    epochs = int(os.environ.get("THESIS_PREP_EPOCHS", str(cfg.epochs)))
    batch_size = int(os.environ.get("THESIS_PREP_BATCH", str(cfg.batch_size)))
    max_train_steps = int(os.environ.get("THESIS_PREP_MAX_STEPS", str(cfg.max_train_steps)))
    max_train_samples = int(os.environ.get("THESIS_PREP_MAX_TRAIN_SAMPLES", str(cfg.max_train_samples)))
    max_eval_samples = int(os.environ.get("THESIS_PREP_MAX_EVAL_SAMPLES", str(cfg.max_eval_samples)))
    max_test_samples = int(os.environ.get("THESIS_PREP_MAX_TEST_SAMPLES", str(cfg.max_test_samples)))
    model_limit = int(os.environ.get("THESIS_PREP_MODEL_LIMIT", str(cfg.model_limit)))

    print(
        "Prep profile="
        f"{profile_name} | epochs={epochs}, batch={batch_size}, lr={learning_rate}, "
        f"max_steps={max_train_steps}, train/eval/test limits={max_train_samples}/{max_eval_samples}/{max_test_samples}, "
        f"model_limit={model_limit if model_limit > 0 else 'all'}"
    )

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    model_dirs = _prepare_models()

    local_override = (os.environ.get("THESIS_PREP_DATASET_LOCAL") or "").strip()
    if local_override:
        print(f"Loading local dataset: {local_override}")
    else:
        print(f"Loading dataset: {DATASET_ID}")

    try:
        raw_ds = _load_nemo_dataset()
    except Exception as exc:
        if local_override:
            raise
        active_http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "<unset>"
        active_https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "<unset>"
        active_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "<unset>"
        active_ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or "<unset>"
        ssl_verify_disabled = os.environ.get("THESIS_DISABLE_SSL_VERIFY") or "0"
        raise RuntimeError(
            "Failed to download dataset from Hugging Face. "
            f"Active network config -> HTTP: {active_http_proxy}, HTTPS: {active_https_proxy}, "
            f"NO_PROXY: {active_no_proxy}, CA_BUNDLE: {active_ca_bundle}, "
            f"THESIS_DISABLE_SSL_VERIFY: {ssl_verify_disabled}. "
            "If you are on a corporate network, verify VPN/proxy reachability or set "
            "THESIS_HTTP_PROXY / THESIS_HTTPS_PROXY explicitly. "
            "For certificate issues, set THESIS_CA_BUNDLE to your corporate PEM bundle; "
            "as a last resort, set THESIS_DISABLE_SSL_VERIFY=1. "
            "You can also set THESIS_PREP_DATASET_LOCAL to a local CSV/JSON file and re-run."
        ) from exc

    ds = _prepare_splits(raw_ds, validation_size=validation_size, seed=seed)
    token_col, label_col = _guess_columns(ds)
    label_list = _get_label_list(ds, label_col)
    ds = _ensure_numeric_labels(ds, label_col, label_list)

    print(f"Using columns -> tokens: {token_col}, labels: {label_col}")
    print(f"Label count: {len(label_list)}")

    for split_name in ("train", "validation", "test"):
        split_entity_tokens = 0
        split_token_total = 0
        for example in ds[split_name]:
            labels = example[label_col]
            split_token_total += len(labels)
            split_entity_tokens += sum(1 for label in labels if int(label) != 0)
        print(
            f"Split stats [{split_name}] -> entity tokens: {split_entity_tokens}, total tokens: {split_token_total}"
        )

    test_entity_tokens = 0
    for example in ds["test"]:
        test_entity_tokens += sum(1 for label in example[label_col] if int(label) != 0)
    if test_entity_tokens == 0:
        print(
            "Warning: test split has no entity tokens; entity-level precision/recall/F1 will be 0.0 by definition."
        )

    enabled_specs = [spec for spec in RUN_SPECS if spec.model_id in model_dirs]
    if model_limit > 0:
        enabled_specs = enabled_specs[:model_limit]

    results: list[dict[str, Any]] = []
    for spec in enabled_specs:
        print(f"\nRunning model: {spec.model_id} | train={spec.train}")
        result = _train_and_evaluate(
            model_id=spec.model_id,
            model_dir=model_dirs[spec.model_id],
            train_enabled=spec.train,
            ds=ds,
            token_col=token_col,
            label_col=label_col,
            label_list=label_list,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_train_steps=max_train_steps,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            seed=seed,
        )
        results.append(result)
        print(
            f"Finished {spec.model_id}: "
            f"F1={result['f1']:.4f}, "
            f"P={result['precision']:.4f}, "
            f"R={result['recall']:.4f}, "
            f"true_entity_tokens={result['true_entity_tokens']}, "
            f"pred_entity_tokens={result['pred_entity_tokens']}"
        )

    results_sorted = sorted(results, key=lambda item: item["f1"], reverse=True)
    payload = {
        "date": datetime.now().isoformat(),
        "dataset": DATASET_ID,
        "training": {
            "profile": profile_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_train_steps": max_train_steps,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "max_test_samples": max_test_samples,
            "model_limit": model_limit,
            "seed": seed,
        },
        "results": results_sorted,
    }

    stamp = _now()
    json_out = OUTPUT_DIR / f"ner_model_exploration_{stamp}.json"
    csv_out = OUTPUT_DIR / f"ner_model_exploration_{stamp}.csv"

    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_out.open("w", encoding="utf-8", newline="") as fp:
        fp.write(
            "model,train_enabled,epochs,max_train_steps,samples_train,samples_validation,samples_test,"
            "learning_rate,f1,precision,recall\n"
        )
        for row in results_sorted:
            fp.write(
                f"{row['model']},{row['train_enabled']},{row['epochs']},"
                f"{row['max_train_steps']},{row['samples_train']},{row['samples_validation']},{row['samples_test']},"
                f"{row['learning_rate']},{row['f1']:.6f},{row['precision']:.6f},{row['recall']:.6f}\n"
            )

    print("\n=== Final ranking by F1 ===")
    for idx, row in enumerate(results_sorted, start=1):
        print(f"{idx}. {row['model']} -> F1={row['f1']:.4f}")

    print(f"\nSaved JSON: {json_out}")
    print(f"Saved CSV:  {csv_out}")


if __name__ == "__main__":
    main()
