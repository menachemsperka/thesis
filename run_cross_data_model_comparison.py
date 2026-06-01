"""
run_cross_data_model_comparison.py — Cross-Data × Multi-Model Comparison (Ready Setup)
========================================================================================

Purpose
-------
This runner compares models across split/augmentation conditions using a
ready-results architecture for consistency and fusion experiments.

Execution model:
1. Exp01 and Exp04 are treated as base artifacts (train or reuse).
2. Exp05_ready and all Exp06 ready variants run on top of those artifacts.
3. Base artifact reuse is controlled by ``--base-mode``:
   * ``auto``: reuse if cached, otherwise train
   * ``reuse``: reuse only; fail if cache is missing
   * ``retrain``: retrain base artifacts for each model/condition

Models available in this runner:
* ``dictabert``
* ``berel``
* ``hero``
* ``alephbertgimmel``

Ready experiments:
* ``05_ready``
* ``06_ready``
* ``06_normalized_ready``
* ``06_entropy_ready``
* ``06_learned_ready``
* ``06_ensemble_ready``
* ``06_svm_ready``

Outputs
-------
* ``outputs/cross_comparison/cross_comparison_<timestamp>.xlsx``
* ``outputs/cross_comparison/cross_comparison_latest.xlsx``
* ``outputs/cross_comparison/cross_comparison_<timestamp>.json``
* ``outputs/cross_comparison/cross_comparison_latest.json``
* ``outputs/cross_comparison/cross_comparison_base_ready_index.json``

Usage
-----
Inspect current base cache and exit:

::

    python run_cross_data_model_comparison.py --list-base-cache
    python run_cross_data_model_comparison.py --resume --base-mode reuse

Warm up base artifacts (Exp01 + Exp04) for selected models/conditions:

::

    python run_cross_data_model_comparison.py --experiments 01,04 --models dictabert,berel --base-mode auto

Run ready consistency + all ready fusion variants without retraining:

::

    python run_cross_data_model_comparison.py --experiments 05_ready,06_ready,06_normalized_ready,06_entropy_ready,06_learned_ready,06_ensemble_ready,06_svm_ready --models dictabert,berel --base-mode reuse

Run the default comparison set:

::

    python run_cross_data_model_comparison.py

Resume a previous run:

::

    python run_cross_data_model_comparison.py --resume
    python run_cross_data_model_comparison.py --resume --checkpoint-file outputs/cross_comparison/my_run_checkpoint.json

Environment Variables
---------------------
``THESIS_EXP07_SOURCE``
    Exp07 split artifact policy: ``auto`` (default), ``saved``, ``rerun``.
``THESIS_CROSS_EXPERIMENTS``
    Comma-separated experiment IDs (default:
    ``01,04,05_ready,06_ready,06_svm_ready``).
``THESIS_CROSS_MODELS``
    Comma-separated model keys (default: ``dictabert,berel,hero,alephbertgimmel``).
``THESIS_CROSS_NUM_SEEDS``
    Seed count for exp07/exp08 preparation (default: ``20`` for publication-quality).
``THESIS_SAVE_TRAINED_MODELS``
    Set to ``1`` to save trained models for later reuse (fusion experiments).
``THESIS_CROSS_BASE_MODE``
    Base artifact mode: ``auto`` / ``reuse`` / ``retrain``.
``THESIS_DEBUG``
    Set to ``1`` for verbose subprocess output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None

# ---------------------------------------------------------------------------
# Suppress noisy transformer warnings (BertForMaskedLm unexpected keys, etc.)
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not used.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
try:
    from transformers import logging as _transformers_logging
    _transformers_logging.set_verbosity_error()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXP07_SPLITS_DIR = OUTPUTS_DIR / "exp07" / "splits"
EXP08_SPLITS_DIR = OUTPUTS_DIR / "exp08" / "splits"
EXP07_AUG_SPLITS_DIR = OUTPUTS_DIR / "exp07_augmented" / "splits"
COMPARISON_DIR = OUTPUTS_DIR / "cross_comparison"
DEFAULT_BASE_SEED = 42

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "dictabert": {
        "model_id": "dicta-il/dictabert",
        "display_name": "DictaBERT",
        "description": "General-purpose Hebrew BERT (dicta-il/dictabert)",
    },
    "berel": {
        "model_id": "dicta-il/BEREL_3.0",
        "display_name": "BEREL 3.0",
        "description": "Biblical/Rabbinical Hebrew BERT (dicta-il/BEREL_3.0)",
    },
    "hero": {
        "model_id": "HeNLP/HeRo",
        "display_name": "HeRo",
        "description": "Hebrew RoBERTa-style model (HeNLP/HeRo)",
    },
    "alephbertgimmel": {
        "model_id": "dicta-il/alephbertgimmel-base",
        "display_name": "AlephBERT-Gimmel",
        "description": "AlephBERT-Gimmel base (dicta-il/alephbertgimmel-base)",
    },
}

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXP_NAMES: dict[str, str] = {
    "01": "Regular NER",
    "03": "AUC-2T",
    "04": "AUC Cascaded Pipeline",
    "05_ready": "AUC Cascaded Step-3 Consistency (Ready)",
    # Ready variants (no retraining — read from Exp01 + Exp04)
    "06_ready": "Fusion Regular+Cascaded (Ready)",
    "06_normalized_ready": "Fusion Calibrated (Ready)",
    "06_entropy_ready": "Fusion Entropy (Ready)",
    "06_learned_ready": "Fusion Learned Weights (Ready)",
    "06_ensemble_ready": "Fusion Ensemble Rules (Ready)",
    "06_svm_ready": "SVM Router Fusion (Ready)",
}

EXP_SCRIPTS: dict[str, str] = {
    "01": "experiment_01_regular_ner",
    "03": "experiment_03_auc_2t",
    "04": "experiment_04_auc_cascaded_pipeline",
    "05_ready": "experiment_05_ready",
    # Ready variants (no retraining — read from Exp01 + Exp04)
    "06_ready": "experiment_06_fusion_ready",
    "06_normalized_ready": "experiment_06_fusion_normalized_ready",
    "06_entropy_ready": "experiment_06_fusion_entropy_ready",
    "06_learned_ready": "experiment_06_fusion_learned_weights_ready",
    "06_ensemble_ready": "experiment_06_fusion_ensemble_rules_ready",
    "06_svm_ready": "experiment_06_fusion_svm_ready",
    "07": "experiment_07_sentence_split_strategy",
    "08": "experiment_08_llm_augmentation",
}

# Ready experiments that depend on Exp01/Exp04 artifacts.
READY_DEPENDENT_EXP_IDS: set[str] = {
    "05_ready",
    "06_ready",
    "06_normalized_ready",
    "06_entropy_ready",
    "06_learned_ready",
    "06_ensemble_ready",
    "06_svm_ready",
}

# Experiments that require expensive GPU training (vs cheap inference).
TRAINING_EXP_IDS: set[str] = {"01", "03", "04"}

# Drop the non-paper multilabel stratified split variants from cross-comparison.
EXCLUDED_CONDITION_KEYS: set[str] = {
    "exp07_after_multilabel_stratified",
    "exp07aug_after_multilabel_stratified",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


_EXPERIMENT_MODULES: dict[str, Any] = {}


def _import_experiment(exp_id: str):
    """Lazily import (or reload) an experiment module."""
    module_name = EXP_SCRIPTS.get(exp_id)
    if module_name is None:
        raise ValueError(f"Unknown experiment ID: {exp_id}")
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    mod = importlib.import_module(module_name)
    _EXPERIMENT_MODULES[exp_id] = mod
    return mod


def _set_presplit_env(train_json: Path, eval_json: Path) -> None:
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(train_json)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(eval_json)


def _clear_presplit_env() -> None:
    os.environ.pop("THESIS_PRESPLIT_TRAIN_JSON", None)
    os.environ.pop("THESIS_PRESPLIT_EVAL_JSON", None)


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a HF model ID to a local directory path if available."""
    # Check built-in model dir (models/<short_name>)
    short_name = model_id.split("/")[-1]
    builtin = MODELS_DIR / short_name
    if (builtin / "config.json").exists():
        return str(builtin)

    # Check HF cache dir (models/hf_models/<org>__<name>)
    cache_name = model_id.replace("/", "__")
    cached = MODELS_DIR / "hf_models" / cache_name
    if (cached / "config.json").exists():
        return str(cached)

    # Fallback to HF model ID (will attempt download)
    return model_id


def _set_model_env(model_id: str) -> None:
    """Point all model-resolution logic to the requested model (local-first)."""
    resolved = _resolve_local_model_path(model_id)
    os.environ["THESIS_MODEL_NAME"] = resolved
    if Path(resolved).exists():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["THESIS_MODEL_LOCAL_ONLY"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("THESIS_MODEL_LOCAL_ONLY", None)


def _extract_metrics(payload: dict) -> dict:
    return {
        "f1": _to_float(payload.get("f1")),
        "precision": _to_float(payload.get("precision")),
        "recall": _to_float(payload.get("recall")),
        "status": payload.get("status", "ok"),
    }


def _conditions_from_rows(rows: list[dict]) -> list[dict[str, Any]]:
    """Build condition metadata from checkpoint rows for rebuild-only mode."""
    by_key: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = str(r.get("condition_key", "")).strip()
        if not key:
            continue
        if key in by_key:
            continue
        by_key[key] = {
            "source": str(r.get("data_source", "unknown")).strip(),
            "key": key,
            "label": str(r.get("condition_label", key)).strip(),
            "short_label": str(r.get("condition_short", key)).strip(),
            "description": str(r.get("condition_description", "")).strip(),
            "is_baseline": bool(r.get("is_baseline", False)),
        }
    return [by_key[k] for k in sorted(by_key.keys())]


def _run_key(model_id: str, exp_id: str, condition_key: str) -> str:
    return f"{model_id}||exp{exp_id}||{condition_key}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _save_progress_checkpoint(
    checkpoint_path: Path,
    rows: list[dict],
    models: list[dict[str, str]],
    experiment_ids: list[str],
    conditions: list[dict[str, Any]],
    prep07: dict[str, Any],
    prep08: dict[str, Any],
    run_counter: int,
    total_runs: int,
    started_at: str,
) -> None:
    payload = {
        "name": "cross_comparison_progress_checkpoint",
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(),
        "run_counter": run_counter,
        "total_runs": total_runs,
        "models": [m.get("model_id") for m in models],
        "experiments": experiment_ids,
        "condition_keys": [c.get("key") for c in conditions],
        "exp07_preparation": prep07,
        "exp08_preparation": prep08,
        "rows": rows,
    }
    _atomic_write_json(checkpoint_path, payload)


def _base_cache_key(model_id: str, condition: dict[str, Any]) -> str:
    train_path = str(Path(condition["train_path"]).resolve())
    eval_path = str(Path(condition["eval_path"]).resolve())
    return f"{model_id}||{condition['key']}||{train_path}||{eval_path}"


def _load_base_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_base_index(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    payload = {
        "name": "cross_comparison_base_ready_index",
        "updated_at": datetime.now().isoformat(),
        "entries": entries,
    }
    _atomic_write_json(path, payload)


def _print_base_cache_summary(path: Path) -> int:
    """Print a concise summary of cached base artifacts and return entry count."""
    entries = _load_base_index(path)
    if not entries:
        print("Base cache index is empty or missing.")
        print(f"Path: {path}")
        return 0

    valid_rows: list[dict[str, Any]] = []
    invalid_count = 0
    for _, entry in entries.items():
        if isinstance(entry, dict) and _is_valid_base_entry(entry):
            valid_rows.append(entry)
        else:
            invalid_count += 1

    print("Base cache summary")
    print(f"Path: {path}")
    print(f"Entries total: {len(entries)}")
    print(f"Entries valid: {len(valid_rows)}")
    print(f"Entries invalid/missing files: {invalid_count}")

    if not valid_rows:
        return 0

    by_model: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    for row in valid_rows:
        model_id = str(row.get("model_id", "unknown")).strip() or "unknown"
        condition_key = str(row.get("condition_key", "unknown")).strip() or "unknown"
        by_model[model_id] = by_model.get(model_id, 0) + 1
        by_condition[condition_key] = by_condition.get(condition_key, 0) + 1

    print("\nValid entries by model:")
    for model_id in sorted(by_model.keys()):
        print(f"  {model_id}: {by_model[model_id]}")

    print("\nSample valid entries (first 10):")
    for row in valid_rows[:10]:
        model_id = str(row.get("model_id", ""))
        condition_key = str(row.get("condition_key", ""))
        updated_at = str(row.get("updated_at", ""))
        print(f"  model={model_id} | condition={condition_key} | updated={updated_at}")

    return len(valid_rows)


def _missing_base_cache_entries(
    models: list[dict[str, str]],
    conditions: list[dict[str, Any]],
    base_index: dict[str, dict[str, Any]],
) -> list[str]:
    """Return human-readable list of missing/invalid base cache entries."""
    missing: list[str] = []
    for model_info in models:
        model_id = model_info["model_id"]
        model_display = model_info["display_name"]
        for cond in conditions:
            key = _base_cache_key(model_id, cond)
            entry = base_index.get(key)
            if not (isinstance(entry, dict) and _is_valid_base_entry(entry)):
                missing.append(f"{model_display} / {cond['short_label']}")
    return missing


def _extract_timestamp_token(text: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", str(text))
    return m.group(1) if m else ""


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_model_id(model_ref: str) -> str | None:
    v = str(model_ref or "").strip().replace("\\", "/").lower()
    if not v:
        return None

    if "dicta-il/dictabert" in v or v.endswith("/models/dictabert") or "__dictabert" in v:
        return "dicta-il/dictabert"
    if "dicta-il/berel_3.0" in v or "__berel_3.0" in v:
        return "dicta-il/BEREL_3.0"
    if "henlp/hero" in v or "__hero" in v:
        return "HeNLP/HeRo"
    if "dicta-il/alephbertgimmel-base" in v or "__alephbertgimmel-base" in v:
        return "dicta-il/alephbertgimmel-base"

    for info in MODEL_REGISTRY.values():
        mid = info["model_id"]
        if mid.lower() == v:
            return mid
    return None


def _condition_key_from_split_strategy(split_strategy: str) -> str | None:
    txt = str(split_strategy or "")
    m = re.search(r"\(([^)]+)\)", txt)
    marker = m.group(1).strip() if m else ""
    if not marker:
        return None

    if marker.endswith("_augmented_train"):
        variant = marker[: -len("_augmented_train")]
        return f"exp07aug_{variant}"
    if marker.endswith("_train"):
        variant = marker[: -len("_train")]
        if variant == "baseline":
            return "exp08_baseline"
        if variant == "augmented":
            return "exp08_augmented"
        return f"exp07_{variant}"
    return None


def _bootstrap_from_cross_comparison_rows(
    *,
    conditions: list[dict[str, Any]],
    allowed_model_ids: set[str],
    base_index: dict[str, dict[str, Any]],
) -> int:
    """Populate base index entries from prior cross_comparison JSON rows."""
    by_condition = {str(c["key"]): c for c in conditions}
    cc_dir = COMPARISON_DIR
    if not cc_dir.exists():
        return 0

    # Collect latest successful exp01/exp04 rows per (model_id, condition_key).
    rows_by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for f in sorted(cc_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if f.name in {
            "cross_comparison_base_ready_index.json",
            "cross_comparison_progress_latest.json",
        }:
            continue
        payload = _safe_load_json(f)
        if not payload:
            continue
        rows = payload.get("results")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "")).strip().lower()
            if status != "ok" and not status.startswith("ok_"):
                continue
            exp_id = str(row.get("experiment_id", "")).strip()
            if exp_id not in {"exp01", "exp04"}:
                continue
            model_id = str(row.get("model_id", "")).strip()
            if model_id not in allowed_model_ids:
                continue
            condition_key = str(row.get("condition_key", "")).strip()
            if condition_key not in by_condition:
                continue
            pair_key = (model_id, condition_key)
            if pair_key not in rows_by_pair:
                rows_by_pair[pair_key] = {}
            rows_by_pair[pair_key][exp_id] = row

    added = 0
    for (model_id, condition_key), pair in rows_by_pair.items():
        r01 = pair.get("exp01")
        r04 = pair.get("exp04")
        if not (isinstance(r01, dict) and isinstance(r04, dict)):
            continue

        exp01_metrics = str(r01.get("metrics_file", "")).strip()
        exp01_result = str(r01.get("result_file", "")).strip()
        exp04_metrics = str(r04.get("metrics_file", "")).strip()
        exp04_result = str(r04.get("result_file", "")).strip()
        if not all([exp01_metrics, exp01_result, exp04_metrics, exp04_result]):
            continue

        cond = by_condition[condition_key]
        entry = {
            "model_id": model_id,
            "condition_key": condition_key,
            "condition_label": cond.get("label"),
            "train_path": str(cond["train_path"]),
            "eval_path": str(cond["eval_path"]),
            "updated_at": datetime.now().isoformat(),
            "exp01_metrics_file": exp01_metrics,
            "exp01_result_file": exp01_result,
            "exp04_metrics_file": exp04_metrics,
            "exp04_result_file": exp04_result,
            "source": "bootstrap_cross_comparison_rows",
        }
        if not _is_valid_base_entry(entry):
            continue

        cache_key = _base_cache_key(model_id, cond)
        if cache_key not in base_index:
            base_index[cache_key] = entry
            added += 1

    return added


def _bootstrap_from_standalone_exp_outputs(
    *,
    conditions: list[dict[str, Any]],
    allowed_model_ids: set[str],
    base_index: dict[str, dict[str, Any]],
) -> int:
    """Heuristic fallback: pair exp01 condition-tagged files with nearest exp04 by model/time."""
    by_condition = {str(c["key"]): c for c in conditions}

    exp01_dir = OUTPUTS_DIR / "exp01"
    exp04_dir = OUTPUTS_DIR / "exp04"
    if not exp01_dir.exists() or not exp04_dir.exists():
        return 0

    exp01_latest: dict[tuple[str, str], dict[str, Any]] = {}
    for f in exp01_dir.glob("*.json"):
        payload = _safe_load_json(f)
        if not payload or str(payload.get("experiment_id", "")) != "exp01":
            continue
        if str(payload.get("status", "")).strip().lower() != "ok":
            continue

        model_id = _normalize_model_id(str(payload.get("model", "")))
        if model_id not in allowed_model_ids:
            continue

        split_strategy = ""
        tp = payload.get("training_parameters")
        if isinstance(tp, dict):
            split_strategy = str(tp.get("split_strategy", ""))
        condition_key = _condition_key_from_split_strategy(split_strategy)
        if condition_key not in by_condition:
            continue

        metrics_file = str(payload.get("metrics_file", "")).strip()
        result_file = str(payload.get("result_file", "")).strip()
        if not metrics_file or not result_file:
            continue

        ts = _extract_timestamp_token(f.name) or _extract_timestamp_token(result_file)
        candidate = {
            "ts": ts,
            "metrics_file": metrics_file,
            "result_file": result_file,
        }
        k = (model_id, condition_key)
        prev = exp01_latest.get(k)
        if prev is None or candidate["ts"] >= prev["ts"]:
            exp01_latest[k] = candidate

    exp04_by_model: dict[str, list[dict[str, Any]]] = {}
    for f in exp04_dir.glob("*.json"):
        payload = _safe_load_json(f)
        if not payload or str(payload.get("experiment_id", "")) != "exp04":
            continue
        if str(payload.get("status", "")).strip().lower() != "ok":
            continue

        model_id = _normalize_model_id(str(payload.get("model", "")))
        if model_id not in allowed_model_ids:
            continue

        metrics_file = str(payload.get("metrics_file", "")).strip()
        result_file = str(payload.get("result_file", "")).strip()
        if not metrics_file or not result_file:
            continue

        ts = _extract_timestamp_token(f.name) or _extract_timestamp_token(result_file)
        exp04_by_model.setdefault(model_id, []).append(
            {
                "ts": ts,
                "metrics_file": metrics_file,
                "result_file": result_file,
            }
        )

    for model_id in list(exp04_by_model.keys()):
        exp04_by_model[model_id] = sorted(exp04_by_model[model_id], key=lambda x: x["ts"])

    used_exp04: set[tuple[str, str, str]] = set()
    added = 0

    for (model_id, condition_key), e01 in sorted(exp01_latest.items()):
        cond = by_condition[condition_key]
        cache_key = _base_cache_key(model_id, cond)
        if cache_key in base_index and _is_valid_base_entry(base_index[cache_key]):
            continue

        candidates = exp04_by_model.get(model_id, [])
        if not candidates:
            continue

        # Choose closest timestamp exp04 not used yet for this model.
        target_ts = e01.get("ts", "")
        best = None
        best_score = None
        for c in candidates:
            marker = (model_id, c.get("metrics_file", ""), c.get("result_file", ""))
            if marker in used_exp04:
                continue
            cts = c.get("ts", "")
            score = abs(int(cts.replace("_", "")) - int(target_ts.replace("_", ""))) if cts and target_ts else 10**18
            if best is None or score < best_score:
                best = c
                best_score = score

        if not best:
            continue

        used_exp04.add((model_id, best.get("metrics_file", ""), best.get("result_file", "")))
        entry = {
            "model_id": model_id,
            "condition_key": condition_key,
            "condition_label": cond.get("label"),
            "train_path": str(cond["train_path"]),
            "eval_path": str(cond["eval_path"]),
            "updated_at": datetime.now().isoformat(),
            "exp01_metrics_file": e01["metrics_file"],
            "exp01_result_file": e01["result_file"],
            "exp04_metrics_file": best["metrics_file"],
            "exp04_result_file": best["result_file"],
            "source": "bootstrap_standalone_outputs_heuristic",
        }
        if not _is_valid_base_entry(entry):
            continue
        base_index[cache_key] = entry
        added += 1

    return added


def _bootstrap_base_cache_from_saved_outputs(
    *,
    models: list[dict[str, str]],
    conditions: list[dict[str, Any]],
    base_index: dict[str, dict[str, Any]],
    base_index_path: Path,
) -> dict[str, int]:
    """Best-effort cache hydration from existing outputs without retraining."""
    allowed_model_ids = {m["model_id"] for m in models}
    added_from_cc = _bootstrap_from_cross_comparison_rows(
        conditions=conditions,
        allowed_model_ids=allowed_model_ids,
        base_index=base_index,
    )
    added_from_outputs = _bootstrap_from_standalone_exp_outputs(
        conditions=conditions,
        allowed_model_ids=allowed_model_ids,
        base_index=base_index,
    )

    if added_from_cc or added_from_outputs:
        _save_base_index(base_index_path, base_index)

    return {
        "added_from_cross_comparison": added_from_cc,
        "added_from_standalone_outputs": added_from_outputs,
        "added_total": added_from_cc + added_from_outputs,
    }


def _backfill_rows_from_history_json(
    *,
    rows: list[dict[str, Any]],
    models: list[dict[str, str]],
    experiment_ids: list[str],
    conditions: list[dict[str, Any]],
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Backfill missing rows from prior cross_comparison JSON outputs.

    This is useful for ``--resume`` when a checkpoint is partial but historical
    cross-comparison JSON files contain successful runs for the same
    (model, experiment, condition) tuples.
    """
    selected_model_ids = {m["model_id"] for m in models}
    selected_experiments = {f"exp{e}" for e in experiment_ids}
    selected_conditions = {str(c["key"]) for c in conditions}

    by_key: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        mk = str(r.get("model_id", "")).strip()
        exp_id = str(r.get("experiment_id", "")).strip().replace("exp", "")
        cond_key = str(r.get("condition_key", "")).strip()
        if not (mk and exp_id and cond_key):
            continue
        by_key[_run_key(mk, exp_id, cond_key)] = r

    json_files = sorted(
        COMPARISON_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    added = 0
    for jf in json_files:
        if jf == checkpoint_path:
            continue
        if jf.name in {
            "cross_comparison_base_ready_index.json",
            "cross_comparison_progress_latest.json",
        }:
            continue

        payload = _safe_load_json(jf)
        if not payload:
            continue
        history_rows = payload.get("results")
        if not isinstance(history_rows, list):
            continue

        for hr in history_rows:
            if not isinstance(hr, dict):
                continue

            model_id = str(hr.get("model_id", "")).strip()
            exp_id = str(hr.get("experiment_id", "")).strip()
            cond_key = str(hr.get("condition_key", "")).strip()
            if model_id not in selected_model_ids:
                continue
            if exp_id not in selected_experiments:
                continue
            if cond_key not in selected_conditions:
                continue

            run_key = _run_key(model_id, exp_id.replace("exp", ""), cond_key)
            existing = by_key.get(run_key)
            if existing is None:
                by_key[run_key] = hr
                added += 1
                continue

            existing_status = str(existing.get("status", "")).strip().lower()
            new_status = str(hr.get("status", "")).strip().lower()

            # Prefer successful rows over errors; otherwise keep the newer row
            # due chronological file iteration (later files overwrite earlier).
            existing_ok = not existing_status.startswith("error")
            new_ok = not new_status.startswith("error")
            if new_ok and not existing_ok:
                by_key[run_key] = hr
            elif new_ok and existing_ok:
                by_key[run_key] = hr

    return list(by_key.values()), added


def _is_valid_base_entry(entry: dict[str, Any]) -> bool:
    needed = [
        "exp01_metrics_file",
        "exp01_result_file",
        "exp04_metrics_file",
        "exp04_result_file",
    ]
    for k in needed:
        p = str(entry.get(k, "")).strip()
        if not p or not Path(p).exists():
            return False
    return True


def _load_result_payload(result_file: str) -> dict[str, Any]:
    p = Path(result_file)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid result payload: {p}")
    return raw


def _set_ready_env(exp01_metrics_file: str, exp04_metrics_file: str) -> None:
    os.environ["THESIS_READY_EXP01_XLSX"] = exp01_metrics_file
    os.environ["THESIS_READY_EXP04_XLSX"] = exp04_metrics_file


def _clear_ready_env() -> None:
    os.environ.pop("THESIS_READY_EXP01_XLSX", None)
    os.environ.pop("THESIS_READY_EXP04_XLSX", None)


def _ensure_base_artifacts(
    *,
    model_id: str,
    model_display: str,
    condition: dict[str, Any],
    base_mode: str,
    base_mem: dict[str, dict[str, Any]],
    base_index: dict[str, dict[str, Any]],
    base_index_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Ensure Exp01/Exp04 artifacts exist for (model, condition).

    Returns (entry, reused_existing).
    """
    key = _base_cache_key(model_id, condition)

    # Always reuse already-materialized artifacts in this process to avoid
    # repeating expensive base training within one invocation.
    if key in base_mem and _is_valid_base_entry(base_mem[key]):
        return base_mem[key], True

    existing = base_index.get(key)
    if base_mode in {"auto", "reuse"} and isinstance(existing, dict) and _is_valid_base_entry(existing):
        base_mem[key] = existing
        return existing, True

    if base_mode == "reuse":
        raise RuntimeError(
            "Base mode is 'reuse' but no valid cached Exp01/Exp04 artifacts were found for "
            f"{model_display} / {condition.get('short_label', condition.get('key', 'unknown condition'))}."
        )

    _log(
        f"Preparing base artifacts (exp01 + exp04) | {model_display} | "
        f"{condition.get('short_label', condition.get('key', 'condition'))}"
    )

    _set_presplit_env(condition["train_path"], condition["eval_path"])
    # Set context for model saving (used by th_functions.py)
    os.environ["THESIS_CURRENT_CONDITION_KEY"] = str(condition.get("key", "default"))
    try:
        os.environ["THESIS_CURRENT_EXP_ID"] = "exp01"
        payload01 = _import_experiment("01").run()
        os.environ["THESIS_CURRENT_EXP_ID"] = "exp04"
        payload04 = _import_experiment("04").run()
    finally:
        _clear_presplit_env()
        os.environ.pop("THESIS_CURRENT_EXP_ID", None)
        os.environ.pop("THESIS_CURRENT_CONDITION_KEY", None)

    entry = {
        "model_id": model_id,
        "condition_key": condition.get("key"),
        "condition_label": condition.get("label"),
        "train_path": str(condition["train_path"]),
        "eval_path": str(condition["eval_path"]),
        "updated_at": datetime.now().isoformat(),
        "exp01_metrics_file": str(payload01.get("metrics_file", "")),
        "exp01_result_file": str(payload01.get("result_file", "")),
        "exp04_metrics_file": str(payload04.get("metrics_file", "")),
        "exp04_result_file": str(payload04.get("result_file", "")),
    }

    if not _is_valid_base_entry(entry):
        raise RuntimeError(
            "Failed to materialize valid base artifacts (exp01/exp04). "
            "Ensure both experiments completed successfully."
        )

    base_mem[key] = entry
    base_index[key] = entry
    _save_base_index(base_index_path, base_index)
    return entry, False


# ---------------------------------------------------------------------------
# Exp07 split preparation (reuses same logic as run_split_comparison.py)
# ---------------------------------------------------------------------------
def _exp07_artifacts_ready() -> tuple[bool, str]:
    meta_path = EXP07_SPLITS_DIR / "split_meta.json"
    if not meta_path.exists():
        return False, f"Missing: {meta_path}"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Unreadable split_meta.json: {exc}"
    variants = meta.get("variants")
    if not isinstance(variants, list) or not variants:
        return False, "No variants in split_meta.json"

    # In Colab mode we require exp07 metadata to explicitly confirm full seed coverage
    # before allowing downstream exp07+augmentation generation.
    is_colab = os.environ.get("THESIS_RUN_ENV") == "colab"
    if is_colab:
        seeds_raw = (
            os.environ.get("THESIS_EXP07_NUM_SEEDS")
            or os.environ.get("THESIS_DIRECT_SPLIT_RUNS")
            or "20"
        ).strip()
        try:
            expected_num_seeds = max(2, int(seeds_raw))
        except ValueError:
            expected_num_seeds = 20

        actual_num_seeds = meta.get("num_seeds")
        if not isinstance(actual_num_seeds, int):
            return False, (
                "Exp07 split metadata missing 'num_seeds' in Colab mode; "
                "rerun exp07 to generate complete multi-seed artifacts"
            )
        if actual_num_seeds < expected_num_seeds:
            return False, (
                f"Exp07 splits incomplete for Colab mode: expected {expected_num_seeds} seeds, "
                f"found {actual_num_seeds}"
            )

    for vm in variants:
        tf = vm.get("train_file")
        ef = vm.get("eval_file")
        if not tf or not ef:
            return False, f"Variant entry missing files: {vm}"
        if not (EXP07_SPLITS_DIR / tf).exists() or not (EXP07_SPLITS_DIR / ef).exists():
            return False, f"Missing split file for variant {vm.get('variant')}"
    return True, "ok"


def _prepare_exp07_splits(source: str) -> dict[str, Any]:
    mode = (source or "auto").strip().lower()
    if mode not in {"saved", "rerun", "auto"}:
        raise ValueError(f"Invalid exp07 source: {source}")
    ready, reason = _exp07_artifacts_ready()
    if mode == "saved":
        if not ready:
            raise FileNotFoundError(f"Exp07 artifacts not ready: {reason}")
        _log("Using saved exp07 split artifacts.")
        return {"source": "saved", "reran_exp07": False}
    if mode == "auto" and ready:
        _log("Exp07 split artifacts valid; reusing.")
        return {"source": "saved", "reran_exp07": False}
    _log("Running experiment 07 to generate split artifacts...")

    # Keep regular mode untouched; in Colab default to 20 seeds unless explicitly set.
    if os.environ.get("THESIS_RUN_ENV") == "colab":
        desired_raw = (
            os.environ.get("THESIS_EXP07_NUM_SEEDS")
            or os.environ.get("THESIS_DIRECT_SPLIT_RUNS")
            or "20"
        ).strip()
        os.environ["THESIS_EXP07_NUM_SEEDS"] = desired_raw
        _log(f"Colab exp07 seed target: {desired_raw}")

    t0 = time.time()
    mod07 = _import_experiment("07")
    mod07.run()
    elapsed = time.time() - t0
    ok, msg = _exp07_artifacts_ready()
    if not ok:
        raise RuntimeError(f"Exp07 finished but artifacts incomplete: {msg}")
    _log(f"Experiment 07 done in {elapsed:.1f}s.")
    return {"source": "rerun", "reran_exp07": True}


def _load_exp07_meta() -> dict:
    path = EXP07_SPLITS_DIR / "split_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Exp08 split preparation (reuses same logic as run_experiments_with_exp08_data)
# ---------------------------------------------------------------------------
def _exp08_artifacts_ready() -> tuple[bool, str]:
    required = ["baseline_train.json", "baseline_eval.json",
                 "augmented_train.json", "augmented_eval.json", "split_meta.json"]
    for name in required:
        if not (EXP08_SPLITS_DIR / name).exists():
            return False, f"Missing: {name}"
    return True, "ok"


def _prepare_exp08_splits(force_rerun: bool = False) -> dict[str, Any]:
    ready, reason = _exp08_artifacts_ready()
    if not force_rerun and ready:
        _log("Using saved exp08 split artifacts.")
        return {"source": "saved", "reran_exp08": False}

    # For cross-comparison we only need split artifacts, not exp08 metric benchmarking.
    # Build baseline/augmented split files directly to avoid expensive per-seed training.
    _log("Generating exp08 split artifacts (no exp08 training)...")
    t0 = time.time()

    mod08 = _import_experiment("08")

    # Resolve dataset and model environment through common helpers.
    from common import configure_model_environment, resolve_dataset, suppress_output_if_needed
    from NERtraining import PrepDataSetNERTraining

    split_seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    multiplier_raw = (os.environ.get("THESIS_EXP08_MULTIPLIER") or "3").strip()
    try:
        split_seed = int(split_seed_raw)
    except ValueError:
        split_seed = 42
    try:
        multiplier = max(1, int(multiplier_raw))
    except ValueError:
        multiplier = 3

    dataset_path = resolve_dataset("ner_dataset.csv")
    model_name, _ = configure_model_environment()
    augmentation_model = mod08._resolve_augmentation_model_name(model_name)

    worker = PrepDataSetNERTraining()
    with suppress_output_if_needed():
        data_df = worker.load_and_prepare_data(str(dataset_path))
        sentences = mod08.tf.train_data_fit(data_df)

    baseline_train, baseline_eval = mod08.tf.split_list(
        sentences,
        split_ratio=0.7,
        seed=split_seed,
        ensure_label_coverage=True,
    )

    with suppress_output_if_needed():
        generated_sents, _ = mod08._augment_training_data(
            baseline_train,
            data_df,
            model_name,
            multiplier=multiplier,
            rng_seed=split_seed,
            augmentation_model_name=augmentation_model,
        )
    augmented_train = baseline_train + generated_sents

    mod08._save_exp08_splits(
        baseline_train=baseline_train,
        baseline_eval=baseline_eval,
        augmented_train=augmented_train,
        seed=split_seed,
        multiplier=multiplier,
        augmentation_model=augmentation_model,
        augmented_f1_mean=None,
        baseline_f1_mean=None,
    )

    elapsed = time.time() - t0
    ok, msg = _exp08_artifacts_ready()
    if not ok:
        raise RuntimeError(f"Exp08 finished but artifacts incomplete: {msg}")
    _log(
        "Exp08 split artifacts generated in "
        f"{elapsed:.1f}s (train={len(baseline_train)}, generated={len(generated_sents)})."
    )
    return {"source": "rerun", "reran_exp08": True}


def _load_exp08_meta() -> dict:
    path = EXP08_SPLITS_DIR / "split_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Combined exp07 + augmentation preparation
# ---------------------------------------------------------------------------
def _exp07_augmented_ready() -> tuple[bool, str]:
    """Check whether augmented versions of exp07 splits already exist."""
    meta_path = EXP07_AUG_SPLITS_DIR / "split_meta.json"
    if not meta_path.exists():
        return False, f"Missing: {meta_path}"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Unreadable split_meta.json: {exc}"
    variants = meta.get("variants")
    if not isinstance(variants, list) or not variants:
        return False, "No variants in split_meta.json"

    seeds_raw = (os.environ.get("THESIS_EXP07_NUM_SEEDS") or os.environ.get("THESIS_DIRECT_SPLIT_RUNS") or "20").strip()
    try:
        num_seeds = max(2, int(seeds_raw))
    except ValueError:
        num_seeds = 20
    expected_seeds = [DEFAULT_BASE_SEED + i for i in range(num_seeds)]

    for vm in variants:
        seed_files = vm.get("seed_files")
        if not isinstance(seed_files, dict) or not seed_files:
            return False, f"Variant missing seed_files map: {vm.get('variant')}"
        for seed in expected_seeds:
            entry = seed_files.get(str(seed))
            if not isinstance(entry, dict):
                return False, f"Missing seed-specific files for variant {vm.get('variant')} seed={seed}"
            tf = entry.get("train_file")
            ef = entry.get("eval_file")
            if not tf or not ef:
                return False, f"Variant seed entry missing files: variant={vm.get('variant')} seed={seed}"
            if not (EXP07_AUG_SPLITS_DIR / tf).exists() or not (EXP07_AUG_SPLITS_DIR / ef).exists():
                return False, f"Missing augmented split file for variant {vm.get('variant')} seed={seed}"
    return True, "ok"


def _prepare_exp07_augmented_splits(force_rerun: bool = False) -> dict[str, Any]:
    """Apply exp08-style augmentation to each exp07 variant's training data.

    For every exp07 split variant, loads its training sentences, runs the
    LLM mask-filling augmentation from experiment_08, and saves the
    augmented training set alongside the original eval set.

    Results are cached in ``outputs/exp07_augmented/splits/``.
    """
    ready, _ = _exp07_augmented_ready()
    if not force_rerun and ready:
        _log("Using saved exp07+augmentation split artifacts.")
        try:
            meta_saved = _load_exp07_augmented_meta()
            aug_model = meta_saved.get("augmentation_model", "")
        except Exception:
            aug_model = ""
        return {"source": "saved", "generated": False, "augmentation_model": aug_model}

    _log("Generating augmented versions of exp07 splits...")
    t0 = time.time()

    # Import augmentation machinery from experiment 08
    exp08_mod = _import_experiment("08")
    augment_fn = exp08_mod._augment_training_data
    resolve_aug_model_fn = getattr(exp08_mod, "_resolve_augmentation_model_name", None)

    # We need the full DataFrame for entity-token lookup during augmentation
    from split_io import load_split, save_split
    from common import resolve_dataset, configure_model_environment, suppress_output_if_needed
    from NERtraining import PrepDataSetNERTraining

    dataset_path = resolve_dataset("ner_dataset.csv")
    worker = PrepDataSetNERTraining()
    data_df = worker.load_and_prepare_data(str(dataset_path))
    model_name, _ = configure_model_environment()
    augmentation_model_name = (
        resolve_aug_model_fn(model_name) if callable(resolve_aug_model_fn) else model_name
    )

    multiplier_raw = (os.environ.get("THESIS_EXP08_MULTIPLIER") or "3").strip()
    try:
        multiplier = max(1, int(multiplier_raw))
    except ValueError:
        multiplier = 3

    seeds_raw = (os.environ.get("THESIS_EXP07_NUM_SEEDS") or os.environ.get("THESIS_DIRECT_SPLIT_RUNS") or "20").strip()
    try:
        num_seeds = max(2, int(seeds_raw))
    except ValueError:
        num_seeds = 20
    seed_list = [DEFAULT_BASE_SEED + i for i in range(num_seeds)]

    meta07 = _load_exp07_meta()
    EXP07_AUG_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    aug_variants: list[dict[str, Any]] = []
    all_variants = [
        vm
        for vm in list(meta07.get("variants", []))
        if f"exp07_{vm.get('variant', '')}" not in EXCLUDED_CONDITION_KEYS
    ]
    total_aug_jobs = len(all_variants) * len(seed_list)
    aug_job_idx = 0

    _log(
        "Generating seed-specific augmentation datasets: "
        f"{len(all_variants)} variants x {len(seed_list)} seeds = {total_aug_jobs} jobs"
    )

    for variant_idx, vm in enumerate(all_variants, start=1):
        variant_name = vm["variant"]
        train_path = EXP07_SPLITS_DIR / vm["train_file"]
        eval_path = EXP07_SPLITS_DIR / vm["eval_file"]

        if not train_path.exists() or not eval_path.exists():
            _log(f"WARNING: skipping augmentation for exp07 variant {variant_name} (missing files)")
            continue

        _log(
            f"Augmenting exp07 variant {variant_idx}/{len(all_variants)}: {vm['label']} "
            f"({len(seed_list)} seeds)"
        )
        train_sentences = load_split(train_path)
        eval_sentences = load_split(eval_path)
        seed_files: dict[str, dict[str, Any]] = {}
        first_seed_train_file = ""
        first_seed_eval_file = ""
        for seed_idx, seed in enumerate(seed_list, start=1):
            aug_job_idx += 1
            aug_train_file = f"{variant_name}_seed{seed}_augmented_train.json"
            aug_eval_file = f"{variant_name}_seed{seed}_eval.json"
            aug_train_path = EXP07_AUG_SPLITS_DIR / aug_train_file
            aug_eval_path = EXP07_AUG_SPLITS_DIR / aug_eval_file

            if aug_train_path.exists() and aug_eval_path.exists():
                _log(
                    f"  Augmentation skipped {aug_job_idx}/{total_aug_jobs} (already exists) | "
                    f"variant {variant_idx}/{len(all_variants)} | seed {seed_idx}/{len(seed_list)} (s{seed})"
                )
                try:
                    augmented_train = load_split(aug_train_path)
                    generated_sents_len = len(augmented_train) - len(train_sentences)
                except Exception:
                    generated_sents_len = 0
                seed_files[str(seed)] = {
                    "seed": int(seed),
                    "train_file": aug_train_file,
                    "eval_file": aug_eval_file,
                    "original_train_sentences": len(train_sentences),
                    "generated_sentences": generated_sents_len,
                    "augmented_train_sentences": len(train_sentences) + generated_sents_len,
                }
                if not first_seed_train_file:
                    first_seed_train_file = aug_train_file
                    first_seed_eval_file = aug_eval_file
                continue

            _log(
                f"  Augmentation progress {aug_job_idx}/{total_aug_jobs} | "
                f"variant {variant_idx}/{len(all_variants)} | seed {seed_idx}/{len(seed_list)} (s{seed})"
            )
            with suppress_output_if_needed():
                generated_sents, _ = augment_fn(
                    train_sentences,
                    data_df,
                    model_name,
                    multiplier=multiplier,
                    rng_seed=int(seed),
                )
            augmented_train = train_sentences + generated_sents

            save_split(augmented_train, aug_train_path)
            # Eval set is deterministic per seed for downstream wiring; save per-seed copy.
            save_split(eval_sentences, aug_eval_path)

            seed_files[str(seed)] = {
                "seed": int(seed),
                "train_file": aug_train_file,
                "eval_file": aug_eval_file,
                "original_train_sentences": len(train_sentences),
                "generated_sentences": len(generated_sents),
                "augmented_train_sentences": len(augmented_train),
            }
            if not first_seed_train_file:
                first_seed_train_file = aug_train_file
                first_seed_eval_file = aug_eval_file

        first_entry = seed_files.get(str(seed_list[0]), {})
        _log(
            f"  {vm['label']}: generated seed-specific augmented files for {len(seed_list)} seeds "
            f"(seed {seed_list[0]} => {first_entry.get('original_train_sentences', len(train_sentences))} original + "
            f"{first_entry.get('generated_sentences', 0)} generated)"
        )

        aug_variants.append({
            "variant": variant_name,
            "label": vm["label"],
            "description": f"{vm.get('description', vm['label'])} + LLM mask-fill augmentation",
            # Keep train/eval file for backward compatibility (first seed entry).
            "train_file": first_seed_train_file,
            "eval_file": first_seed_eval_file,
            "seed_files": seed_files,
        })

    # Write metadata
    meta = {
        "description": "Exp07 split variants with exp08-style LLM augmentation applied to training data",
        "augmentation_multiplier": multiplier,
        "num_seeds": len(seed_list),
        "seed_list": seed_list,
        "augmentation_model": augmentation_model_name,
        "ner_model_for_resolution": model_name,
        "source_exp07_meta": str(EXP07_SPLITS_DIR / "split_meta.json"),
        "variants": aug_variants,
    }
    meta_path = EXP07_AUG_SPLITS_DIR / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    _log(f"Exp07+augmentation done in {elapsed:.1f}s. Generated {len(aug_variants)} augmented variants.")
    return {
        "source": "generated",
        "generated": True,
        "elapsed": round(elapsed, 1),
        "augmentation_model": augmentation_model_name,
    }


def _load_exp07_augmented_meta() -> dict:
    path = EXP07_AUG_SPLITS_DIR / "split_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Build data conditions
# ---------------------------------------------------------------------------
def _build_conditions(
    condition_sources: list[str] | None = None,
    condition_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a flat list of data conditions with optional filtering.

    Parameters
    ----------
    condition_sources : list[str] | None
        Optional subset of sources to include: ``exp07``, ``exp08``, ``exp07+aug``.
    condition_keys : list[str] | None
        Optional explicit subset of condition keys.
    """
    conditions: list[dict[str, Any]] = []

    # ── Exp07 variants ────────────────────────────────────────────────
    meta07 = _load_exp07_meta()
    baseline_variant = meta07.get("baseline_variant", "")
    for vm in meta07.get("variants", []):
        train_path = EXP07_SPLITS_DIR / vm["train_file"]
        eval_path = EXP07_SPLITS_DIR / vm["eval_file"]
        cond_key = f"exp07_{vm['variant']}"
        if cond_key in EXCLUDED_CONDITION_KEYS:
            continue
        if not train_path.exists() or not eval_path.exists():
            _log(f"WARNING: skipping exp07 variant {vm['variant']} (missing files)")
            continue
        conditions.append({
            "source": "exp07",
            "key": cond_key,
            "variant": vm["variant"],
            "label": f"[Exp07] {vm['label']}",
            "short_label": vm["label"],
            "description": vm.get("description", vm["label"]),
            "train_path": train_path,
            "eval_path": eval_path,
            "is_baseline": vm["variant"] == baseline_variant,
        })

    # ── Exp08 conditions ──────────────────────────────────────────────
    exp08_conditions = [
        {
            "key": "exp08_baseline",
            "label": "[Exp08] Baseline (no augmentation)",
            "short_label": "Exp08 Baseline",
            "description": "Original training data without LLM augmentation",
            "train_file": "baseline_train.json",
            "eval_file": "baseline_eval.json",
            "is_baseline": True,
        },
        {
            "key": "exp08_augmented",
            "label": "[Exp08] Augmented (LLM mask-fill)",
            "short_label": "Exp08 Augmented",
            "description": "Training data augmented with LLM mask-filling generated sentences",
            "train_file": "augmented_train.json",
            "eval_file": "augmented_eval.json",
            "is_baseline": False,
        },
    ]
    for c in exp08_conditions:
        train_path = EXP08_SPLITS_DIR / c["train_file"]
        eval_path = EXP08_SPLITS_DIR / c["eval_file"]
        if not train_path.exists() or not eval_path.exists():
            _log(f"WARNING: skipping exp08 condition {c['key']} (missing files)")
            continue
        conditions.append({
            "source": "exp08",
            "key": c["key"],
            "variant": c["key"],
            "label": c["label"],
            "short_label": c["short_label"],
            "description": c["description"],
            "train_path": train_path,
            "eval_path": eval_path,
            "is_baseline": c["is_baseline"],
        })

    # ── Exp07 + Augmentation combined conditions ─────────────────────
    if EXP07_AUG_SPLITS_DIR.exists() and (EXP07_AUG_SPLITS_DIR / "split_meta.json").exists():
        try:
            meta07_aug = _load_exp07_augmented_meta()
            for vm in meta07_aug.get("variants", []):
                cond_key = f"exp07aug_{vm['variant']}"
                if cond_key in EXCLUDED_CONDITION_KEYS:
                    continue
                seed_files = vm.get("seed_files") if isinstance(vm, dict) else None
                canonical = None
                if isinstance(seed_files, dict) and seed_files:
                    canonical = seed_files.get(str(DEFAULT_BASE_SEED))
                    if canonical is None:
                        canonical = next(iter(seed_files.values()))
                if isinstance(canonical, dict):
                    train_file = canonical.get("train_file")
                    eval_file = canonical.get("eval_file")
                else:
                    train_file = vm.get("train_file")
                    eval_file = vm.get("eval_file")
                train_path = EXP07_AUG_SPLITS_DIR / str(train_file)
                eval_path = EXP07_AUG_SPLITS_DIR / str(eval_file)
                if not train_path.exists() or not eval_path.exists():
                    _log(f"WARNING: skipping exp07+aug variant {vm['variant']} (missing files)")
                    continue
                conditions.append({
                    "source": "exp07+aug",
                    "key": cond_key,
                    "variant": vm["variant"],
                    "label": f"[Exp07+Aug] {vm['label']} + Augmented",
                    "short_label": f"{vm['label']} + Aug",
                    "description": vm.get("description", f"{vm['label']} + augmentation"),
                    "train_path": train_path,
                    "eval_path": eval_path,
                    "seed_files": seed_files,
                    "is_baseline": False,
                })
        except Exception as exc:
            _log(f"WARNING: could not load exp07+aug conditions: {exc}")

    allowed_sources = {"exp07", "exp08", "exp07+aug"}

    if condition_sources:
        source_filter = {
            str(s).strip().lower() for s in condition_sources if str(s).strip()
        }
        unknown_sources = sorted(source_filter - allowed_sources)
        if unknown_sources:
            raise ValueError(
                f"Unknown condition source(s): {unknown_sources}. "
                f"Allowed: {sorted(allowed_sources)}"
            )
        conditions = [c for c in conditions if c["source"] in source_filter]

    if condition_keys:
        key_filter = {str(k).strip() for k in condition_keys if str(k).strip()}
        conditions = [c for c in conditions if c["key"] in key_filter]

    return conditions


def _seed_list(num_seeds: int, base_seed: int = DEFAULT_BASE_SEED) -> list[int]:
    return [base_seed + i for i in range(num_seeds)]


def _expand_conditions_by_seed(
    base_conditions: list[dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for cond in base_conditions:
        for seed in seeds:
            enriched = dict(cond)
            seed_files = cond.get("seed_files") if isinstance(cond, dict) else None
            if isinstance(seed_files, dict) and seed_files:
                seed_entry = seed_files.get(str(seed))
                if isinstance(seed_entry, dict):
                    train_file = str(seed_entry.get("train_file") or "").strip()
                    eval_file = str(seed_entry.get("eval_file") or "").strip()
                    if train_file and eval_file:
                        enriched["train_path"] = EXP07_AUG_SPLITS_DIR / train_file
                        enriched["eval_path"] = EXP07_AUG_SPLITS_DIR / eval_file
            enriched["base_condition_key"] = cond["key"]
            enriched["base_condition_short"] = cond["short_label"]
            enriched["seed"] = int(seed)
            enriched["key"] = f"{cond['key']}__seed{seed}"
            enriched["short_label"] = f"{cond['short_label']} [s{seed}]"
            enriched["label"] = f"{cond['label']} [seed {seed}]"
            expanded.append(enriched)
    return expanded


def _paired_stats_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired tests across seeds for augmentation-vs-non-augmentation conditions."""
    if results_df.empty:
        return pd.DataFrame()

    def _stats(vec: pd.Series) -> tuple[float | None, float | None]:
        vals = pd.to_numeric(vec, errors="coerce").dropna()
        if vals.empty:
            return None, None
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return mean_v, std_v

    rows: list[dict[str, Any]] = []
    models = sorted(str(x) for x in results_df["model_name"].dropna().unique())
    experiments = sorted(str(x) for x in results_df["experiment_id"].dropna().unique())

    for model_name in models:
        for exp_id in experiments:
            subset = results_df[
                (results_df["model_name"] == model_name)
                & (results_df["experiment_id"] == exp_id)
            ].copy()
            if subset.empty:
                continue

            # 1) Exp07 variant vs Exp07+Aug variant (paired by seed)
            plain_rows = subset[subset["condition_group_key"].astype(str).str.startswith("exp07_")]
            for plain_key in sorted(plain_rows["condition_group_key"].dropna().unique()):
                suffix = str(plain_key).replace("exp07_", "", 1)
                aug_key = f"exp07aug_{suffix}"

                a = subset[subset["condition_group_key"] == plain_key][["seed", "f1"]].rename(columns={"f1": "split_only_f1"})
                b = subset[subset["condition_group_key"] == aug_key][["seed", "f1"]].rename(columns={"f1": "split_plus_aug_f1"})
                if a.empty or b.empty:
                    continue
                paired = a.merge(b, on="seed", how="inner").dropna()
                if paired.empty:
                    continue

                x = paired["split_only_f1"].astype(float)
                y = paired["split_plus_aug_f1"].astype(float)
                diffs = y - x

                t_stat = p_t = None
                if ttest_rel is not None and len(diffs) >= 2:
                    try:
                        tr = ttest_rel(y, x, nan_policy="omit")
                        t_stat = float(tr.statistic)
                        p_t = float(tr.pvalue)
                    except Exception:
                        pass

                w_stat = p_w = None
                if wilcoxon is not None and len(diffs) >= 2:
                    try:
                        wr = wilcoxon(diffs)
                        w_stat = float(wr.statistic)
                        p_w = float(wr.pvalue)
                    except Exception:
                        pass

                x_mean, x_sd = _stats(x)
                y_mean, y_sd = _stats(y)
                rows.append({
                    "model": model_name,
                    "experiment_id": exp_id,
                    "comparison": "exp07+aug vs exp07",
                    "condition": str(plain_key).replace("exp07_", ""),
                    "n_pairs": int(len(paired)),
                    "split_only_f1_mean": x_mean,
                    "split_only_f1_sd": x_sd,
                    "split_plus_aug_f1_mean": y_mean,
                    "split_plus_aug_f1_sd": y_sd,
                    "delta_mean_f1": float(diffs.mean()),
                    "t_statistic": t_stat,
                    "t_p_value": p_t,
                    "t_p_lt_0_05": (p_t is not None and p_t < 0.05),
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": p_w,
                    "wilcoxon_p_lt_0_05": (p_w is not None and p_w < 0.05),
                })

            # 2) Exp08 augmented vs baseline (if included)
            a = subset[subset["condition_group_key"] == "exp08_baseline"][["seed", "f1"]].rename(columns={"f1": "baseline_f1"})
            b = subset[subset["condition_group_key"] == "exp08_augmented"][["seed", "f1"]].rename(columns={"f1": "augmented_f1"})
            if not a.empty and not b.empty:
                paired = a.merge(b, on="seed", how="inner").dropna()
                if not paired.empty:
                    x = paired["baseline_f1"].astype(float)
                    y = paired["augmented_f1"].astype(float)
                    diffs = y - x

                    t_stat = p_t = None
                    if ttest_rel is not None and len(diffs) >= 2:
                        try:
                            tr = ttest_rel(y, x, nan_policy="omit")
                            t_stat = float(tr.statistic)
                            p_t = float(tr.pvalue)
                        except Exception:
                            pass

                    w_stat = p_w = None
                    if wilcoxon is not None and len(diffs) >= 2:
                        try:
                            wr = wilcoxon(diffs)
                            w_stat = float(wr.statistic)
                            p_w = float(wr.pvalue)
                        except Exception:
                            pass

                    x_mean, x_sd = _stats(x)
                    y_mean, y_sd = _stats(y)
                    rows.append({
                        "model": model_name,
                        "experiment_id": exp_id,
                        "comparison": "exp08_augmented vs exp08_baseline",
                        "condition": "exp08",
                        "n_pairs": int(len(paired)),
                        "split_only_f1_mean": x_mean,
                        "split_only_f1_sd": x_sd,
                        "split_plus_aug_f1_mean": y_mean,
                        "split_plus_aug_f1_sd": y_sd,
                        "delta_mean_f1": float(diffs.mean()),
                        "t_statistic": t_stat,
                        "t_p_value": p_t,
                        "t_p_lt_0_05": (p_t is not None and p_t < 0.05),
                        "wilcoxon_statistic": w_stat,
                        "wilcoxon_p_value": p_w,
                        "wilcoxon_p_lt_0_05": (p_w is not None and p_w < 0.05),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def run_comparison(
    experiment_ids: list[str] | None = None,
    model_keys: list[str] | None = None,
    exp07_source: str = "auto",
    force_exp08: bool = False,
    num_seeds: int = 3,
    resume: bool = False,
    checkpoint_file: str | None = None,
    skip_augmentation: bool = False,
    condition_sources: list[str] | None = None,
    condition_keys: list[str] | None = None,
    rebuild_from_checkpoint: bool = False,
    base_mode: str = "auto",
) -> dict:
    """Run all (model × data-condition × experiment) combinations.

    Parameters
    ----------
    experiment_ids : list[str] | None
        Downstream experiments to run (default: ``["03","04","05","06"]``).
    model_keys : list[str] | None
        Which models from MODEL_REGISTRY to use (default: all).
    exp07_source : str
        How to prepare exp07 splits: ``auto``, ``saved``, ``rerun``.
    force_exp08 : bool
        If True, rerun experiment 08 even if artifacts exist.
    num_seeds : int
        Number of seeds for exp07/exp08 preparation (default: 3).
    resume : bool
        If True, load existing progress checkpoint and skip completed runs.
    checkpoint_file : str | None
        Optional path to checkpoint file. Defaults to
        outputs/cross_comparison/cross_comparison_progress_latest.json.
    condition_sources : list[str] | None
        Optional subset of condition sources to include.
    condition_keys : list[str] | None
        Optional explicit subset of condition keys to include.
    rebuild_from_checkpoint : bool
        If True, skip all experiment execution and rebuild final exports from
        checkpoint rows only.
    """
    from common import configure_network_environment
    configure_network_environment()

    base_mode = (base_mode or "auto").strip().lower()
    if base_mode not in {"auto", "reuse", "retrain"}:
        raise ValueError("base_mode must be one of: auto, reuse, retrain")

    if num_seeds < 2:
        raise ValueError("num_seeds must be >= 2")

    # Seed count controls paired repeated runs for significance testing.
    os.environ["THESIS_EXP07_NUM_SEEDS"] = str(num_seeds)
    os.environ["THESIS_EXP08_NUM_SEEDS"] = str(num_seeds)
    os.environ["THESIS_DIRECT_SPLIT_RUNS"] = str(num_seeds)
    seeds = _seed_list(num_seeds=num_seeds, base_seed=DEFAULT_BASE_SEED)

    if experiment_ids is None:
        raw = (os.environ.get("THESIS_CROSS_EXPERIMENTS") or "01,04,05_ready,06_ready,06_svm_ready").strip()
        experiment_ids = [x.strip() for x in raw.split(",") if x.strip()]

    unknown_experiments = [e for e in experiment_ids if e not in EXP_SCRIPTS]
    if unknown_experiments:
        raise ValueError(
            f"Unknown experiment ID(s): {unknown_experiments}. "
            f"Allowed: {sorted(EXP_SCRIPTS.keys())}"
        )

    if model_keys is None:
        raw = (os.environ.get("THESIS_CROSS_MODELS") or ",".join(MODEL_REGISTRY.keys())).strip()
        model_keys = [x.strip() for x in raw.split(",") if x.strip()]

    models = []
    for mk in model_keys:
        if mk not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {mk}. Available: {list(MODEL_REGISTRY.keys())}")
        models.append(MODEL_REGISTRY[mk])

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    base_index_path = COMPARISON_DIR / "cross_comparison_base_ready_index.json"
    base_index = _load_base_index(base_index_path)
    base_mem: dict[str, dict[str, Any]] = {}
    checkpoint_path = Path(checkpoint_file).expanduser() if checkpoint_file else (
        COMPARISON_DIR / "cross_comparison_progress_latest.json"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if rebuild_from_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for rebuild mode: {checkpoint_path}"
            )

        cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cp_rows = cp.get("rows", [])
        if not isinstance(cp_rows, list) or not cp_rows:
            raise RuntimeError(
                f"Checkpoint has no rows to rebuild from: {checkpoint_path}"
            )

        selected_model_ids = {m["model_id"] for m in models}
        selected_experiments = {f"exp{e}" for e in experiment_ids}
        source_filter = {
            str(s).strip().lower() for s in (condition_sources or []) if str(s).strip()
        }
        key_filter = {
            str(k).strip() for k in (condition_keys or []) if str(k).strip()
        }

        rows: list[dict] = []
        for r in cp_rows:
            model_id = str(r.get("model_id", "")).strip()
            exp_id = str(r.get("experiment_id", "")).strip()
            src = str(r.get("data_source", "")).strip().lower()
            key = str(r.get("condition_key", "")).strip()

            if model_id not in selected_model_ids:
                continue
            if exp_id not in selected_experiments:
                continue
            if source_filter and src not in source_filter:
                continue
            if key_filter and key not in key_filter:
                continue

            rows.append(r)

        if not rows:
            raise RuntimeError(
                "No checkpoint rows matched the selected models/experiments/conditions."
            )

        conditions = _conditions_from_rows(rows)
        total_runs = len(rows)
        run_counter = len(rows)
        comparison_start = time.time()
        started_at = str(cp.get("started_at") or datetime.now().isoformat())
        prep07 = cp.get("exp07_preparation") or {"source": "checkpoint"}
        prep08 = cp.get("exp08_preparation") or {"source": "checkpoint"}

        _log(f"Rebuild-only mode from checkpoint: {checkpoint_path}")
        _log(f"Rows used: {len(rows)}")
    else:
        # ── Prepare data conditions ───────────────────────────────────────
        print(f"\n{'-'*60}")
        print(f"  [PREP] Exp07 / Exp08 / Exp07+Aug artifacts")
        print(f"{'-'*60}")
        _log(f"Preparing exp07 splits (source={exp07_source})...")
        prep07 = _prepare_exp07_splits(exp07_source)

        env_force_08 = (os.environ.get("THESIS_EXP08_FORCE_RERUN") or "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        _log("Preparing exp08 splits...")
        prep08 = _prepare_exp08_splits(force_rerun=force_exp08 or env_force_08)

        # ── Prepare exp07 + augmentation combined splits ──────────────────
        env_force_07aug = (os.environ.get("THESIS_EXP07AUG_FORCE_RERUN") or "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if not skip_augmentation:
            _log("Preparing exp07+augmentation combined splits...")
            prep07_aug = _prepare_exp07_augmented_splits(force_rerun=force_exp08 or env_force_07aug)
        else:
            _log("Skipping exp07+augmentation combined splits (--skip-augmentation).")
            prep07_aug = {"source": "skipped", "generated": False}

        _log(
            "Preparation summary | "
            f"exp07={prep07.get('source', 'unknown')} | "
            f"exp08={prep08.get('source', 'unknown')} | "
            f"exp07+aug={prep07_aug.get('source', 'unknown')}"
        )

        resolved_aug_model = str(prep07_aug.get("augmentation_model") or "").strip()
        if resolved_aug_model:
            _log(f"Resolved augmentation model: {resolved_aug_model}")
        elif skip_augmentation:
            _log("Resolved augmentation model: N/A (augmentation skipped)")
        else:
            _log("Resolved augmentation model: unavailable")

        base_conditions = _build_conditions(
            condition_sources=condition_sources,
            condition_keys=condition_keys,
        )
        if not base_conditions:
            raise RuntimeError("No data conditions available. Check exp07/exp08 split artifacts.")

        conditions = _expand_conditions_by_seed(base_conditions, seeds)

        needs_base_artifacts = any(
            (e in READY_DEPENDENT_EXP_IDS) or (e in {"01", "04"})
            for e in experiment_ids
        )
        if base_mode == "reuse" and needs_base_artifacts:
            hydrated = _bootstrap_base_cache_from_saved_outputs(
                models=models,
                conditions=conditions,
                base_index=base_index,
                base_index_path=base_index_path,
            )
            if hydrated["added_total"]:
                _log(
                    "Hydrated base cache from saved artifacts: "
                    f"+{hydrated['added_total']} "
                    f"(cross-comparison={hydrated['added_from_cross_comparison']}, "
                    f"standalone={hydrated['added_from_standalone_outputs']})."
                )

            missing = _missing_base_cache_entries(models, conditions, base_index)
            if missing:
                preview = "\n".join(f"  - {x}" for x in missing[:12])
                if len(missing) > 12:
                    preview += f"\n  ... and {len(missing) - 12} more"

                model_keys_hint = ",".join(model_keys)
                warmup_cmd = (
                    "python run_cross_data_model_comparison.py "
                    f"--experiments 01,04 --models {model_keys_hint} --base-mode auto --num-seeds {num_seeds}"
                )
                raise RuntimeError(
                    "Base mode is 'reuse', but required Exp01/Exp04 cache entries are missing.\n"
                    "Missing model/condition pairs (sample):\n"
                    f"{preview}\n"
                    "Run a warm-up pass first to populate the cache, then rerun with --base-mode reuse.\n"
                    f"Suggested command:\n  {warmup_cmd}"
                )

        total_runs = len(models) * len(conditions) * len(experiment_ids)
        run_counter = 0
        comparison_start = time.time()
        started_at = datetime.now().isoformat()

        rows: list[dict] = []
        completed_keys: set[str] = set()

        if resume and checkpoint_path.exists():
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            cp_rows = cp.get("rows", [])
            if isinstance(cp_rows, list):
                selected_model_ids = {m["model_id"] for m in models}
                selected_experiments = {f"exp{e}" for e in experiment_ids}
                selected_condition_keys = {str(c["key"]) for c in conditions}
                rows = []
                for r in cp_rows:
                    if not isinstance(r, dict):
                        continue
                    model_id = str(r.get("model_id", "")).strip()
                    exp_id = str(r.get("experiment_id", "")).strip()
                    cond_key = str(r.get("condition_key", "")).strip()
                    if model_id not in selected_model_ids:
                        continue
                    if exp_id not in selected_experiments:
                        continue
                    if cond_key not in selected_condition_keys:
                        continue
                    rows.append(r)
            rows, added_from_history = _backfill_rows_from_history_json(
                rows=rows,
                models=models,
                experiment_ids=experiment_ids,
                conditions=conditions,
                checkpoint_path=checkpoint_path,
            )
            if added_from_history:
                _log(f"Backfilled {added_from_history} missing rows from historical cross-comparison JSON outputs.")
            # Only skip runs that succeeded; retry runs that had errors
            failed_keys: set[str] = set()
            for r in rows:
                try:
                    mk = str(r.get("model_id", "")).strip()
                    exp_num = str(r.get("experiment_id", "")).replace("exp", "").strip()
                    cond_key = str(r.get("condition_key", "")).strip()
                    status = str(r.get("status", "ok")).strip()
                    if mk and exp_num and cond_key:
                        key = _run_key(mk, exp_num, cond_key)
                        if status.startswith("error"):
                            failed_keys.add(key)
                        else:
                            completed_keys.add(key)
                except Exception:
                    continue
            # Remove failed rows so they can be retried
            if failed_keys:
                rows = [r for r in rows
                        if _run_key(
                            str(r.get("model_id", "")),
                            str(r.get("experiment_id", "")).replace("exp", ""),
                            str(r.get("condition_key", "")),
                        ) not in failed_keys]
                _log(f"Will retry {len(failed_keys)} previously failed runs.")
            run_counter = len(completed_keys)
            started_at = str(cp.get("started_at") or started_at)
            _log(f"Resuming from checkpoint: {checkpoint_path}")
            _log(f"Loaded {len(completed_keys)} completed runs; {total_runs - len(completed_keys)} remaining.")
        elif resume:
            _log(f"Resume requested but checkpoint not found: {checkpoint_path}. Starting fresh.")

        print("\n" + "=" * 75)
        print("  CROSS-DATA x MULTI-MODEL COMPARISON")
        print(f"  Models: {', '.join(m['display_name'] for m in models)}")
        training_exps = [e for e in experiment_ids if e in TRAINING_EXP_IDS]
        inference_exps = [e for e in experiment_ids if e not in TRAINING_EXP_IDS]
        print(f"  Experiments: {', '.join(f'exp{e}' for e in experiment_ids)}")
        if training_exps:
            print(f"    TRAINING (GPU, slow): {', '.join(f'exp{e}' for e in training_exps)}")
        if inference_exps:
            print(f"    INFERENCE (cached, fast): {', '.join(f'exp{e}' for e in inference_exps)}")
        print(
            f"  Data conditions: {len(base_conditions)} base x {len(seeds)} seeds = {len(conditions)} runs/experiment "
            f"({sum(1 for c in base_conditions if c['source']=='exp07')} from exp07, "
            f"{sum(1 for c in base_conditions if c['source']=='exp08')} from exp08, "
            f"{sum(1 for c in base_conditions if c['source']=='exp07+aug')} from exp07+aug)"
        )
        print(f"  Seed list: {seeds[0]}..{seeds[-1]} ({len(seeds)} paired seeds)")
        total_training_runs = len(models) * len(conditions) * len(training_exps)
        total_inference_runs = len(models) * len(conditions) * len(inference_exps)
        print(f"  Total runs: {total_runs}  (TRAINING: {total_training_runs}, INFERENCE: {total_inference_runs})")
        print(f"  Checkpoint: {checkpoint_path}")
        print("=" * 75)

        training_done = 0
        inference_done = 0

        for model_info in models:
            model_id = model_info["model_id"]
            model_display = model_info["display_name"]
            _set_model_env(model_id)

            for exp_id in experiment_ids:
                exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
                is_training = exp_id in TRAINING_EXP_IDS
                run_type = "TRAINING" if is_training else "INFERENCE"
                print(f"\n{'-'*60}")
                print(f"  [{run_type}] {model_display} / exp{exp_id}: {exp_name}  ({len(conditions)} conditions)")
                if is_training:
                    print(f"  >>> GPU training — {total_training_runs - training_done} training runs remaining")
                print(f"{'-'*60}")

                for cond in conditions:
                    run_key = _run_key(model_id, exp_id, cond["key"])
                    if run_key in completed_keys:
                        _log(
                            f"Skip completed | {model_display} | exp{exp_id} | {cond['short_label']}"
                        )
                        if is_training:
                            training_done += 1
                        else:
                            inference_done += 1
                        continue

                    run_counter += 1
                    if is_training:
                        training_done += 1
                        progress_tag = f"TRAIN {training_done}/{total_training_runs}"
                    else:
                        inference_done += 1
                        progress_tag = f"INFER {inference_done}/{total_inference_runs}"
                    t0 = time.time()
                    _log(
                        f"Run {run_counter}/{total_runs} [{progress_tag}] | {model_display} | exp{exp_id} | "
                        f"{cond['short_label']}"
                    )

                    payload: dict[str, Any] = {}
                    reused_base_artifacts = False
                    try:
                        os.environ["THESIS_SPLIT_SEED"] = str(cond.get("seed", DEFAULT_BASE_SEED))
                        # Explicitly bind condition key and exp id for HF Trainer isolated checkpointing
                        os.environ["THESIS_CURRENT_CONDITION_KEY"] = str(cond.get("key", "default"))
                        os.environ["THESIS_CURRENT_EXP_ID"] = f"exp{exp_id}"

                        # Exp05 + fusion ready variants are always executed on top of
                        # cached (or newly trained) Exp01/Exp04 outputs.
                        if exp_id in READY_DEPENDENT_EXP_IDS:
                            base_entry, reused_base_artifacts = _ensure_base_artifacts(
                                model_id=model_id,
                                model_display=model_display,
                                condition=cond,
                                base_mode=base_mode,
                                base_mem=base_mem,
                                base_index=base_index,
                                base_index_path=base_index_path,
                            )
                            _set_ready_env(
                                base_entry["exp01_metrics_file"],
                                base_entry["exp04_metrics_file"],
                            )
                            try:
                                mod = _import_experiment(exp_id)
                                payload = mod.run()
                                metrics = _extract_metrics(payload)
                            finally:
                                _clear_ready_env()

                        # Exp01/Exp04 are also routed through the same base cache,
                        # so they can be reused across reruns with identical conditions.
                        elif exp_id in {"01", "04"}:
                            base_entry, reused_base_artifacts = _ensure_base_artifacts(
                                model_id=model_id,
                                model_display=model_display,
                                condition=cond,
                                base_mode=base_mode,
                                base_mem=base_mem,
                                base_index=base_index,
                                base_index_path=base_index_path,
                            )
                            payload = _load_result_payload(base_entry[f"exp{exp_id}_result_file"])
                            metrics = _extract_metrics(payload)
                            if reused_base_artifacts:
                                metrics["status"] = "ok_reused_base"

                        # Other experiments keep the original pre-split execution flow.
                        else:
                            _set_presplit_env(cond["train_path"], cond["eval_path"])
                            try:
                                mod = _import_experiment(exp_id)
                                payload = mod.run()
                                metrics = _extract_metrics(payload)
                            finally:
                                _clear_presplit_env()

                    except Exception as exc:
                        traceback.print_exc()
                        metrics = {
                            "f1": None, "precision": None, "recall": None,
                            "status": f"error: {exc}",
                        }
                    finally:
                        os.environ.pop("THESIS_SPLIT_SEED", None)
                        os.environ.pop("THESIS_CURRENT_CONDITION_KEY", None)
                        os.environ.pop("THESIS_CURRENT_EXP_ID", None)

                    elapsed = time.time() - t0
                    _log(f"  F1={_fmt(metrics.get('f1'))} ({elapsed:.1f}s)")

                    rows.append({
                        "model_key": next(k for k, v in MODEL_REGISTRY.items() if v["model_id"] == model_id),
                        "model_id": model_id,
                        "model_name": model_display,
                        "experiment_id": f"exp{exp_id}",
                        "experiment_name": exp_name,
                        "data_source": cond["source"],
                        "condition_key": cond["key"],
                        "condition_group_key": cond.get("base_condition_key", cond["key"]),
                        "condition_group_short": cond.get("base_condition_short", cond["short_label"]),
                        "condition_label": cond["label"],
                        "condition_short": cond["short_label"],
                        "condition_description": cond["description"],
                        "seed": cond.get("seed"),
                        "is_baseline": cond["is_baseline"],
                        "f1": metrics.get("f1"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "status": metrics.get("status"),
                        "result_file": payload.get("result_file", ""),
                        "metrics_file": payload.get("metrics_file", ""),
                        "base_artifacts_reused": reused_base_artifacts,
                        "base_mode": base_mode,
                        "elapsed_seconds": round(elapsed, 1),
                    })
                    completed_keys.add(run_key)
                    _save_progress_checkpoint(
                        checkpoint_path=checkpoint_path,
                        rows=rows,
                        models=models,
                        experiment_ids=experiment_ids,
                        conditions=conditions,
                        prep07=prep07,
                        prep08=prep08,
                        run_counter=run_counter,
                        total_runs=total_runs,
                        started_at=started_at,
                    )

    total_elapsed = time.time() - comparison_start
    results_df = pd.DataFrame(rows)
    if "condition_group_key" not in results_df.columns:
        results_df["condition_group_key"] = results_df.get("condition_key")
    if "condition_group_short" not in results_df.columns:
        results_df["condition_group_short"] = results_df.get("condition_short")
    if "seed" not in results_df.columns:
        results_df["seed"] = None

    if "base_conditions" not in locals():
        base_conditions = []
        seen_base_keys: set[str] = set()
        for _, r in results_df.iterrows():
            key = str(r.get("condition_group_key") or r.get("condition_key") or "").strip()
            if not key or key in seen_base_keys:
                continue
            seen_base_keys.add(key)
            base_conditions.append({
                "source": str(r.get("data_source", "unknown")),
                "key": key,
                "short_label": str(r.get("condition_group_short") or r.get("condition_short") or key),
                "description": str(r.get("condition_description") or ""),
            })

    # ==================================================================
    # Post-processing: build analytical sheets
    # ==================================================================

    grouped = (
        results_df
        .groupby(
            [
                "model_name",
                "experiment_id",
                "experiment_name",
                "data_source",
                "condition_group_key",
                "condition_group_short",
                "condition_description",
                "is_baseline",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            f1_mean=("f1", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean()) if not pd.to_numeric(s, errors="coerce").dropna().empty else None),
            f1_std=("f1", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().std(ddof=1)) if len(pd.to_numeric(s, errors="coerce").dropna()) > 1 else (0.0 if len(pd.to_numeric(s, errors="coerce").dropna()) == 1 else None)),
            precision_mean=("precision", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean()) if not pd.to_numeric(s, errors="coerce").dropna().empty else None),
            recall_mean=("recall", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean()) if not pd.to_numeric(s, errors="coerce").dropna().empty else None),
            n_seeds=("seed", "nunique"),
        )
    )

    # ── 1. Summary pivot (mean F1 +- SD across paired seeds)
    pivot_rows: list[dict[str, Any]] = []
    for model_info in models:
        m_name = model_info["display_name"]
        m_results = grouped[grouped["model_name"] == m_name]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            exp_results = m_results[m_results["experiment_id"] == f"exp{exp_id}"]
            row: dict[str, Any] = {
                "model": m_name,
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
            }
            for _, r in exp_results.iterrows():
                label = str(r["condition_group_short"])
                row[label] = r.get("f1_mean")
                row[f"{label}_std"] = r.get("f1_std")
            f1s = pd.to_numeric(exp_results["f1_mean"], errors="coerce").dropna()
            row["best_f1"] = float(f1s.max()) if not f1s.empty else None
            row["best_condition"] = (
                exp_results.loc[f1s.idxmax(), "condition_group_short"]
                if not f1s.empty else "N/A"
            )
            pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows)

    # ── 2. Exp07 deltas: each variant vs exp07 baseline (seed-aggregated means)
    try:
        meta07 = _load_exp07_meta()
    except Exception:
        meta07 = {"baseline_variant": ""}
    baseline07_key = f"exp07_{meta07.get('baseline_variant', '')}"
    delta07_rows: list[dict[str, Any]] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            baseline_row = grouped[
                (grouped["model_name"] == m_name)
                & (grouped["experiment_id"] == f"exp{exp_id}")
                & (grouped["condition_group_key"] == baseline07_key)
            ]
            if baseline_row.empty:
                continue
            b = baseline_row.iloc[0]
            exp07_rows = grouped[
                (grouped["model_name"] == m_name)
                & (grouped["experiment_id"] == f"exp{exp_id}")
                & (grouped["data_source"] == "exp07")
                & (grouped["condition_group_key"] != baseline07_key)
            ]
            for _, a in exp07_rows.iterrows():
                delta07_rows.append({
                    "model": m_name,
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "variant": a["condition_group_short"],
                    "baseline_f1": _to_float(b.get("f1_mean")),
                    "variant_f1": _to_float(a.get("f1_mean")),
                    "delta_f1": (_to_float(a.get("f1_mean")) - _to_float(b.get("f1_mean")))
                    if _to_float(a.get("f1_mean")) is not None and _to_float(b.get("f1_mean")) is not None else None,
                    "baseline_precision": _to_float(b.get("precision_mean")),
                    "variant_precision": _to_float(a.get("precision_mean")),
                    "baseline_recall": _to_float(b.get("recall_mean")),
                    "variant_recall": _to_float(a.get("recall_mean")),
                })
    deltas07_df = pd.DataFrame(delta07_rows) if delta07_rows else pd.DataFrame()

    # ── 3. Exp08 deltas: augmented vs baseline (seed-aggregated means)
    delta08_rows: list[dict[str, Any]] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            b_row = grouped[
                (grouped["model_name"] == m_name)
                & (grouped["experiment_id"] == f"exp{exp_id}")
                & (grouped["condition_group_key"] == "exp08_baseline")
            ]
            a_row = grouped[
                (grouped["model_name"] == m_name)
                & (grouped["experiment_id"] == f"exp{exp_id}")
                & (grouped["condition_group_key"] == "exp08_augmented")
            ]
            if b_row.empty or a_row.empty:
                continue
            b, a = b_row.iloc[0], a_row.iloc[0]
            delta08_rows.append({
                "model": m_name,
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
                "baseline_f1": _to_float(b.get("f1_mean")),
                "augmented_f1": _to_float(a.get("f1_mean")),
                "delta_f1": (_to_float(a.get("f1_mean")) - _to_float(b.get("f1_mean")))
                if _to_float(a.get("f1_mean")) is not None and _to_float(b.get("f1_mean")) is not None else None,
            })
    deltas08_df = pd.DataFrame(delta08_rows) if delta08_rows else pd.DataFrame()

    # ── 3b. Exp07+Aug deltas: augmented variant vs non-augmented exp07 variant
    delta07aug_rows: list[dict[str, Any]] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            exp07_rows = grouped[
                (grouped["model_name"] == m_name)
                & (grouped["experiment_id"] == f"exp{exp_id}")
                & (grouped["data_source"] == "exp07")
            ]
            for _, b in exp07_rows.iterrows():
                variant = str(b["condition_group_key"]).replace("exp07_", "")
                aug_key = f"exp07aug_{variant}"
                a_row = grouped[
                    (grouped["model_name"] == m_name)
                    & (grouped["experiment_id"] == f"exp{exp_id}")
                    & (grouped["condition_group_key"] == aug_key)
                ]
                if a_row.empty:
                    continue
                a = a_row.iloc[0]
                bf = _to_float(b.get("f1_mean"))
                af = _to_float(a.get("f1_mean"))
                delta07aug_rows.append({
                    "model": m_name,
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "split_variant": b["condition_group_short"],
                    "split_only_f1": bf,
                    "split_plus_aug_f1": af,
                    "delta_f1": (af - bf) if af is not None and bf is not None else None,
                })
    deltas07aug_df = pd.DataFrame(delta07aug_rows) if delta07aug_rows else pd.DataFrame()

    # ── 3c. Publication-ready paired tests (same seeds, paired design)
    paired_tests_df = _paired_stats_rows(results_df)

    # ── 4. Model comparison: same condition, head-to-head ────────────
    model_cmp_rows: list[dict] = []
    if len(models) == 2:
        m0, m1 = models[0]["display_name"], models[1]["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            for cond in base_conditions:
                cond_key = cond["key"]
                r0 = grouped[
                    (grouped["model_name"] == m0)
                    & (grouped["experiment_id"] == f"exp{exp_id}")
                    & (grouped["condition_group_key"] == cond_key)
                ]
                r1 = grouped[
                    (grouped["model_name"] == m1)
                    & (grouped["experiment_id"] == f"exp{exp_id}")
                    & (grouped["condition_group_key"] == cond_key)
                ]
                if r0.empty or r1.empty:
                    continue
                v0, v1 = r0.iloc[0], r1.iloc[0]
                f1_0 = _to_float(v0.get("f1_mean"))
                f1_1 = _to_float(v1.get("f1_mean"))
                model_cmp_rows.append({
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "data_source": cond["source"],
                    "condition": cond["short_label"],
                    f"{m0}_f1": f1_0,
                    f"{m1}_f1": f1_1,
                    "delta_f1 (model)": (f1_0 - f1_1) if f1_0 is not None and f1_1 is not None else None,
                    "better_model": (
                        m0 if (f1_0 or 0) > (f1_1 or 0) else m1
                        if (f1_0 or 0) != (f1_1 or 0) else "tie"
                    ),
                    f"{m0}_precision": _to_float(v0.get("precision_mean")),
                    f"{m1}_precision": _to_float(v1.get("precision_mean")),
                    f"{m0}_recall": _to_float(v0.get("recall_mean")),
                    f"{m1}_recall": _to_float(v1.get("recall_mean")),
                })
    model_cmp_df = pd.DataFrame(model_cmp_rows) if model_cmp_rows else pd.DataFrame()

    # ── 5. Variant summary across models and experiments ─────────────
    variant_rows: list[dict] = []
    for cond in base_conditions:
        cond_results = results_df[results_df["condition_group_key"] == cond["key"]]
        f1s = pd.to_numeric(cond_results["f1"], errors="coerce").dropna()
        variant_rows.append({
            "data_source": cond["source"],
            "condition": cond["short_label"],
            "description": cond["description"],
            "num_runs": len(cond_results),
            "f1_mean": float(f1s.mean()) if not f1s.empty else None,
            "f1_std": float(f1s.std()) if len(f1s) > 1 else None,
            "f1_min": float(f1s.min()) if not f1s.empty else None,
            "f1_max": float(f1s.max()) if not f1s.empty else None,
        })
    variant_summary_df = (
        pd.DataFrame(variant_rows)
        .sort_values(by="f1_mean", ascending=False, ignore_index=True)
        if variant_rows else pd.DataFrame()
    )

    # ── 6. Documentation sheet ────────────────────────────────────────
    from split_io import build_thesis_documentation_df
    try:
        meta08 = _load_exp08_meta()
    except Exception:
        meta08 = {}

    model_names_str = ", ".join(m["display_name"] for m in models)
    exp07_variant_str = "; ".join(
        vm["label"] for vm in meta07.get("variants", [])
    )
    doc_df = build_thesis_documentation_df(
        "cross_comparison",
        "Cross-Data × Multi-Model Comparison (Ready Setup)",
        extra_rows=[
            {"Section": "Design", "Key": "Models",
             "Value": model_names_str},
            {"Section": "Design", "Key": "Paired Seed Design",
             "Value": f"Same seed list used for every model/condition: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)"},
            {"Section": "Design", "Key": "Exp07 Conditions",
             "Value": f"{len([c for c in base_conditions if c['source']=='exp07'])} variants: {exp07_variant_str}"},
            {"Section": "Design", "Key": "Exp08 Conditions",
             "Value": "Baseline (no augmentation) vs Augmented (LLM mask-fill)"},
            {"Section": "Design", "Key": "Exp07+Aug Conditions",
             "Value": f"{len([c for c in base_conditions if c['source']=='exp07+aug'])} variants: "
                      f"each exp07 split variant with LLM augmentation applied to training data"},
            {"Section": "Design", "Key": "Exp08 Augmentation",
             "Value": (
                 f"Seed={meta08.get('seed')}, multiplier={meta08.get('multiplier')}, "
                 f"baseline_train={meta08.get('baseline_train_sentences')} sents, "
                 f"augmented_train={meta08.get('augmented_train_sentences')} sents "
                 f"(+{meta08.get('generated_sentences')} generated)"
             )},
            {"Section": "Design", "Key": "Downstream Experiments",
             "Value": "; ".join(f"exp{e}: {EXP_NAMES.get(e, e)}" for e in experiment_ids)},
            {"Section": "Design", "Key": "Total Runs",
             "Value": str(total_runs)},
            {"Section": "Interpretation", "Key": "Positive delta_f1 (exp07)",
             "Value": "The split variant improved F1 vs the exp07 baseline (simple random split)"},
            {"Section": "Interpretation", "Key": "Positive delta_f1 (exp08)",
             "Value": "LLM augmentation improved F1 vs the non-augmented baseline"},
            {"Section": "Interpretation", "Key": "delta_f1 (model)",
             "Value": f"Positive = {models[0]['display_name']} better; negative = {models[1]['display_name']} better"
             if len(models) == 2 else "Head-to-head delta between models"},
            {"Section": "Sheets", "Key": "summary_pivot",
             "Value": "One row per (model × experiment) with F1 for every condition + best overall"},
            {"Section": "Sheets", "Key": "all_runs",
             "Value": "Complete per-run detail: model, experiment, condition, F1/P/R, timing"},
            {"Section": "Sheets", "Key": "deltas_exp07",
             "Value": "Paired delta (variant − baseline) for exp07 conditions per model per experiment"},
            {"Section": "Sheets", "Key": "deltas_exp08",
             "Value": "Paired delta (augmented − baseline) for exp08 conditions per model per experiment"},
            {"Section": "Sheets", "Key": "deltas_exp07_aug",
             "Value": "Paired delta (split+augmentation − split only) for each exp07 variant per model per experiment"},
            {"Section": "Sheets", "Key": "paired_tests",
             "Value": "Paired t-test + Wilcoxon signed-rank across shared seeds (mean±SD, test statistic, p-value, p<0.05)"},
            {"Section": "Sheets", "Key": "model_comparison",
             "Value": "Head-to-head F1 comparison per (experiment × condition) pair"},
            {"Section": "Sheets", "Key": "variant_summary",
             "Value": "Per-condition aggregate statistics (mean/std/min/max F1) across all models and experiments"},
            {"Section": "Sheets", "Key": "experiment_details",
             "Value": "Extended detail including file paths and descriptions"},
            {"Section": "Sheets", "Key": "documentation",
             "Value": "This sheet — describes columns and interpretation guidelines"},
        ],
    )

    # ==================================================================
    # Write outputs
    # ==================================================================
    ts = _now_ts()

    # ── Excel ─────────────────────────────────────────────────────────
    xlsx_path = COMPARISON_DIR / f"cross_comparison_{ts}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not pivot_df.empty:
            pivot_df.to_excel(writer, sheet_name="summary_pivot", index=False)
        results_df.to_excel(writer, sheet_name="all_runs", index=False)
        if not deltas07_df.empty:
            deltas07_df.to_excel(writer, sheet_name="deltas_exp07", index=False)
        if not deltas08_df.empty:
            deltas08_df.to_excel(writer, sheet_name="deltas_exp08", index=False)
        if not deltas07aug_df.empty:
            deltas07aug_df.to_excel(writer, sheet_name="deltas_exp07_aug", index=False)
        if not paired_tests_df.empty:
            paired_tests_df.to_excel(writer, sheet_name="paired_tests", index=False)
        if not model_cmp_df.empty:
            model_cmp_df.to_excel(writer, sheet_name="model_comparison", index=False)
        if not variant_summary_df.empty:
            variant_summary_df.to_excel(writer, sheet_name="variant_summary", index=False)
        details_df = results_df[[
            "model_name", "experiment_id", "experiment_name",
            "data_source", "condition_group_key", "condition_group_short",
            "condition_label", "condition_description", "seed",
            "f1", "precision", "recall", "status",
            "result_file", "metrics_file", "elapsed_seconds",
        ]].copy()
        details_df.to_excel(writer, sheet_name="experiment_details", index=False)
        doc_df.to_excel(writer, sheet_name="documentation", index=False)

    latest_xlsx = COMPARISON_DIR / "cross_comparison_latest.xlsx"
    latest_xlsx_effective = latest_xlsx
    try:
        if latest_xlsx.exists():
            latest_xlsx.unlink()
        shutil.copy2(xlsx_path, latest_xlsx)
    except PermissionError:
        # Common on Windows if the workbook is open in Excel.
        latest_xlsx_effective = COMPARISON_DIR / f"cross_comparison_latest_locked_{ts}.xlsx"
        shutil.copy2(xlsx_path, latest_xlsx_effective)
        _log(
            "WARNING: Could not overwrite cross_comparison_latest.xlsx (file is locked). "
            f"Wrote fallback file instead: {latest_xlsx_effective}"
        )

    # ── JSON ──────────────────────────────────────────────────────────
    payload_out: dict[str, Any] = {
        "name": "Cross-Data × Multi-Model Comparison (Ready Setup)",
        "description": (
            f"Runs selected experiments with {len(models)} models × "
            f"{len(base_conditions)} base data conditions × {len(seeds)} paired seeds from exp07/exp08. "
            f"Total runs: {total_runs}."
        ),
        "models": [m["model_id"] for m in models],
        "model_details": models,
        "experiments": experiment_ids,
        "exp07_preparation": prep07,
        "exp07_meta": meta07,
        "exp08_preparation": prep08,
        "exp08_meta": meta08,
        "num_models": len(models),
        "num_conditions": len(base_conditions),
        "num_seeded_conditions": len(conditions),
        "num_seeds": len(seeds),
        "seed_list": seeds,
        "num_experiments": len(experiment_ids),
        "total_runs": total_runs,
        "elapsed_seconds": round(total_elapsed, 1),
        "results": results_df.to_dict(orient="records"),
        "summary_pivot": pivot_df.to_dict(orient="records") if not pivot_df.empty else [],
        "deltas_exp07": deltas07_df.to_dict(orient="records") if not deltas07_df.empty else [],
        "deltas_exp08": deltas08_df.to_dict(orient="records") if not deltas08_df.empty else [],
        "deltas_exp07_aug": deltas07aug_df.to_dict(orient="records") if not deltas07aug_df.empty else [],
        "paired_tests": paired_tests_df.to_dict(orient="records") if not paired_tests_df.empty else [],
        "model_comparison": model_cmp_df.to_dict(orient="records") if not model_cmp_df.empty else [],
        "variant_summary": variant_summary_df.to_dict(orient="records") if not variant_summary_df.empty else [],
        "xlsx": str(xlsx_path),
        "xlsx_latest": str(latest_xlsx_effective),
        "status": "ok",
    }
    json_path = COMPARISON_DIR / f"cross_comparison_{ts}.json"
    _atomic_write_json(json_path, payload_out)
    latest_json = COMPARISON_DIR / "cross_comparison_latest.json"
    _atomic_write_json(latest_json, payload_out)

    if not rebuild_from_checkpoint:
        _save_progress_checkpoint(
            checkpoint_path=checkpoint_path,
            rows=rows,
            models=models,
            experiment_ids=experiment_ids,
            conditions=conditions,
            prep07=prep07,
            prep08=prep08,
            run_counter=len(rows),
            total_runs=total_runs,
            started_at=started_at,
        )

    # Console summary
    print(f"\n{'='*75}")
    print("  CROSS-DATA x MULTI-MODEL COMPARISON - RESULTS")
    print(f"{'='*75}")

    for model_info in models:
        m_name = model_info["display_name"]
        print(f"\n  Model: {m_name}")
        m_results = grouped[grouped["model_name"] == m_name]
        for exp_id in experiment_ids:
            exp_results = m_results[m_results["experiment_id"] == f"exp{exp_id}"]
            if exp_results.empty:
                continue
            numeric_f1 = pd.to_numeric(exp_results["f1_mean"], errors="coerce")
            if numeric_f1.isna().all():
                print(
                    f"    exp{exp_id} ({EXP_NAMES.get(exp_id, exp_id)}): "
                    f"best F1=N/A (all runs failed)"
                )
                continue
            best = exp_results.loc[numeric_f1.idxmax()]
            best_std = _to_float(best.get("f1_std"))
            std_txt = f"±{best_std:.4f}" if best_std is not None else ""
            print(
                f"    exp{exp_id} ({EXP_NAMES.get(exp_id, exp_id)}): "
                f"best mean F1={_fmt(_to_float(best.get('f1_mean')))}{std_txt} "
                f"[{best.get('condition_group_short', 'N/A')}]"
            )

    if not paired_tests_df.empty:
        sig_t = int((paired_tests_df["t_p_lt_0_05"] == True).sum())
        sig_w = int((paired_tests_df["wilcoxon_p_lt_0_05"] == True).sum())
        print(
            f"\n  Paired tests: {len(paired_tests_df)} comparisons | "
            f"t-test significant={sig_t} | Wilcoxon significant={sig_w}"
        )

    if not model_cmp_df.empty:
        print(f"\n  Head-to-head (model comparison):")
        m0, m1 = models[0]["display_name"], models[1]["display_name"]
        wins0 = len(model_cmp_df[model_cmp_df["better_model"] == m0])
        wins1 = len(model_cmp_df[model_cmp_df["better_model"] == m1])
        ties = len(model_cmp_df[model_cmp_df["better_model"] == "tie"])
        print(f"    {m0}: {wins0} wins | {m1}: {wins1} wins | Ties: {ties}")
        avg_delta = model_cmp_df["delta_f1 (model)"].dropna().mean()
        print(f"    Avg delta F1 ({m0} − {m1}): {avg_delta:+.4f}")

    print(f"\n  Total elapsed: {total_elapsed:.0f}s")
    print(f"  Excel: {xlsx_path}")
    print(f"  JSON:  {json_path}")

    _log("Comparison complete.")
    return payload_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run selected experiments with data from exp07 (split strategies) and exp08 "
            "(LLM augmentation), comparing models in the ready-results setup."
        )
    )
    parser.add_argument(
        "--exp07-source",
        choices=["auto", "saved", "rerun"],
        default=(os.environ.get("THESIS_EXP07_SOURCE") or "auto").strip().lower(),
        help="Exp07 split artifact policy: auto (default), saved, or rerun.",
    )
    parser.add_argument(
        "--force-exp08",
        action="store_true",
        help="Force rerun experiment 08 before comparison.",
    )
    parser.add_argument(
        "--experiments",
        default=(os.environ.get("THESIS_CROSS_EXPERIMENTS") or "01,04,05_ready,06_ready,06_svm_ready").strip(),
        help=(
            "Comma-separated experiment IDs (default: 01,04,05_ready,06_ready,06_svm_ready)."
        ),
    )
    parser.add_argument(
        "--base-mode",
        choices=["auto", "reuse", "retrain"],
        default=(os.environ.get("THESIS_CROSS_BASE_MODE") or "auto").strip().lower(),
        help=(
            "How to obtain base artifacts (exp01+exp04) per model/condition: "
            "auto=reuse if cached else train, reuse=only reuse (fail if missing), "
            "retrain=always retrain once per model/condition."
        ),
    )
    parser.add_argument(
        "--models",
        default=(os.environ.get("THESIS_CROSS_MODELS") or "dictabert,berel,hero,alephbertgimmel").strip(),
        help="Comma-separated model keys (default: dictabert,berel,hero,alephbertgimmel). Available: "
             + ", ".join(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=int((os.environ.get("THESIS_CROSS_NUM_SEEDS") or "20").strip()),
        help="Seed count for exp07/exp08 (default: 20 for publication-quality significance).",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=True,
        help="Deprecated toggle; trained models are always saved to outputs/trained_models/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved progress checkpoint and skip completed runs.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=(os.environ.get("THESIS_CROSS_CHECKPOINT_FILE") or "").strip(),
        help=(
            "Optional checkpoint file path. Default: "
            "outputs/cross_comparison/cross_comparison_progress_latest.json"
        ),
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip the exp07+augmentation combined conditions (faster runs).",
    )
    parser.add_argument(
        "--condition-sources",
        default=(os.environ.get("THESIS_CROSS_CONDITION_SOURCES") or "exp07,exp07+aug").strip(),
        help=(
            "Comma-separated condition sources to include "
            "(default: exp07,exp07+aug). Allowed: exp07, exp08, exp07+aug."
        ),
    )
    parser.add_argument(
        "--condition-keys",
        default=(os.environ.get("THESIS_CROSS_CONDITION_KEYS") or "").strip(),
        help="Comma-separated explicit condition keys to include (default: all keys in selected sources).",
    )
    parser.add_argument(
        "--rebuild-from-checkpoint",
        action="store_true",
        help=(
            "Rebuild cross-comparison Excel/JSON from checkpoint rows only, "
            "without running any experiments."
        ),
    )
    parser.add_argument(
        "--list-base-cache",
        action="store_true",
        help=(
            "Print cached base artifact entries from "
            "outputs/cross_comparison/cross_comparison_base_ready_index.json and exit."
        ),
    )
    args = parser.parse_args()

    if args.list_base_cache:
        base_index_path = COMPARISON_DIR / "cross_comparison_base_ready_index.json"
        _print_base_cache_summary(base_index_path)
        raise SystemExit(0)

    experiment_ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    condition_sources = [x.strip() for x in args.condition_sources.split(",") if x.strip()]
    condition_keys = [x.strip() for x in args.condition_keys.split(",") if x.strip()]

    # Always enable model saving for artifact reuse in future fusion runs.
    os.environ["THESIS_SAVE_TRAINED_MODELS"] = "1"

    result = run_comparison(
        experiment_ids=experiment_ids,
        model_keys=model_keys,
        exp07_source=args.exp07_source,
        force_exp08=args.force_exp08,
        num_seeds=args.num_seeds,
        resume=args.resume,
        checkpoint_file=(args.checkpoint_file or None),
        skip_augmentation=args.skip_augmentation,
        condition_sources=condition_sources,
        condition_keys=condition_keys,
        rebuild_from_checkpoint=args.rebuild_from_checkpoint,
        base_mode=args.base_mode,
    )
