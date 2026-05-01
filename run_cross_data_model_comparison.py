"""
run_cross_data_model_comparison.py — Cross-Data × Multi-Model Comparison (Exp07 & Exp08 → Exp01,03-06)
====================================================================================================

Purpose
-------
This script systematically evaluates **two base models** across **all training-data
conditions** produced by experiments 07 (sentence-split strategies) and 08 (LLM
data augmentation), feeding each condition into experiments 01, 03–06.

The two models are:
  * ``dicta-il/dictabert``   — general-purpose Hebrew BERT
  * ``dicta-il/BEREL_3.0``   — Biblical/Rabbinical Hebrew BERT

The data conditions are:
  * **Experiment 07** variants (typically 8 sentence-split strategies)
  * **Experiment 08** conditions (baseline vs LLM-augmented training data)
  * **Experiment 07 + Augmentation** combined (each exp07 split + LLM augmentation)

For every *(model × data-condition × downstream-experiment)* triple the script
records F1, precision, recall, and status.

Outputs
-------
* ``outputs/cross_comparison/cross_comparison_<timestamp>.xlsx``
  Sheets: summary_pivot, all_runs, deltas_exp07, deltas_exp08,
  model_comparison, variant_summary, experiment_details, documentation
* ``outputs/cross_comparison/cross_comparison_latest.xlsx`` (copy)
* ``outputs/cross_comparison/cross_comparison_<timestamp>.json`` + ``latest.json``

Usage
-----
::

    # Default: use saved exp07 & exp08 splits (auto-generate if missing)
    python run_cross_data_model_comparison.py

    # Force rerun exp07 and exp08 before comparison
    python run_cross_data_model_comparison.py --exp07-source rerun --force-exp08

    # Run only specific experiments
    python run_cross_data_model_comparison.py --experiments 03,04

    # Run only specific models
    python run_cross_data_model_comparison.py --models dictabert

    # Resume a stopped run from latest checkpoint
    set THESIS_EXP07AUG_FORCE_RERUN=1
    python run_cross_data_model_comparison.py --resume

    # Resume from a specific checkpoint file
    python run_cross_data_model_comparison.py --resume --checkpoint-file outputs/cross_comparison/my_run_checkpoint.json

    python run_cross_data_model_comparison.py --resume --models dictabert,berel
    python run_cross_data_model_comparison.py --resume --models berel


Environment Variables
---------------------
``THESIS_EXP07_SOURCE``
    Split artifact policy: ``auto`` (default), ``saved``, ``rerun``.
``THESIS_CROSS_EXPERIMENTS``
    Comma-separated experiment IDs (default: ``03,04,05,06``).
``THESIS_CROSS_MODELS``
    Comma-separated model keys (default: ``dictabert,berel``).
``THESIS_CROSS_NUM_SEEDS``
    Seed count for exp07/exp08 (and direct split fallback), default ``3``.
``THESIS_MODEL_NAME``
    Overridden per-model internally; do **not** set this manually.
``THESIS_DEBUG``
    Set to ``1`` for verbose subprocess output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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
}

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXP_NAMES: dict[str, str] = {
    "01": "Regular NER",
    "03": "AUC-2T",
    "04": "AUC Cascaded Pipeline",
    "05": "AUC Cascaded Step-3 Consistency",
    "06": "Fusion (Regular + Cascaded)",
}

EXP_SCRIPTS: dict[str, str] = {
    "01": "experiment_01_regular_ner",
    "03": "experiment_03_auc_2t",
    "04": "experiment_04_auc_cascaded_pipeline",
    "05": "experiment_05_auc_cascaded_pipeline_step3_consistency",
    "06": "experiment_06_fusion_regular_and_cascaded",
    "07": "experiment_07_sentence_split_strategy",
    "08": "experiment_08_llm_augmentation",
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
    _log("Running experiment 08 to generate augmentation data...")
    t0 = time.time()
    mod08 = _import_experiment("08")
    mod08.run()
    elapsed = time.time() - t0
    ok, msg = _exp08_artifacts_ready()
    if not ok:
        raise RuntimeError(f"Exp08 finished but artifacts incomplete: {msg}")
    _log(f"Experiment 08 done in {elapsed:.1f}s.")
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
    for vm in variants:
        tf = vm.get("train_file")
        ef = vm.get("eval_file")
        if not tf or not ef:
            return False, f"Variant entry missing files: {vm}"
        if not (EXP07_AUG_SPLITS_DIR / tf).exists() or not (EXP07_AUG_SPLITS_DIR / ef).exists():
            return False, f"Missing augmented split file for variant {vm.get('variant')}"
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

    meta07 = _load_exp07_meta()
    EXP07_AUG_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    aug_variants: list[dict[str, Any]] = []

    for vm in meta07.get("variants", []):
        variant_name = vm["variant"]
        train_path = EXP07_SPLITS_DIR / vm["train_file"]
        eval_path = EXP07_SPLITS_DIR / vm["eval_file"]

        if not train_path.exists() or not eval_path.exists():
            _log(f"WARNING: skipping augmentation for exp07 variant {variant_name} (missing files)")
            continue

        _log(f"Augmenting exp07 variant: {vm['label']}...")
        train_sentences = load_split(train_path)

        with suppress_output_if_needed():
            generated_sents, _ = augment_fn(
                train_sentences, data_df, model_name,
                multiplier=multiplier, rng_seed=42,
            )
        augmented_train = train_sentences + generated_sents
        _log(
            f"  {vm['label']}: {len(train_sentences)} original + "
            f"{len(generated_sents)} generated = {len(augmented_train)} total"
        )

        # Save augmented train and original eval
        aug_train_file = f"{variant_name}_augmented_train.json"
        aug_eval_file = f"{variant_name}_eval.json"
        save_split(augmented_train, EXP07_AUG_SPLITS_DIR / aug_train_file)
        # Eval is unchanged — just copy for clarity
        eval_sentences = load_split(eval_path)
        save_split(eval_sentences, EXP07_AUG_SPLITS_DIR / aug_eval_file)

        aug_variants.append({
            "variant": variant_name,
            "label": vm["label"],
            "description": f"{vm.get('description', vm['label'])} + LLM mask-fill augmentation",
            "original_train_sentences": len(train_sentences),
            "generated_sentences": len(generated_sents),
            "augmented_train_sentences": len(augmented_train),
            "train_file": aug_train_file,
            "eval_file": aug_eval_file,
        })

    # Write metadata
    meta = {
        "description": "Exp07 split variants with exp08-style LLM augmentation applied to training data",
        "augmentation_multiplier": multiplier,
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
        if not train_path.exists() or not eval_path.exists():
            _log(f"WARNING: skipping exp07 variant {vm['variant']} (missing files)")
            continue
        conditions.append({
            "source": "exp07",
            "key": f"exp07_{vm['variant']}",
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
                train_path = EXP07_AUG_SPLITS_DIR / vm["train_file"]
                eval_path = EXP07_AUG_SPLITS_DIR / vm["eval_file"]
                if not train_path.exists() or not eval_path.exists():
                    _log(f"WARNING: skipping exp07+aug variant {vm['variant']} (missing files)")
                    continue
                conditions.append({
                    "source": "exp07+aug",
                    "key": f"exp07aug_{vm['variant']}",
                    "variant": vm["variant"],
                    "label": f"[Exp07+Aug] {vm['label']} + Augmented",
                    "short_label": f"{vm['label']} + Aug",
                    "description": vm.get("description", f"{vm['label']} + augmentation"),
                    "train_path": train_path,
                    "eval_path": eval_path,
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
    """
    from common import configure_network_environment
    configure_network_environment()

    if num_seeds < 2:
        raise ValueError("num_seeds must be >= 2")

    # Use smaller seed count for faster initial iteration.
    os.environ["THESIS_EXP07_NUM_SEEDS"] = str(num_seeds)
    os.environ["THESIS_EXP08_NUM_SEEDS"] = str(num_seeds)
    os.environ["THESIS_DIRECT_SPLIT_RUNS"] = str(num_seeds)

    if experiment_ids is None:
        raw = (os.environ.get("THESIS_CROSS_EXPERIMENTS") or "01,03,04,05,06").strip()
        experiment_ids = [x.strip() for x in raw.split(",") if x.strip()]

    if model_keys is None:
        raw = (os.environ.get("THESIS_CROSS_MODELS") or ",".join(MODEL_REGISTRY.keys())).strip()
        model_keys = [x.strip() for x in raw.split(",") if x.strip()]

    models = []
    for mk in model_keys:
        if mk not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {mk}. Available: {list(MODEL_REGISTRY.keys())}")
        models.append(MODEL_REGISTRY[mk])

    # ── Prepare data conditions ───────────────────────────────────────
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

    resolved_aug_model = str(prep07_aug.get("augmentation_model") or "").strip()
    if resolved_aug_model:
        _log(f"Resolved augmentation model: {resolved_aug_model}")
    elif skip_augmentation:
        _log("Resolved augmentation model: N/A (augmentation skipped)")
    else:
        _log("Resolved augmentation model: unavailable")

    conditions = _build_conditions(
        condition_sources=condition_sources,
        condition_keys=condition_keys,
    )
    if not conditions:
        raise RuntimeError("No data conditions available. Check exp07/exp08 split artifacts.")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_file).expanduser() if checkpoint_file else (
        COMPARISON_DIR / "cross_comparison_progress_latest.json"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

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
            rows = cp_rows
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
    print("  CROSS-DATA × MULTI-MODEL COMPARISON")
    print(f"  Models: {', '.join(m['display_name'] for m in models)}")
    print(f"  Experiments: {', '.join(f'exp{e}' for e in experiment_ids)}")
    print(f"  Data conditions: {len(conditions)} ({sum(1 for c in conditions if c['source']=='exp07')} from exp07, "
          f"{sum(1 for c in conditions if c['source']=='exp08')} from exp08, "
          f"{sum(1 for c in conditions if c['source']=='exp07+aug')} from exp07+aug)")
    print(f"  Seed count per multi-seed step: {num_seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 75)

    for model_info in models:
        model_id = model_info["model_id"]
        model_display = model_info["display_name"]
        _set_model_env(model_id)

        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            print(f"\n{'─'*60}")
            print(f"  {model_display} / exp{exp_id}: {exp_name}  ({len(conditions)} conditions)")
            print(f"{'─'*60}")

            for cond in conditions:
                run_key = _run_key(model_id, exp_id, cond["key"])
                if run_key in completed_keys:
                    _log(
                        f"Skip completed | {model_display} | exp{exp_id} | {cond['short_label']}"
                    )
                    continue

                run_counter += 1
                t0 = time.time()
                _log(
                    f"Run {run_counter}/{total_runs} | {model_display} | exp{exp_id} | "
                    f"{cond['short_label']}"
                )

                _set_presplit_env(cond["train_path"], cond["eval_path"])
                try:
                    mod = _import_experiment(exp_id)
                    payload = mod.run()
                    metrics = _extract_metrics(payload)
                except Exception as exc:
                    traceback.print_exc()
                    metrics = {
                        "f1": None, "precision": None, "recall": None,
                        "status": f"error: {exc}",
                    }
                    payload = {}
                finally:
                    _clear_presplit_env()

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
                    "condition_label": cond["label"],
                    "condition_short": cond["short_label"],
                    "condition_description": cond["description"],
                    "is_baseline": cond["is_baseline"],
                    "f1": metrics.get("f1"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "status": metrics.get("status"),
                    "result_file": payload.get("result_file", ""),
                    "metrics_file": payload.get("metrics_file", ""),
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

    # ==================================================================
    # Post-processing: build analytical sheets
    # ==================================================================

    # ── 1. Summary pivot: model × experiment × data-source best-F1 ───
    pivot_rows: list[dict] = []
    for model_info in models:
        m_name = model_info["display_name"]
        m_results = results_df[results_df["model_name"] == m_name]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            exp_results = m_results[m_results["experiment_id"] == f"exp{exp_id}"]
            row: dict[str, Any] = {
                "model": m_name,
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
            }
            for _, r in exp_results.iterrows():
                row[r["condition_short"]] = r.get("f1")
            # Best F1 across all conditions
            f1s = pd.to_numeric(exp_results["f1"], errors="coerce").dropna()
            row["best_f1"] = float(f1s.max()) if not f1s.empty else None
            row["best_condition"] = (
                exp_results.loc[f1s.idxmax(), "condition_short"]
                if not f1s.empty else "N/A"
            )
            pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows)

    # ── 2. Exp07 deltas: each variant vs exp07 baseline ──────────────
    meta07 = _load_exp07_meta()
    baseline07_key = f"exp07_{meta07.get('baseline_variant', '')}"
    delta07_rows: list[dict] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            baseline_row = results_df[
                (results_df["model_name"] == m_name)
                & (results_df["experiment_id"] == f"exp{exp_id}")
                & (results_df["condition_key"] == baseline07_key)
            ]
            if baseline_row.empty:
                continue
            b = baseline_row.iloc[0]
            exp07_results = results_df[
                (results_df["model_name"] == m_name)
                & (results_df["experiment_id"] == f"exp{exp_id}")
                & (results_df["data_source"] == "exp07")
                & (results_df["condition_key"] != baseline07_key)
            ]
            for _, a in exp07_results.iterrows():
                def _d(metric: str):
                    bv, av = _to_float(b.get(metric)), _to_float(a.get(metric))
                    return (av - bv) if av is not None and bv is not None else None
                delta07_rows.append({
                    "model": m_name,
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "variant": a["condition_short"],
                    "baseline_f1": _to_float(b.get("f1")),
                    "variant_f1": _to_float(a.get("f1")),
                    "delta_f1": _d("f1"),
                    "baseline_precision": _to_float(b.get("precision")),
                    "variant_precision": _to_float(a.get("precision")),
                    "delta_precision": _d("precision"),
                    "baseline_recall": _to_float(b.get("recall")),
                    "variant_recall": _to_float(a.get("recall")),
                    "delta_recall": _d("recall"),
                })
    deltas07_df = pd.DataFrame(delta07_rows) if delta07_rows else pd.DataFrame()

    # ── 3. Exp08 deltas: augmented vs baseline ───────────────────────
    delta08_rows: list[dict] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            b_row = results_df[
                (results_df["model_name"] == m_name)
                & (results_df["experiment_id"] == f"exp{exp_id}")
                & (results_df["condition_key"] == "exp08_baseline")
            ]
            a_row = results_df[
                (results_df["model_name"] == m_name)
                & (results_df["experiment_id"] == f"exp{exp_id}")
                & (results_df["condition_key"] == "exp08_augmented")
            ]
            if b_row.empty or a_row.empty:
                continue
            b, a = b_row.iloc[0], a_row.iloc[0]

            def _d(metric: str):
                bv, av = _to_float(b.get(metric)), _to_float(a.get(metric))
                return (av - bv) if av is not None and bv is not None else None

            delta08_rows.append({
                "model": m_name,
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
                "baseline_f1": _to_float(b.get("f1")),
                "augmented_f1": _to_float(a.get("f1")),
                "delta_f1": _d("f1"),
                "baseline_precision": _to_float(b.get("precision")),
                "augmented_precision": _to_float(a.get("precision")),
                "delta_precision": _d("precision"),
                "baseline_recall": _to_float(b.get("recall")),
                "augmented_recall": _to_float(a.get("recall")),
                "delta_recall": _d("recall"),
            })
    deltas08_df = pd.DataFrame(delta08_rows) if delta08_rows else pd.DataFrame()

    # ── 3b. Exp07+Aug deltas: augmented variant vs non-augmented exp07 variant
    delta07aug_rows: list[dict] = []
    for model_info in models:
        m_name = model_info["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            # For each exp07 variant, compare with its augmented counterpart
            exp07_results = results_df[
                (results_df["model_name"] == m_name)
                & (results_df["experiment_id"] == f"exp{exp_id}")
                & (results_df["data_source"] == "exp07")
            ]
            for _, b in exp07_results.iterrows():
                variant = str(b["condition_key"]).replace("exp07_", "")
                aug_key = f"exp07aug_{variant}"
                a_row = results_df[
                    (results_df["model_name"] == m_name)
                    & (results_df["experiment_id"] == f"exp{exp_id}")
                    & (results_df["condition_key"] == aug_key)
                ]
                if a_row.empty:
                    continue
                a = a_row.iloc[0]

                def _d(metric: str):
                    bv, av = _to_float(b.get(metric)), _to_float(a.get(metric))
                    return (av - bv) if av is not None and bv is not None else None

                delta07aug_rows.append({
                    "model": m_name,
                    "experiment_id": f"exp{exp_id}",
                    "experiment_name": exp_name,
                    "split_variant": b["condition_short"],
                    "split_only_f1": _to_float(b.get("f1")),
                    "split_plus_aug_f1": _to_float(a.get("f1")),
                    "delta_f1": _d("f1"),
                    "split_only_precision": _to_float(b.get("precision")),
                    "split_plus_aug_precision": _to_float(a.get("precision")),
                    "delta_precision": _d("precision"),
                    "split_only_recall": _to_float(b.get("recall")),
                    "split_plus_aug_recall": _to_float(a.get("recall")),
                    "delta_recall": _d("recall"),
                })
    deltas07aug_df = pd.DataFrame(delta07aug_rows) if delta07aug_rows else pd.DataFrame()

    # ── 4. Model comparison: same condition, head-to-head ────────────
    model_cmp_rows: list[dict] = []
    if len(models) == 2:
        m0, m1 = models[0]["display_name"], models[1]["display_name"]
        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, f"Experiment {exp_id}")
            for cond in conditions:
                r0 = results_df[
                    (results_df["model_name"] == m0)
                    & (results_df["experiment_id"] == f"exp{exp_id}")
                    & (results_df["condition_key"] == cond["key"])
                ]
                r1 = results_df[
                    (results_df["model_name"] == m1)
                    & (results_df["experiment_id"] == f"exp{exp_id}")
                    & (results_df["condition_key"] == cond["key"])
                ]
                if r0.empty or r1.empty:
                    continue
                v0, v1 = r0.iloc[0], r1.iloc[0]
                f1_0 = _to_float(v0.get("f1"))
                f1_1 = _to_float(v1.get("f1"))
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
                    f"{m0}_precision": _to_float(v0.get("precision")),
                    f"{m1}_precision": _to_float(v1.get("precision")),
                    f"{m0}_recall": _to_float(v0.get("recall")),
                    f"{m1}_recall": _to_float(v1.get("recall")),
                })
    model_cmp_df = pd.DataFrame(model_cmp_rows) if model_cmp_rows else pd.DataFrame()

    # ── 5. Variant summary across models and experiments ─────────────
    variant_rows: list[dict] = []
    for cond in conditions:
        cond_results = results_df[results_df["condition_key"] == cond["key"]]
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
    meta08 = _load_exp08_meta()

    model_names_str = ", ".join(m["display_name"] for m in models)
    exp07_variant_str = "; ".join(
        vm["label"] for vm in meta07.get("variants", [])
    )
    doc_df = build_thesis_documentation_df(
        "cross_comparison",
        "Cross-Data × Multi-Model Comparison (Exp07 & Exp08 → Exp03-06)",
        extra_rows=[
            {"Section": "Design", "Key": "Models",
             "Value": model_names_str},
            {"Section": "Design", "Key": "Exp07 Conditions",
             "Value": f"{len([c for c in conditions if c['source']=='exp07'])} variants: {exp07_variant_str}"},
            {"Section": "Design", "Key": "Exp08 Conditions",
             "Value": "Baseline (no augmentation) vs Augmented (LLM mask-fill)"},
            {"Section": "Design", "Key": "Exp07+Aug Conditions",
             "Value": f"{len([c for c in conditions if c['source']=='exp07+aug'])} variants: "
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
        if not model_cmp_df.empty:
            model_cmp_df.to_excel(writer, sheet_name="model_comparison", index=False)
        if not variant_summary_df.empty:
            variant_summary_df.to_excel(writer, sheet_name="variant_summary", index=False)
        details_df = results_df[[
            "model_name", "experiment_id", "experiment_name",
            "data_source", "condition_label", "condition_description",
            "f1", "precision", "recall", "status",
            "result_file", "metrics_file", "elapsed_seconds",
        ]].copy()
        details_df.to_excel(writer, sheet_name="experiment_details", index=False)
        doc_df.to_excel(writer, sheet_name="documentation", index=False)

    latest_xlsx = COMPARISON_DIR / "cross_comparison_latest.xlsx"
    if latest_xlsx.exists():
        latest_xlsx.unlink()
    shutil.copy2(xlsx_path, latest_xlsx)

    # ── JSON ──────────────────────────────────────────────────────────
    payload_out: dict[str, Any] = {
        "name": "Cross-Data × Multi-Model Comparison (Exp07 & Exp08 → Exp03-06)",
        "description": (
            f"Runs experiments 03–06 with {len(models)} models × "
            f"{len(conditions)} data conditions from exp07 and exp08. "
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
        "num_conditions": len(conditions),
        "num_experiments": len(experiment_ids),
        "total_runs": total_runs,
        "elapsed_seconds": round(total_elapsed, 1),
        "results": results_df.to_dict(orient="records"),
        "summary_pivot": pivot_df.to_dict(orient="records") if not pivot_df.empty else [],
        "deltas_exp07": deltas07_df.to_dict(orient="records") if not deltas07_df.empty else [],
        "deltas_exp08": deltas08_df.to_dict(orient="records") if not deltas08_df.empty else [],
        "deltas_exp07_aug": deltas07aug_df.to_dict(orient="records") if not deltas07aug_df.empty else [],
        "model_comparison": model_cmp_df.to_dict(orient="records") if not model_cmp_df.empty else [],
        "variant_summary": variant_summary_df.to_dict(orient="records") if not variant_summary_df.empty else [],
        "xlsx": str(xlsx_path),
        "xlsx_latest": str(latest_xlsx),
        "status": "ok",
    }
    json_path = COMPARISON_DIR / f"cross_comparison_{ts}.json"
    json_path.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    latest_json = COMPARISON_DIR / "cross_comparison_latest.json"
    latest_json.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

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

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  CROSS-DATA × MULTI-MODEL COMPARISON — RESULTS")
    print(f"{'='*75}")

    for model_info in models:
        m_name = model_info["display_name"]
        print(f"\n  Model: {m_name}")
        m_results = results_df[results_df["model_name"] == m_name]
        for exp_id in experiment_ids:
            exp_results = m_results[m_results["experiment_id"] == f"exp{exp_id}"]
            if exp_results.empty:
                continue
            numeric_f1 = pd.to_numeric(exp_results["f1"], errors="coerce")
            if numeric_f1.isna().all():
                print(
                    f"    exp{exp_id} ({EXP_NAMES.get(exp_id, exp_id)}): "
                    f"best F1=N/A (all runs failed)"
                )
                continue
            best = exp_results.loc[numeric_f1.idxmax()]
            print(
                f"    exp{exp_id} ({EXP_NAMES.get(exp_id, exp_id)}): "
                f"best F1={_fmt(_to_float(best.get('f1')))} "
                f"[{best.get('condition_short', 'N/A')}]"
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
            "Run experiments 03-06 with data from exp07 (split strategies) and exp08 "
            "(LLM augmentation), comparing dicta-il/dictabert and dicta-il/BEREL_3.0."
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
        default=(os.environ.get("THESIS_CROSS_EXPERIMENTS") or "01,03,04,05,06").strip(),
        help="Comma-separated experiment IDs (default: 01,03,04,05,06).",
    )
    parser.add_argument(
        "--models",
        default=(os.environ.get("THESIS_CROSS_MODELS") or "dictabert,berel").strip(),
        help="Comma-separated model keys (default: dictabert,berel). Available: "
             + ", ".join(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=int((os.environ.get("THESIS_CROSS_NUM_SEEDS") or "3").strip()),
        help="Seed count for exp07/exp08 (default: 3).",
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
    args = parser.parse_args()

    experiment_ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    condition_sources = [x.strip() for x in args.condition_sources.split(",") if x.strip()]
    condition_keys = [x.strip() for x in args.condition_keys.split(",") if x.strip()]

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
    )
