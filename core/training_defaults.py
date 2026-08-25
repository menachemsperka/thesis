"""
Fair cross-model training defaults for ``run_cross_data_model_comparison.py``.

Reference profile: Colab 150-sentence pilot where DictaBERT and BEREL completed
(Aug 2025) before multilingual-only overrides. All models in a study should use
the same Exp01 / Exp04 settings unless you explicitly override env vars.
"""

from __future__ import annotations

import os
from typing import Any

# Exp01 (Hugging Face Trainer) — matches completed Hebrew cross-comparison runs.
EXP01_PROFILE_ENV: dict[str, str] = {
    "THESIS_NUM_EPOCHS": "3",
    "THESIS_LEARNING_RATE": "5e-5",
    "THESIS_TRAINER_FP16": "1",
    "THESIS_TRAINER_WEIGHT_DECAY": "0",
    "THESIS_TRAINER_BEST_F1_CHECKPOINT": "0",
}

# Exp04 (cascaded pipeline in core/auc_cascaded_pipeline.py) — same for every encoder.
EXP04_PROFILE_ENV: dict[str, str] = {
    "THESIS_EXP04_EPOCHS": "10",
}

# Fixed Exp04 values (no env hook in code today); documented for manifests and overview.
EXP04_FIXED_DEFAULTS: dict[str, Any] = {
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "encoder_lr": 2e-5,
    "head_lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "grad_accum_steps": 2,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-3,
}

# Hugging Face Trainer defaults used when not set in TrainingArguments (Exp01).
EXP01_HF_IMPLICIT_DEFAULTS: dict[str, Any] = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "warmup_ratio": 0.0,
    "save_strategy": "no",
    "eval_strategy": "no",
    "load_best_model_at_end": False,
    "balanced_class_weights": False,
}


def apply_fair_comparison_training_defaults() -> dict[str, str]:
    """
    Apply cross-model defaults via ``os.environ.setdefault`` (user overrides win).

    Returns the effective string values for keys that were set or already present.
    """
    effective: dict[str, str] = {}
    for key, default in {**EXP01_PROFILE_ENV, **EXP04_PROFILE_ENV}.items():
        existing = (os.environ.get(key) or "").strip()
        if existing:
            effective[key] = existing
        else:
            os.environ[key] = default
            effective[key] = default
    return effective


def snapshot_training_hyperparameters() -> dict[str, Any]:
    """Build a JSON-serializable snapshot for ``run_manifest.json``."""

    def _float(key: str, default: float) -> float:
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _int(key: str, default: int) -> int:
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _flag(key: str, default: bool = False) -> bool:
        raw = (os.environ.get(key) or "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    exp04_epochs = _int("THESIS_EXP04_EPOCHS", int(EXP04_PROFILE_ENV["THESIS_EXP04_EPOCHS"]))

    return {
        "profile": "fair_cross_model_comparison",
        "reference": "DictaBERT/BEREL 150-sent Colab completion (Exp01: 3 ep, 5e-5 LR)",
        "exp01_trainer": {
            "num_train_epochs": _float("THESIS_NUM_EPOCHS", float(EXP01_PROFILE_ENV["THESIS_NUM_EPOCHS"])),
            "learning_rate": _float("THESIS_LEARNING_RATE", 5e-5),
            "fp16": _flag("THESIS_TRAINER_FP16", default=True),
            "weight_decay": _float("THESIS_TRAINER_WEIGHT_DECAY", 0.0),
            "balanced_class_weights": _flag("THESIS_BALANCED_CLASS_WEIGHTS"),
            "best_f1_checkpoint_reload": _flag("THESIS_TRAINER_BEST_F1_CHECKPOINT"),
            **EXP01_HF_IMPLICIT_DEFAULTS,
        },
        "exp04_cascaded": {
            "epochs": exp04_epochs,
            "train_batch_size": _int(
                "THESIS_EXP04_TRAIN_BATCH", int(EXP04_FIXED_DEFAULTS["train_batch_size"])
            ),
            "eval_batch_size": _int(
                "THESIS_EXP04_EVAL_BATCH", int(EXP04_FIXED_DEFAULTS["eval_batch_size"])
            ),
            "grad_accum_steps": _int(
                "THESIS_EXP04_GRAD_ACCUM", int(EXP04_FIXED_DEFAULTS["grad_accum_steps"])
            ),
            **{
                k: EXP04_FIXED_DEFAULTS[k]
                for k in (
                    "encoder_lr",
                    "head_lr",
                    "weight_decay",
                    "warmup_fraction",
                    "max_grad_norm",
                    "early_stopping_patience",
                    "early_stopping_min_delta",
                )
            },
        },
        "notes": [
            "Exp01 and Exp04 use different epoch counts by design (3 vs 10).",
            "Do not mix checkpoint resume rows trained under a different profile.",
            "See cross_comparison_fair_training_overview.md for Colab commands.",
        ],
    }
