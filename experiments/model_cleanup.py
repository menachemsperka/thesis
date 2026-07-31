"""
model_cleanup.py — Remove heavy training checkpoints after Experiment 10
========================================================================

Why this exists
---------------
Transformer fine-tuning writes **multi-gigabyte** checkpoint folders under
``outputs/trainer_checkpoints/``, ``outputs/tmp_trainer/``, and similar paths.
For thesis cross-comparison we only need **Excel metrics**, **JSON result manifests**, and
error-analysis workbooks—not every intermediate ``pytorch_model.bin``.

Behavior
--------
When ``THESIS_DELETE_MODELS_AFTER_TRAIN`` is ``1`` (default), ``cleanup_training_artifacts()``
deletes known checkpoint directories after ``10_regular`` and ``10_cascade`` complete.
Set the variable to ``0`` if you intentionally want to keep weights for Hugging Face upload
(see also ``THESIS_SAVE_TRAINED_MODELS`` in ``core/th_functions.py``).

Called from
-----------
* ``experiments/experiment_10_regular_ner_crf.py``
* ``experiments/experiment_10_cascaded_pipeline_crf.py``
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def should_delete_models_after_train() -> bool:
    """Return True unless the user disabled post-training cleanup via environment variable."""
    raw = (os.environ.get("THESIS_DELETE_MODELS_AFTER_TRAIN") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def cleanup_training_artifacts(project_root: Path | None = None) -> list[str]:
    """
    Delete trainer checkpoints and cascaded model bundles; return removed paths.

    This function is intentionally conservative: it only removes paths listed in
    ``candidates`` and a few globs under ``core/``. It never deletes ``outputs/exp10_*``
    Excel or JSON results.
    """
    if not should_delete_models_after_train():
        return []

    root = project_root or Path(__file__).resolve().parents[1]
    removed: list[str] = []

    candidates = [
        root / "outputs" / "trainer_checkpoints",
        root / "outputs" / "tmp_trainer",
        root / "outputs" / "trained_models",
        root / "core" / "cascaded_model_artifact",
        root / "core" / "cascaded_model_artifact_crf",
    ]
    for path in candidates:
        if path.exists():
            try:
                shutil.rmtree(path)
                removed.append(str(path))
            except Exception:
                pass

    core_dir = root / "core"
    for pattern in ("checkpoint-*", "cascaded_pipeline_*.pt", "cascaded_pipeline_*.bin"):
        for path in core_dir.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(str(path))
            except Exception:
                pass
    return removed
