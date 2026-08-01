"""
model_cleanup.py — Remove heavy training checkpoints after any experiment
=========================================================================

Transformer and cascaded training writes large checkpoint trees under
``outputs/trainer_checkpoints/``, ``outputs/tmp_trainer/``, ``outputs/trained_models/``,
and similar paths. Thesis runs keep **Excel/JSON metrics** and error-analysis workbooks;
weights are dropped by default to save disk.

Behavior
--------
When ``THESIS_DELETE_MODELS_AFTER_TRAIN`` is unset or ``1`` (default),
``cleanup_training_artifacts()`` removes known checkpoint directories after training.
Set the variable to ``0`` to keep weights (e.g. Hugging Face upload with
``THESIS_SAVE_TRAINED_MODELS=1``).
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

    Only removes paths listed in ``candidates`` and a few globs under ``core/``.
    Never deletes ``outputs/exp*/`` Excel or JSON result files.
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
        root / "core" / "tmp_trainer",
    ]
    scratch = (os.environ.get("THESIS_COLAB_TRAINER_SCRATCH") or "/content/thesis_trainer_scratch").strip()
    if scratch:
        candidates.append(Path(scratch))
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


def cleanup_training_artifacts_if_enabled(project_root: Path | None = None) -> None:
    """Run cleanup and log a short message when anything was removed."""
    removed = cleanup_training_artifacts(project_root)
    if removed:
        print(
            f"[Disk cleanup] Removed {len(removed)} checkpoint path(s) "
            f"(THESIS_DELETE_MODELS_AFTER_TRAIN=1)."
        )


def colab_trainer_scratch_dir(unique_run_name: str) -> str:
    """Prefer Colab VM disk for Trainer temp files (avoid filling Google Drive)."""
    base = (os.environ.get("THESIS_COLAB_TRAINER_SCRATCH") or "/content/thesis_trainer_scratch").strip()
    path = os.path.join(base, unique_run_name)
    os.makedirs(path, exist_ok=True)
    return path


def use_disk_minimal_colab_training() -> bool:
    """When true, Colab training skips checkpoint saves on Drive (metrics-only runs)."""
    return os.environ.get("THESIS_RUN_ENV") == "colab" and should_delete_models_after_train()
