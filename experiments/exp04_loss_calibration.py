"""
exp04_loss_calibration.py — Resume-aware cascade loss-weight grids + env application.

Called from ``run_cross_data_model_comparison.py`` before the main comparison loop:

* ``exp04`` track — Exp04 cascade (Exp04 and ``06_*_oof`` training) -> ``THESIS_EXP04_LAMBDA_*``
* ``exp10`` track — Exp10 CRF cascade (``10_cascade`` and ``10_*_oof`` training) -> ``THESIS_EXP10_LAMBDA_*``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "exp04_lambda_grid_progress.json"
SELECTION_FILENAME = "exp04_lambda_selection.json"


@dataclass(frozen=True)
class CalibrationTrack:
    cache_dir_name: str
    selection_filename: str
    lambda_bio_env: str
    lambda_type_env: str
    cache_dir_env: str
    log_tag: str


CALIBRATION_TRACKS: dict[str, CalibrationTrack] = {
    "exp04": CalibrationTrack(
        cache_dir_name="exp04_lambda_grid_cache",
        selection_filename=SELECTION_FILENAME,
        lambda_bio_env="THESIS_EXP04_LAMBDA_BIO",
        lambda_type_env="THESIS_EXP04_LAMBDA_TYPE",
        cache_dir_env="THESIS_EXP04_LAMBDA_CACHE_DIR",
        log_tag="exp04-calibration",
    ),
    "exp10": CalibrationTrack(
        cache_dir_name="exp10_lambda_grid_cache",
        selection_filename="exp10_lambda_selection.json",
        lambda_bio_env="THESIS_EXP10_LAMBDA_BIO",
        lambda_type_env="THESIS_EXP10_LAMBDA_TYPE",
        cache_dir_env="THESIS_EXP10_LAMBDA_CACHE_DIR",
        log_tag="exp10-calibration",
    ),
}


def calibration_cache_dir(comparison_dir: Path, track: str = "exp04") -> Path:
    d = comparison_dir / CALIBRATION_TRACKS[track].cache_dir_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pair_key(lambda_bio: float, lambda_type: float) -> str:
    return f"{float(lambda_bio):g}_{float(lambda_type):g}"


def load_selection(cache_dir: Path, track: str = "exp04") -> dict[str, Any] | None:
    path = cache_dir / CALIBRATION_TRACKS[track].selection_filename
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def apply_selection_to_env(selection: dict[str, Any], track: str = "exp04") -> tuple[float, float]:
    spec = CALIBRATION_TRACKS[track]
    bio = selection.get("selected_lambda_bio")
    typ = selection.get("selected_lambda_type")
    if bio is None or typ is None:
        raise ValueError(f"Incomplete lambda selection: {selection}")
    os.environ[spec.lambda_bio_env] = str(bio)
    os.environ[spec.lambda_type_env] = str(typ)
    return float(bio), float(typ)


def ensure_loss_weights_calibrated(
    comparison_dir: Path,
    *,
    track: str,
    resume: bool = True,
    force_rebuild: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    """
    Run or resume the validation grid for *track*; set its λ env vars from the selection file.

    Skips grid when ``resume`` and a valid selection file exists (unless ``force_rebuild``).
    """
    spec = CALIBRATION_TRACKS[track]
    cache_dir = calibration_cache_dir(comparison_dir, track)
    os.environ[spec.cache_dir_env] = str(cache_dir)

    if not force_rebuild and resume:
        existing = load_selection(cache_dir, track)
        if existing and existing.get("selected_lambda_bio") is not None:
            bio, typ = apply_selection_to_env(existing, track)
            log_fn(
                f"[{spec.log_tag}] Reusing λ_bio={bio}, λ_type={typ} "
                f"from {cache_dir / spec.selection_filename}"
            )
            return {"source": "cached_selection", "cache_dir": str(cache_dir), **existing}

    from experiment_04_loss_weight_grid import run_grid

    result = run_grid(cache_dir=cache_dir, resume=resume, force_rebuild=force_rebuild, track=track)
    selection = {
        "cascade_track": track,
        "selected_lambda_bio": result.get("selected_lambda_bio"),
        "selected_lambda_type": result.get("selected_lambda_type"),
        "selected_validation_pipeline_span_f1": result.get("selected_validation_pipeline_span_f1"),
        "baseline_10_5_f1": result.get("baseline_10_5_f1"),
        "grid_excel": result.get("grid_excel"),
        "selection_metric": (result.get("training_parameters") or {}).get("selection_metric"),
        "model": result.get("model"),
        "split_seed": (result.get("training_parameters") or {}).get("split_seed"),
    }
    sel_path = cache_dir / spec.selection_filename
    sel_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    apply_selection_to_env(selection, track)
    log_fn(f"[{spec.log_tag}] Selected λ_bio={selection['selected_lambda_bio']} "
           f"λ_type={selection['selected_lambda_type']} -> {sel_path}")
    return {"source": "grid_run", "cache_dir": str(cache_dir), **selection}


def ensure_exp04_loss_weights_calibrated(
    comparison_dir: Path,
    *,
    resume: bool = True,
    force_rebuild: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    return ensure_loss_weights_calibrated(
        comparison_dir, track="exp04", resume=resume, force_rebuild=force_rebuild, log_fn=log_fn
    )


def ensure_exp10_loss_weights_calibrated(
    comparison_dir: Path,
    *,
    resume: bool = True,
    force_rebuild: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    return ensure_loss_weights_calibrated(
        comparison_dir, track="exp10", resume=resume, force_rebuild=force_rebuild, log_fn=log_fn
    )
