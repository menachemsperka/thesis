"""
exp04_loss_calibration.py — Resume-aware Exp04 loss-weight grid + env application.

Called from ``run_cross_data_model_comparison.py`` before the main comparison loop
when Exp04 and/or OOF fusion experiments need cascaded training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "exp04_lambda_grid_progress.json"
SELECTION_FILENAME = "exp04_lambda_selection.json"


def calibration_cache_dir(comparison_dir: Path) -> Path:
    d = comparison_dir / "exp04_lambda_grid_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pair_key(lambda_bio: float, lambda_type: float) -> str:
    return f"{float(lambda_bio):g}_{float(lambda_type):g}"


def load_selection(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / SELECTION_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def apply_selection_to_env(selection: dict[str, Any]) -> tuple[float, float]:
    bio = selection.get("selected_lambda_bio")
    typ = selection.get("selected_lambda_type")
    if bio is None or typ is None:
        raise ValueError(f"Incomplete lambda selection: {selection}")
    os.environ["THESIS_EXP04_LAMBDA_BIO"] = str(bio)
    os.environ["THESIS_EXP04_LAMBDA_TYPE"] = str(typ)
    return float(bio), float(typ)


def ensure_exp04_loss_weights_calibrated(
    comparison_dir: Path,
    *,
    resume: bool = True,
    force_rebuild: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    """
    Run or resume validation grid; set THESIS_EXP04_LAMBDA_* from ``lambda_selection.json``.

    Skips grid when ``resume`` and a valid selection file exists (unless ``force_rebuild``).
    """
    cache_dir = calibration_cache_dir(comparison_dir)
    os.environ["THESIS_EXP04_LAMBDA_CACHE_DIR"] = str(cache_dir)

    if not force_rebuild and resume:
        existing = load_selection(cache_dir)
        if existing and existing.get("selected_lambda_bio") is not None:
            bio, typ = apply_selection_to_env(existing)
            log_fn(
                f"[exp04-calibration] Reusing λ_bio={bio}, λ_type={typ} from {cache_dir / SELECTION_FILENAME}"
            )
            return {"source": "cached_selection", "cache_dir": str(cache_dir), **existing}

    from experiment_04_loss_weight_grid import run_grid

    result = run_grid(cache_dir=cache_dir, resume=resume, force_rebuild=force_rebuild)
    selection = {
        "selected_lambda_bio": result.get("selected_lambda_bio"),
        "selected_lambda_type": result.get("selected_lambda_type"),
        "selected_validation_pipeline_span_f1": result.get("selected_validation_pipeline_span_f1"),
        "baseline_10_5_f1": result.get("baseline_10_5_f1"),
        "grid_excel": result.get("grid_excel"),
        "selection_metric": (result.get("training_parameters") or {}).get("selection_metric"),
        "model": result.get("model"),
        "split_seed": (result.get("training_parameters") or {}).get("split_seed"),
    }
    sel_path = cache_dir / SELECTION_FILENAME
    sel_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    apply_selection_to_env(selection)
    log_fn(f"[exp04-calibration] Selected λ_bio={selection['selected_lambda_bio']} "
           f"λ_type={selection['selected_lambda_type']} -> {sel_path}")
    return {"source": "grid_run", "cache_dir": str(cache_dir), **selection}
