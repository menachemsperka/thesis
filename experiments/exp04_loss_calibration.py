"""
exp04_loss_calibration.py — Resume-aware cascade loss-weight grids + env application.

Called from ``run_cross_data_model_comparison.py`` before the main comparison loop:

* ``exp04`` track — Exp04 cascade (Exp04 and ``06_*_oof`` training) -> ``THESIS_EXP04_LAMBDA_*``
* ``exp10`` track — Exp10 CRF cascade (``10_cascade`` and ``10_*_oof`` training) -> ``THESIS_EXP10_LAMBDA_*``
"""
from __future__ import annotations

from datetime import datetime
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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


def _sanitize_for_path(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    safe = safe.strip("._-")
    return safe or "model"


def _set_model_env(model_ref: str) -> tuple[str, bool]:
    model_name = str(model_ref or "").strip()
    if not model_name:
        raise ValueError("Calibration model reference is empty.")

    is_local = Path(model_name).exists()
    os.environ["THESIS_MODEL_NAME"] = model_name
    os.environ["THESIS_MODEL_LOCAL_ONLY"] = "1" if is_local else "0"
    if is_local:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
    return model_name, is_local


def _scope_payload(model_refs: list[str], split_seeds: list[int]) -> dict[str, Any]:
    return {
        "models": [str(m) for m in model_refs],
        "split_seeds": [int(s) for s in split_seeds],
        "num_models": len(model_refs),
        "num_split_seeds": len(split_seeds),
        "num_calibration_runs": len(model_refs) * len(split_seeds),
    }


def _selection_matches_scope(
    selection: dict[str, Any],
    model_refs: list[str],
    split_seeds: list[int],
) -> bool:
    target_models = [str(m) for m in model_refs]
    target_seeds = [int(s) for s in split_seeds]

    scope = selection.get("calibration_scope")
    if isinstance(scope, dict):
        got_models = [str(m) for m in (scope.get("models") or [])]
        got_seeds = [int(s) for s in (scope.get("split_seeds") or [])]
        return got_models == target_models and got_seeds == target_seeds

    # Backward compatibility with legacy single-run selections.
    if len(target_models) == 1 and len(target_seeds) == 1:
        got_model = str(selection.get("model") or "")
        got_seed = selection.get("split_seed")
        try:
            got_seed = int(got_seed)
        except Exception:
            return False
        return got_model == target_models[0] and got_seed == target_seeds[0]

    return False


def _aggregate_grid_runs(run_outputs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    for idx, run_out in enumerate(run_outputs, start=1):
        model = run_out.get("model")
        model_ref = run_out.get("model_ref")
        split_seed = run_out.get("split_seed")
        for row in run_out.get("grid_runs") or []:
            f1 = row.get("validation_pipeline_span_f1")
            if f1 is None:
                continue
            try:
                bio = float(row.get("lambda_bio"))
                typ = float(row.get("lambda_type"))
                f1 = float(f1)
            except Exception:
                continue
            long_rows.append(
                {
                    "run_index": idx,
                    "model": model,
                    "model_ref": model_ref,
                    "split_seed": int(split_seed),
                    "lambda_bio": bio,
                    "lambda_type": typ,
                    "validation_pipeline_span_f1": f1,
                }
            )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summary_df = long_df.groupby(["lambda_bio", "lambda_type"], as_index=False).agg(
        mean_validation_pipeline_span_f1=("validation_pipeline_span_f1", "mean"),
        std_validation_pipeline_span_f1=("validation_pipeline_span_f1", "std"),
        n_runs=("validation_pipeline_span_f1", "count"),
        min_validation_pipeline_span_f1=("validation_pipeline_span_f1", "min"),
        max_validation_pipeline_span_f1=("validation_pipeline_span_f1", "max"),
    )
    summary_df["std_validation_pipeline_span_f1"] = summary_df[
        "std_validation_pipeline_span_f1"
    ].fillna(0.0)
    summary_df = summary_df.sort_values(
        ["mean_validation_pipeline_span_f1", "lambda_bio", "lambda_type"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return long_df, summary_df


def _pick_best_lambda_pair(summary_df: pd.DataFrame, expected_runs: int) -> tuple[dict[str, Any] | None, str]:
    if summary_df.empty:
        return None, "none"

    complete = summary_df[summary_df["n_runs"] == int(expected_runs)]
    candidates = complete if not complete.empty else summary_df
    coverage_mode = "complete" if not complete.empty else "partial"

    best_row = max(
        candidates.to_dict(orient="records"),
        key=lambda r: (
            float(r["mean_validation_pipeline_span_f1"]),
            -float(r["lambda_bio"]),
            -float(r["lambda_type"]),
        ),
    )
    return best_row, coverage_mode


def _write_aggregate_workbook(
    cache_dir: Path,
    run_overview: list[dict[str, Any]],
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cache_dir / f"loss_weight_grid_aggregate_{ts}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(run_overview).to_excel(writer, sheet_name="calibration_runs", index=False)
        long_df.to_excel(writer, sheet_name="grid_runs_long", index=False)
        summary_df.to_excel(writer, sheet_name="pair_summary", index=False)
        if not summary_df.empty:
            heatmap = summary_df.pivot(
                index="lambda_bio",
                columns="lambda_type",
                values="mean_validation_pipeline_span_f1",
            )
            heatmap.to_excel(writer, sheet_name="heatmap_mean_f1")

    latest_path = cache_dir / "loss_weight_grid_aggregate_latest.xlsx"
    latest_path.write_bytes(out_path.read_bytes())
    return out_path


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
    model_refs: list[str] | None = None,
    split_seeds: list[int] | None = None,
    log_fn=print,
) -> dict[str, Any]:
    """
    Run or resume the validation grid for *track*; set its λ env vars from the selection file.

    Skips grid when ``resume`` and a valid selection file exists (unless ``force_rebuild``).
    """
    spec = CALIBRATION_TRACKS[track]
    cache_dir = calibration_cache_dir(comparison_dir, track)
    os.environ[spec.cache_dir_env] = str(cache_dir)

    normalized_model_refs = [str(m).strip() for m in (model_refs or []) if str(m).strip()]
    if not normalized_model_refs:
        current = (os.environ.get("THESIS_MODEL_NAME") or "").strip()
        if current:
            normalized_model_refs = [current]

    normalized_split_seeds: list[int] = []
    for raw_seed in (split_seeds or []):
        try:
            normalized_split_seeds.append(int(raw_seed))
        except Exception:
            continue
    if not normalized_split_seeds:
        raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
        try:
            normalized_split_seeds = [int(raw)]
        except ValueError:
            normalized_split_seeds = [42]

    if not normalized_model_refs:
        # Let configure_model_environment() fall back to default encoder.
        normalized_model_refs = ["dicta-il/dictabert"]

    scope_payload = _scope_payload(normalized_model_refs, normalized_split_seeds)

    if not force_rebuild and resume:
        existing = load_selection(cache_dir, track)
        if (
            existing
            and existing.get("selected_lambda_bio") is not None
            and _selection_matches_scope(existing, normalized_model_refs, normalized_split_seeds)
        ):
            bio, typ = apply_selection_to_env(existing, track)
            log_fn(
                f"[{spec.log_tag}] Reusing λ_bio={bio}, λ_type={typ} "
                f"from {cache_dir / spec.selection_filename}"
            )
            return {"source": "cached_selection", "cache_dir": str(cache_dir), **existing}

    from experiment_04_loss_weight_grid import run_grid

    saved_env = {
        "THESIS_MODEL_NAME": os.environ.get("THESIS_MODEL_NAME"),
        "THESIS_MODEL_LOCAL_ONLY": os.environ.get("THESIS_MODEL_LOCAL_ONLY"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "THESIS_SPLIT_SEED": os.environ.get("THESIS_SPLIT_SEED"),
    }

    run_overview: list[dict[str, Any]] = []
    try:
        multi_scope = len(normalized_model_refs) > 1 or len(normalized_split_seeds) > 1
        for model_idx, model_ref in enumerate(normalized_model_refs, start=1):
            _set_model_env(model_ref)
            for split_seed in normalized_split_seeds:
                os.environ["THESIS_SPLIT_SEED"] = str(split_seed)
                run_cache = cache_dir
                if multi_scope:
                    model_tag = _sanitize_for_path(model_ref)
                    run_cache = cache_dir / "runs" / f"m{model_idx:02d}_{model_tag}" / f"seed_{int(split_seed)}"

                result = run_grid(
                    cache_dir=run_cache,
                    resume=resume,
                    force_rebuild=force_rebuild,
                    track=track,
                )
                run_overview.append(
                    {
                        "model_ref": model_ref,
                        "model": result.get("model") or model_ref,
                        "split_seed": int(split_seed),
                        "cache_dir": str(run_cache),
                        "grid_excel": result.get("grid_excel"),
                        "selected_lambda_bio": result.get("selected_lambda_bio"),
                        "selected_lambda_type": result.get("selected_lambda_type"),
                        "selected_validation_pipeline_span_f1": result.get("selected_validation_pipeline_span_f1"),
                        "status": result.get("status"),
                        "grid_runs": result.get("grid_runs") or [],
                    }
                )
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if not run_overview:
        raise RuntimeError(f"[{spec.log_tag}] No calibration runs were executed.")

    if len(run_overview) == 1:
        single = run_overview[0]
        single_metric = (
            "validation pipeline span F1 (predicted, final_optimised, after threshold sweep)"
        )
        single_grid_runs = single.get("grid_runs") or []
        baseline_f1 = None
        for row in single_grid_runs:
            try:
                if float(row.get("lambda_bio")) == 10.0 and float(row.get("lambda_type")) == 5.0:
                    f1 = row.get("validation_pipeline_span_f1")
                    baseline_f1 = float(f1) if f1 is not None else None
                    break
            except Exception:
                continue
        selection = {
            "cascade_track": track,
            "selected_lambda_bio": single.get("selected_lambda_bio"),
            "selected_lambda_type": single.get("selected_lambda_type"),
            "selected_validation_pipeline_span_f1": single.get("selected_validation_pipeline_span_f1"),
            "baseline_10_5_f1": baseline_f1,
            "grid_excel": single.get("grid_excel"),
            "selection_metric": single_metric,
            "model": single.get("model"),
            "split_seed": single.get("split_seed"),
            "calibration_scope": scope_payload,
            "coverage_mode": "single",
            "calibration_runs": [
                {
                    k: v for k, v in single.items() if k != "grid_runs"
                }
            ],
        }
    else:
        long_df, summary_df = _aggregate_grid_runs(run_overview)
        best_row, coverage_mode = _pick_best_lambda_pair(summary_df, expected_runs=len(run_overview))
        if best_row is None:
            raise RuntimeError(
                f"[{spec.log_tag}] Calibration produced no valid F1 values across "
                f"{len(run_overview)} runs."
            )

        aggregate_xlsx = _write_aggregate_workbook(cache_dir, run_overview, long_df, summary_df)

        baseline_rows = summary_df[
            (summary_df["lambda_bio"] == 10.0) & (summary_df["lambda_type"] == 5.0)
        ]
        baseline_f1 = (
            float(baseline_rows.iloc[0]["mean_validation_pipeline_span_f1"])
            if not baseline_rows.empty
            else None
        )

        selection = {
            "cascade_track": track,
            "selected_lambda_bio": float(best_row["lambda_bio"]),
            "selected_lambda_type": float(best_row["lambda_type"]),
            "selected_validation_pipeline_span_f1": float(best_row["mean_validation_pipeline_span_f1"]),
            "selected_validation_pipeline_span_f1_std": float(best_row["std_validation_pipeline_span_f1"]),
            "selected_validation_pipeline_span_f1_runs": int(best_row["n_runs"]),
            "baseline_10_5_f1": baseline_f1,
            "grid_excel": str(aggregate_xlsx),
            "selection_metric": (
                "mean validation pipeline span F1 across calibration runs "
                "(predicted, final_optimised, after threshold sweep)"
            ),
            "model": "multi",
            "split_seed": "multi",
            "calibration_scope": scope_payload,
            "coverage_mode": coverage_mode,
            "candidate_pairs_evaluated": int(len(summary_df)),
            "calibration_runs": [
                {
                    k: v for k, v in run.items() if k != "grid_runs"
                }
                for run in run_overview
            ],
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
