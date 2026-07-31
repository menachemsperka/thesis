"""
fusion_crf_ready_sources.py — Ready fusion for Experiment 10 CRF outputs
=========================================================================

Problem solved
--------------
Experiments ``10_fusion_ready`` and ``10_svm_ready`` must combine **two already-trained**
predictors without running GPU training again:

* **Regular BERT-CRF** → ``outputs/exp10_regular/`` Excel (``token_predictions`` sheet)
* **Cascaded CRF** → ``outputs/exp10_cascade/`` Excel (``detailed_results``, ``eval_mode=predicted``)

This module resolves those paths (from env vars or ``latest.json``) and delegates to
``fusion_ready_sources.run_ready_fusion``, which implements merge + metrics + error-analysis sheets.

Environment (set by ``run_cross_data_model_comparison.py`` when base cache hits)
---------------------------------------------------------------------------------
* ``THESIS_READY_EXP10_REGULAR_XLSX`` — metrics workbook from ``10_regular``
* ``THESIS_READY_EXP10_CASCADE_XLSX`` — metrics workbook from ``10_cascade``

Internally, paths are mapped to the Exp01/Exp04 env names expected by ``run_ready_fusion`` so
**no duplicate fusion math** is maintained in two places.

Teaching note
-------------
Compare this file to ``experiments/fusion_ready_sources.py`` docstring: the only difference is
*which output folders* supply the two prediction streams.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from fusion_ready_sources import run_ready_fusion


def _resolve_exp10_source(exp_folder: str, env_var: str) -> Path:
    """Find the latest Exp10 metrics Excel for *exp_folder*, or honor an explicit env path."""
    explicit = (os.environ.get(env_var) or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points to a missing file: {p}")
        return p
    latest_json = Path("outputs") / exp_folder / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(
            f"Cannot auto-resolve {exp_folder} output. Set {env_var} or run that experiment first."
        )
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    metrics_file = payload.get("metrics_file")
    if not metrics_file:
        raise ValueError(f"metrics_file missing in {latest_json}")
    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(f"metrics_file from {latest_json} not found: {p}")
    return p


def run_ready_fusion_crf(
    *,
    strategy_fn: Callable,
    experiment_id: str,
    experiment_name: str,
    description: str,
    result_basename: str,
    extra_info: dict | None = None,
) -> dict:
    """
    Load Exp10 regular + cascade CRF workbooks, apply *strategy_fn*, save fusion results.

    Parameters match ``fusion_ready_sources.run_ready_fusion``; see that function for the
    required columns ``fused_pred_label``, ``selected_source``, ``selected_confidence``.
    """
    exp01_xlsx = _resolve_exp10_source("exp10_regular", "THESIS_READY_EXP10_REGULAR_XLSX")
    cascade_xlsx = _resolve_exp10_source("exp10_cascade", "THESIS_READY_EXP10_CASCADE_XLSX")

    os.environ["THESIS_READY_EXP01_XLSX"] = str(exp01_xlsx)
    os.environ["THESIS_READY_EXP04_XLSX"] = str(cascade_xlsx)
    try:
        payload = run_ready_fusion(
            strategy_fn=strategy_fn,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            result_basename=result_basename,
            cascade_source="exp04",
            extra_info=extra_info,
        )
    finally:
        os.environ.pop("THESIS_READY_EXP01_XLSX", None)
        os.environ.pop("THESIS_READY_EXP04_XLSX", None)
    payload["regular_crf_source"] = str(exp01_xlsx)
    payload["cascade_crf_source"] = str(cascade_xlsx)
    return payload
