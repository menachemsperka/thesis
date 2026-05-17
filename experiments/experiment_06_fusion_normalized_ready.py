"""
experiment_06_fusion_normalized_ready.py — Calibrated Confidence Fusion (Ready)

Temperature-scaled calibration applied to Exp01 + Exp04 ready outputs.
Learns calibration temperatures from the evaluation data, then fuses
using calibrated confidence comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fusion_ready_sources import run_ready_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from confidence_calibration import apply_temperature_scaling, calibrate_pair  # type: ignore


def _calibrated_fusion(merged):
    """Learn calibration temperatures, then fuse on calibrated confidences."""
    y_true_regular = (merged["regular_pred_label"] == merged["true_label"]).astype(int).values
    y_true_cascade = (merged["cascade_pred_label"] == merged["cascade_true_label"]).astype(int).values

    calibration_result = calibrate_pair(
        y_true_regular,
        merged["regular_prob"].values,
        y_true_cascade,
        merged["cascade_prob"].values,
    )

    regular_temp = calibration_result["regular_temperature"]
    cascade_temp = calibration_result["cascade_temperature"]

    merged["calibrated_regular_prob"] = merged["regular_prob"].apply(
        lambda x: apply_temperature_scaling(x, regular_temp)
    )
    merged["calibrated_cascade_prob"] = merged["cascade_prob"].apply(
        lambda x: apply_temperature_scaling(x, cascade_temp)
    )

    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values
    reg_cal = merged["calibrated_regular_prob"].values
    cas_cal = merged["calibrated_cascade_prob"].values
    agree = ~merged["disagree"].values

    use_regular = agree | (reg_cal >= cas_cal)
    merged["fused_pred_label"] = np.where(use_regular, reg_label, cas_label)
    merged["selected_source"] = np.where(
        agree, "agree",
        np.where(use_regular, "regular", "cascade"),
    )
    merged["selected_confidence"] = np.where(use_regular, reg_cal, cas_cal)

    merged.attrs["regular_temperature"] = regular_temp
    merged.attrs["cascade_temperature"] = cascade_temp
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_calibrated_fusion,
        experiment_id="exp06_normalized_ready",
        experiment_name="Fusion with Calibrated Confidence (Ready)",
        description=(
            "No-retraining fusion with temperature-scaled calibration from "
            "Exp01 + Exp04 ready outputs."
        ),
        result_basename="fusion_normalized_ready",
        extra_info={"calibration": "temperature_scaling"},
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_normalized_ready] F1={f1_str}")
