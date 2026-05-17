"""
experiment_06_fusion_ready.py — Base Fusion (Ready from Exp01 + Exp04)

Confidence-comparison fusion without retraining: when models disagree,
select the prediction with the higher confidence score.
"""
from __future__ import annotations

import numpy as np

from fusion_ready_sources import run_ready_fusion


def _confidence_fusion(merged):
    """Base fusion: agree → keep, disagree → higher confidence wins."""
    regular_label = merged["regular_pred_label"].values
    cascade_label = merged["cascade_pred_label"].values
    regular_prob = merged["regular_prob"].values
    cascade_prob = merged["cascade_prob"].values
    agree = ~merged["disagree"].values

    use_regular = agree | (regular_prob >= cascade_prob)
    merged["fused_pred_label"] = np.where(use_regular, regular_label, cascade_label)
    merged["selected_source"] = np.where(
        agree, "agree",
        np.where(use_regular, "regular", "cascade"),
    )
    merged["selected_confidence"] = np.where(
        use_regular, regular_prob, cascade_prob,
    )
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_confidence_fusion,
        experiment_id="exp06_ready",
        experiment_name="Fusion of Regular+Cascaded (Ready)",
        description=(
            "No-retraining confidence-comparison fusion from Exp01 (regular NER) "
            "and Exp04 (cascaded pipeline) ready outputs."
        ),
        result_basename="fusion_ready",
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_ready] F1={f1_str}")
