"""
experiment_06_fusion_entropy_ready.py — Entropy-Weighted Fusion (Ready)

Uses pre-computed entropy values from Exp01 + Exp04 to weight confidence
scores during fusion.  No retraining required.
"""
from __future__ import annotations

import numpy as np

from fusion_ready_sources import run_ready_fusion


def _entropy_fusion(merged):
    """Weight predictions by (1 - normalized_entropy) for confidence weighting."""
    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values
    reg_prob = merged["regular_prob"].values
    cas_prob = merged["cascade_prob"].values
    reg_entropy = merged["regular_entropy"].values
    cas_entropy = merged["cascade_entropy"].values
    agree = ~merged["disagree"].values

    # Normalize entropies to weights: low entropy → high weight
    max_ent = np.maximum(reg_entropy, cas_entropy)
    safe_max = np.where(max_ent > 0, max_ent, 1.0)
    weight_regular = 1.0 - (reg_entropy / safe_max)
    weight_cascade = 1.0 - (cas_entropy / safe_max)

    weighted_regular = reg_prob * weight_regular
    weighted_cascade = cas_prob * weight_cascade

    use_regular = agree | (weighted_regular >= weighted_cascade)
    merged["fused_pred_label"] = np.where(use_regular, reg_label, cas_label)
    merged["selected_source"] = np.where(
        agree, "agree",
        np.where(use_regular, "entropy_regular", "entropy_cascade"),
    )
    merged["selected_confidence"] = np.where(
        use_regular, weighted_regular, weighted_cascade,
    )
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_entropy_fusion,
        experiment_id="exp06_entropy_ready",
        experiment_name="Entropy-Weighted Fusion (Ready)",
        description=(
            "No-retraining entropy-weighted fusion from Exp01 + Exp04 ready outputs. "
            "Low-entropy (high-certainty) predictions are weighted more heavily."
        ),
        result_basename="fusion_entropy_ready",
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_entropy_ready] F1={f1_str}")
