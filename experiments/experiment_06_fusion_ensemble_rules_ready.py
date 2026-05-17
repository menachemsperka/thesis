"""
experiment_06_fusion_ensemble_rules_ready.py — Ensemble Rules Fusion (Ready)

Rule-based fusion from Exp01 + Exp04 ready outputs, no retraining.

Rules:
1. Predictions agree → keep (high confidence).
2. Large confidence gap (> AGREEMENT_THRESHOLD) → trust higher confidence.
3. Small confidence gap → default to cascade (more conservative).
"""
from __future__ import annotations

import numpy as np

from fusion_ready_sources import run_ready_fusion

AGREEMENT_THRESHOLD = 0.2


def _ensemble_rules_fusion(merged):
    """Apply ensemble decision rules."""
    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values
    reg_prob = merged["regular_prob"].values
    cas_prob = merged["cascade_prob"].values
    agree = ~merged["disagree"].values
    conf_diff = np.abs(reg_prob - cas_prob)

    # Start with cascade as default for disagreements
    fused = cas_label.copy()
    source = np.full(len(merged), "rules_cascade_conservative", dtype=object)
    confidence = cas_prob.copy()

    # Rule 2a: large confidence gap → higher confidence wins
    large_gap = (~agree) & (conf_diff > AGREEMENT_THRESHOLD)
    reg_wins = large_gap & (reg_prob > cas_prob)
    cas_wins = large_gap & (~reg_wins)
    fused = np.where(reg_wins, reg_label, fused)
    source = np.where(reg_wins, "rules_regular_confident", source)
    confidence = np.where(reg_wins, reg_prob, confidence)
    source = np.where(cas_wins, "rules_cascade_confident", source)
    confidence = np.where(cas_wins, cas_prob, confidence)

    # Rule 1: agreement
    fused = np.where(agree, reg_label, fused)
    source = np.where(agree, "agree", source)
    confidence = np.where(agree, np.maximum(reg_prob, cas_prob), confidence)

    merged["fused_pred_label"] = fused
    merged["selected_source"] = source
    merged["selected_confidence"] = confidence
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_ensemble_rules_fusion,
        experiment_id="exp06_ensemble_ready",
        experiment_name="Ensemble Rules Fusion (Ready)",
        description=(
            "No-retraining rule-based ensemble fusion from Exp01 + Exp04 ready outputs. "
            f"Confidence gap threshold: {AGREEMENT_THRESHOLD}."
        ),
        result_basename="fusion_ensemble_rules_ready",
        extra_info={"agreement_threshold": AGREEMENT_THRESHOLD},
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_ensemble_ready] F1={f1_str}")
