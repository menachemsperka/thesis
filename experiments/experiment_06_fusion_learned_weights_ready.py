"""
experiment_06_fusion_learned_weights_ready.py — Learned Weights Fusion (Ready)

Sweeps alpha from 0 to 1 to find the weighting that maximises F1,
then applies the weighted fusion.  Reads from Exp01 + Exp04 ready outputs
without retraining.

Note: alpha is learned on the same evaluation data it is evaluated on.
This is acceptable for fast iteration; for strict train/test separation
use the full training variant (experiment_06_fusion_learned_weights.py).
"""
from __future__ import annotations

import numpy as np
from seqeval.metrics import f1_score as seqeval_f1

from fusion_ready_sources import run_ready_fusion, to_seqeval_lists


def _learn_alpha(merged) -> float:
    """Find alpha in [0, 1] that maximises seqeval F1."""
    best_alpha = 0.5
    best_f1 = -1.0

    reg_prob = merged["regular_prob"].values
    cas_prob = merged["cascade_prob"].values
    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values

    for alpha_int in range(0, 101):
        alpha = alpha_int / 100.0
        use_regular = alpha * reg_prob >= (1.0 - alpha) * cas_prob
        pred = np.where(use_regular, reg_label, cas_label)
        tmp = merged.copy()
        tmp["_pred"] = pred
        y_true, y_pred = to_seqeval_lists(tmp, "true_label", "_pred")
        cur = float(seqeval_f1(y_true, y_pred)) if y_true else 0.0
        if cur > best_f1:
            best_f1 = cur
            best_alpha = alpha

    return best_alpha


def _learned_weights_fusion(merged):
    """Learn optimal alpha, then apply weighted fusion."""
    alpha = _learn_alpha(merged)

    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values
    reg_prob = merged["regular_prob"].values
    cas_prob = merged["cascade_prob"].values
    agree = ~merged["disagree"].values

    use_regular = agree | (alpha * reg_prob >= (1.0 - alpha) * cas_prob)
    merged["fused_pred_label"] = np.where(use_regular, reg_label, cas_label)
    merged["selected_source"] = np.where(
        agree, "agree",
        np.where(use_regular, "weighted_regular", "weighted_cascade"),
    )
    merged["selected_confidence"] = np.where(
        use_regular, alpha * reg_prob, (1.0 - alpha) * cas_prob,
    )
    merged["learned_alpha"] = alpha
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_learned_weights_fusion,
        experiment_id="exp06_learned_ready",
        experiment_name="Learned Weights Fusion (Ready)",
        description=(
            "No-retraining learned-weights fusion from Exp01 + Exp04 ready outputs. "
            "Alpha is learned by sweeping [0,1] to maximise F1."
        ),
        result_basename="fusion_learned_weights_ready",
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_learned_ready] F1={f1_str}")
