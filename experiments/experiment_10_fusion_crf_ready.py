"""
experiment_10_fusion_crf_ready.py — Confidence fusion of CRF regular + CRF cascade
==================================================================================

Uses the same rule as ``experiment_06_fusion_ready.py``:

* If regular-CRF and cascade-CRF **agree** on a token → keep that label.
* If they **disagree** → pick the prediction with higher confidence
  (``regular_prob`` vs ``cascade_prob`` from the merged token table).

No GPU training. Requires cached Excel outputs from ``10_regular`` and ``10_cascade``
(see ``fusion_crf_ready_sources.run_ready_fusion_crf``).
"""

from __future__ import annotations

import numpy as np

from fusion_crf_ready_sources import run_ready_fusion_crf


def _confidence_fusion(merged):
    regular_label = merged["regular_pred_label"].values
    cascade_label = merged["cascade_pred_label"].values
    regular_prob = merged["regular_prob"].values
    cascade_prob = merged["cascade_prob"].values
    agree = ~merged["disagree"].values
    use_regular = agree | (regular_prob >= cascade_prob)
    merged["fused_pred_label"] = np.where(use_regular, regular_label, cascade_label)
    merged["selected_source"] = np.where(
        agree,
        "agree",
        np.where(use_regular, "regular_crf", "cascade_crf"),
    )
    merged["selected_confidence"] = np.where(use_regular, regular_prob, cascade_prob)
    return merged


def run() -> dict:
    return run_ready_fusion_crf(
        strategy_fn=_confidence_fusion,
        experiment_id="exp10_fusion_ready",
        experiment_name="Fusion Regular-CRF + Cascaded-CRF (Ready)",
        description=(
            "Confidence-comparison fusion of Experiment 10 regular BERT-CRF and "
            "cascaded CRF pipeline outputs (no retraining)."
        ),
        result_basename="fusion_crf_ready",
    )


if __name__ == "__main__":
    payload = run()
    f1 = payload.get("f1")
    print(f"[exp10_fusion_ready] F1={f1:.4f}" if f1 is not None else "[exp10_fusion_ready] F1=N/A")
