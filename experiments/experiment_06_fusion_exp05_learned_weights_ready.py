from __future__ import annotations

from fusion_exp05_ready_common import run_ready_fusion


def run() -> dict:
    return run_ready_fusion(
        mode="learned_weights",
        experiment_id="exp06_fusion_exp05_learned_ready",
        experiment_name="Fusion of Regular NER and Exp05 (Learned Weights, Ready Results)",
        description=(
            "No-retraining fusion built from ready artifacts: regular side from Exp06 detailed outputs "
            "and cascaded side from Exp05 (Step3 consistency). Learns global regular-vs-exp05 weights "
            "to maximize F1 on the available aligned evaluation tokens."
        ),
        result_basename="fusion_exp05_learned_ready",
    )


if __name__ == "__main__":
    payload = run()
    if payload.get("f1") is None:
        print("[exp06_fusion_exp05_learned_ready] F1=N/A")
    else:
        print(f"[exp06_fusion_exp05_learned_ready] F1={payload['f1']:.4f}")
