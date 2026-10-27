"""MLP router on CRF base models — OOF nested CV fusion (primary protocol, CRF track)."""
from __future__ import annotations

from fusion_router_oof_common import exp10_oof_run_factory

run = exp10_oof_run_factory("mlp")

if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp10_mlp_oof] F1={f1_str}")
