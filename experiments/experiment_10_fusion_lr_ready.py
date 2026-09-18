"""Logistic regression router on Exp10 CRF ready outputs."""
from __future__ import annotations

from fusion_router_ready_common import exp10_run_factory

run = exp10_run_factory("lr")

if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp10_lr_ready] F1={f1_str}")
