"""Random forest disagreement router — ready fusion on Exp01 + Exp04."""
from __future__ import annotations

from fusion_router_ready_common import exp06_run_factory

run = exp06_run_factory("rf")

if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_rf_ready] F1={f1_str}")
