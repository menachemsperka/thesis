"""Linear SVM router — OOF nested CV fusion (primary protocol)."""
from __future__ import annotations

from fusion_router_oof_common import exp06_oof_run_factory

run = exp06_oof_run_factory("linear_svm")

if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_svm_oof] F1={f1_str}")
