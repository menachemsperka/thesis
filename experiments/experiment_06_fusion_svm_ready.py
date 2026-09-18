"""
experiment_06_fusion_svm_ready.py — Linear SVM Router Fusion (Ready)

Alias for the linear-SVM router (``LinearSVC``). See ``fusion_router_ready_common.py``
and ``theisis overview.md`` §12.3.
"""
from __future__ import annotations

from fusion_router_ready_common import exp06_run_factory

run = exp06_run_factory("linear_svm")


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_svm_ready] F1={f1_str}")
