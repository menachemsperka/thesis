"""
Runtime detection for local vs Google Colab and CPU vs GPU.

Call ``bootstrap_thesis_runtime()`` as early as possible (``experiments/common.py``
does this on import). Safe to call multiple times.
"""
from __future__ import annotations

import os
from typing import Any


def is_google_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_run_env() -> str:
    explicit = (os.environ.get("THESIS_RUN_ENV") or "").strip().lower()
    if explicit:
        return explicit
    if is_google_colab():
        return "colab"
    return "local"


def _apply_fp16_default(info: dict[str, Any]) -> None:
    if (os.environ.get("THESIS_TRAINER_FP16") or "").strip():
        info["THESIS_TRAINER_FP16"] = os.environ["THESIS_TRAINER_FP16"].strip()
        return
    use_fp16 = "1" if torch_cuda_available() else "0"
    os.environ["THESIS_TRAINER_FP16"] = use_fp16
    info["THESIS_TRAINER_FP16"] = use_fp16


def _apply_colab_cpu_batch_defaults(info: dict[str, Any]) -> None:
    """Reduce Exp04 memory on Colab CPU unless the user already set batch env vars."""
    if torch_cuda_available():
        return
    defaults = {
        "THESIS_EXP04_TRAIN_BATCH": "4",
        "THESIS_EXP04_EVAL_BATCH": "4",
        "THESIS_EXP04_GRAD_ACCUM": "4",
    }
    applied: list[str] = []
    for key, value in defaults.items():
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = value
            applied.append(f"{key}={value}")
    if applied:
        info["colab_cpu_exp04_batches"] = ", ".join(applied)


def bootstrap_thesis_runtime() -> dict[str, Any]:
    """
    Detect Colab, set ``THESIS_RUN_ENV``, and apply safe CPU/GPU defaults.

    User overrides always win (env vars already set are not replaced).
    """
    info: dict[str, Any] = {}
    prev = (os.environ.get("THESIS_RUN_ENV") or "").strip()
    run_env = resolve_run_env()
    if not prev and run_env == "colab":
        os.environ["THESIS_RUN_ENV"] = "colab"
        info["auto_detected"] = "colab"
    info["THESIS_RUN_ENV"] = (os.environ.get("THESIS_RUN_ENV") or run_env).strip() or "local"

    if info["THESIS_RUN_ENV"] == "colab":
        os.environ.setdefault("WANDB_DISABLED", "true")
        if not torch_cuda_available():
            _apply_colab_cpu_batch_defaults(info)

    _apply_fp16_default(info)
    info["cuda_available"] = torch_cuda_available()
    return info


def describe_runtime() -> str:
    info = bootstrap_thesis_runtime()
    device = "GPU (CUDA)" if info.get("cuda_available") else "CPU"
    parts = [
        f"runtime={info.get('THESIS_RUN_ENV', 'local')}",
        f"device={device}",
        f"fp16={os.environ.get('THESIS_TRAINER_FP16', '?')}",
    ]
    if info.get("auto_detected"):
        parts.append(f"auto_detected={info['auto_detected']}")
    if info.get("colab_cpu_exp04_batches"):
        parts.append(f"exp04_batches={info['colab_cpu_exp04_batches']}")
    return "Runtime: " + ", ".join(parts)


def trainer_fp16_enabled() -> bool:
    """Whether Hugging Face Trainer should use fp16 (never on CPU unless misconfigured)."""
    raw = (os.environ.get("THESIS_TRAINER_FP16") or "").strip().lower()
    if not raw:
        return torch_cuda_available()
    enabled = raw in {"1", "true", "yes", "on"}
    if enabled and not torch_cuda_available():
        return False
    return enabled
