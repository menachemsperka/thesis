"""Backward-compatible re-export; prefer ``core.model_cleanup``."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.model_cleanup import (  # noqa: E402
    cleanup_training_artifacts,
    cleanup_training_artifacts_if_enabled,
    should_delete_models_after_train,
)
