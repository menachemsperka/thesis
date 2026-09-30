"""
DEACTIVATED — ``after_multilabel_stratified`` was removed from the Exp07 pipeline.

Use ``after_multilabel_iterative_paper`` only. Regenerate splits with
``experiments/exp07_split_artifacts.py`` / ``run_cross_data_model_comparison.py``.

Legacy script body retained below (commented) for reference only.
"""
from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "DEACTIVATED: after_multilabel_stratified is no longer generated. "
        "Use after_multilabel_iterative_paper and rerun exp07 / cross-comparison prep."
    )


if __name__ == "__main__":
    main()

# --- Legacy implementation (after_multilabel_stratified exp07+aug patcher) ---
# from pathlib import Path
# import json
# ...
# TARGET_VARIANT = "after_multilabel_stratified"
