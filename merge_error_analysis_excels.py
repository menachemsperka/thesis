"""
merge_error_analysis_excels.py — Combine many per-model error-analysis Excel files
into a single workbook (low-RAM consolidated summaries).

See ``consolidate_error_analysis.py`` for merge rules. This CLI scans a folder of
``.xlsx`` files and writes one consolidated output.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from consolidate_error_analysis import consolidate_workbooks_from_rows


_DEFAULT_INPUT_DIR = os.path.join("outputs", "00 final_results", "error analysis")


def merge(input_dir: str, output_path: str, *, progress_every: int = 50) -> None:
    input_path = Path(input_dir)
    output_stem = Path(output_path).stem
    files = sorted(
        p for p in input_path.glob("*.xlsx")
        if not p.name.startswith("~$") and not p.stem.startswith(output_stem)
    )
    if not files:
        raise SystemExit(f"No .xlsx files found in: {input_dir}")

    rows = [
        {"metrics_file": str(p), "status": "ok", "experiment_id": "", "model_name": p.stem}
        for p in files
    ]
    consolidate_workbooks_from_rows(
        rows,
        output_path,
        progress_every=progress_every or 25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-model error-analysis Excel files into one workbook.")
    parser.add_argument("--input-dir", default=_DEFAULT_INPUT_DIR,
                        help=f"Folder containing the .xlsx files (default: {_DEFAULT_INPUT_DIR}).")
    parser.add_argument("--output", default=None,
                        help="Output .xlsx path (default: <input-dir>/combined_error_analysis.xlsx).")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.input_dir, "combined_error_analysis.xlsx")
    merge(args.input_dir, output_path)


if __name__ == "__main__":
    main()
