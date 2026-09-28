#!/usr/bin/env python3
"""Build ``data/ner_dataset.pkl`` from ``data/ner_dataset.xlsx`` (canonical Hebrew text)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
XLSX = DATA_DIR / "ner_dataset.xlsx"
PKL = DATA_DIR / "ner_dataset.pkl"


def main() -> int:
    if not XLSX.exists():
        print(f"Missing source workbook: {XLSX}", file=sys.stderr)
        return 1

    df = pd.read_excel(XLSX)
    PKL.parent.mkdir(parents=True, exist_ok=True)
    with PKL.open("wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {PKL} ({len(df)} rows, {PKL.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
