#!/usr/bin/env python3
"""Quick check: ner_dataset.csv and exp07 splits contain Hebrew, not ``?`` placeholders."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from common import resolve_dataset  # noqa: E402
from hebrew_text_io import HebrewCorpusEncodingError, read_ner_dataset_csv, validate_hebrew_dataframe  # noqa: E402
from split_io import load_split  # noqa: E402

EXP07_SPLITS = PROJECT_ROOT / "outputs" / "exp07" / "splits"


def main() -> int:
    csv_path = resolve_dataset("ner_dataset.csv")
    print(f"Checking {csv_path} ...")
    df, enc = read_ner_dataset_csv(csv_path)
    validate_hebrew_dataframe(df, context=f"ner_dataset ({enc})")
    print(f"OK: CSV decoded as {enc!r} with Hebrew content.")

    meta_path = EXP07_SPLITS / "split_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for vm in meta.get("variants", []):
            train_file = vm.get("train_file")
            if train_file and (EXP07_SPLITS / train_file).exists():
                load_split(EXP07_SPLITS / train_file)
                print(f"OK: split {train_file} contains Hebrew.")
                break
    else:
        print("No exp07 splits yet (split_meta.json missing); CSV check only.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HebrewCorpusEncodingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
