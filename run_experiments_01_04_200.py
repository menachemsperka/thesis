from __future__ import annotations

import chardet
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.common import configure_network_environment


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "ner_dataset.csv"
SUBSET_DIR = PROJECT_ROOT / "outputs" / "subsets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def build_sentence_subset(
    source_csv: Path,
    subset_csv: Path,
    num_sentences: int = 200,
    seed: int = 42,
) -> tuple[int, int]:
    if not source_csv.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_csv}")

    with open(source_csv, "rb") as f:
        result = chardet.detect(f.read())
    detected_encoding = result["encoding"]

    df = pd.read_csv(source_csv, encoding=detected_encoding)
    if "id" not in df.columns:
        raise ValueError("Expected an 'id' column in dataset for sentence grouping.")

    sentence_ids = df["id"].dropna().drop_duplicates().astype(str).tolist()
    if not sentence_ids:
        raise ValueError("No sentence ids found in dataset.")

    take = min(num_sentences, len(sentence_ids))
    sampled_ids = (
        pd.Series(sentence_ids)
        .sample(n=take, random_state=seed, replace=False)
        .tolist()
    )

    subset_df = df[df["id"].astype(str).isin(sampled_ids)].copy()
    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(subset_csv, index=False, encoding="utf-8")
    return take, len(subset_df)


def run_experiment(script_name: str, env: dict[str, str]) -> None:
    script_path = EXPERIMENTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    print(f"\nRunning {script_name} ...")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(EXPERIMENTS_DIR),
        check=True,
        env=env,
    )


def read_latest_f1(exp_id: str) -> float | None:
    latest_path = PROJECT_ROOT / "outputs" / exp_id / "latest.json"
    if not latest_path.exists():
        return None
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    value = payload.get("f1")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def main() -> None:
    configure_network_environment()

    subset_size = int((os.environ.get("THESIS_SUBSET_SENTENCES") or "200").strip() or "200")
    subset_seed = int((os.environ.get("THESIS_SUBSET_SEED") or "42").strip() or "42")
    split_seed = int((os.environ.get("THESIS_SPLIT_SEED") or "42").strip() or "42")

    subset_csv = SUBSET_DIR / f"ner_dataset_{subset_size}_seed{subset_seed}.csv"
    used_sentences, used_rows = build_sentence_subset(
        DATASET_PATH,
        subset_csv,
        num_sentences=subset_size,
        seed=subset_seed,
    )

    print(f"Prepared subset: {subset_csv}")
    print(f"Sentence count: {used_sentences} | Token rows: {used_rows}")

    env = {
        **os.environ,
        "THESIS_NER_CSV": str(subset_csv),
        "THESIS_SPLIT_SEED": str(split_seed),
    }

    run_experiment("experiment_01_regular_ner.py", env)
    run_experiment("experiment_04_auc_cascaded_pipeline.py", env)

    exp01_f1 = read_latest_f1("exp01")
    exp04_f1 = read_latest_f1("exp04")
    print("\nDone.")
    print(f"exp01 F1: {exp01_f1 if exp01_f1 is not None else 'N/A'}")
    print(f"exp04 F1: {exp04_f1 if exp04_f1 is not None else 'N/A'}")


if __name__ == "__main__":
    main()
