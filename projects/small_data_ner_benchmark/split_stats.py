"""Per-split dataset statistics for benchmark exports (sentences, tokens, entities)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_sentences(sentences: list[dict]) -> dict[str, Any]:
    """Count sentences, tokens, entity spans (B-), and entity tokens per type (B-+I-)."""
    n_tokens = 0
    spans_by_type: dict[str, int] = defaultdict(int)
    tokens_by_type: dict[str, int] = defaultdict(int)

    for sent in sentences:
        labels = list(sent.get("labels") or [])
        text_tokens = str(sent.get("text", "")).split()
        n_tokens += len(labels) if labels else len(text_tokens)
        for lab in labels:
            lab = str(lab).strip()
            if lab.startswith("B-"):
                etype = lab[2:]
                spans_by_type[etype] += 1
                tokens_by_type[etype] += 1
            elif lab.startswith("I-"):
                etype = lab[2:]
                tokens_by_type[etype] += 1

    spans_by_type = dict(sorted(spans_by_type.items()))
    tokens_by_type = dict(sorted(tokens_by_type.items()))
    return {
        "n_sentences": len(sentences),
        "n_tokens": n_tokens,
        "n_entity_spans": sum(spans_by_type.values()),
        "entity_spans_by_type": spans_by_type,
        "entity_tokens_by_type": tokens_by_type,
    }


def _stats_row(
    *,
    benchmark_key: str,
    benchmark_label: str,
    regime: str,
    variant: str,
    variant_label: str,
    seed: int,
    pool_n_sentences: int,
    split_ratio: float,
    train: dict[str, Any],
    eval_: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_key": benchmark_key,
        "benchmark_label": benchmark_label,
        "regime": regime,
        "variant": variant,
        "variant_label": variant_label,
        "seed": seed,
        "pool_n_sentences": pool_n_sentences,
        "split_ratio_train": split_ratio,
        "train_n_sentences": train["n_sentences"],
        "train_n_tokens": train["n_tokens"],
        "train_n_entity_spans": train["n_entity_spans"],
        "train_entity_spans_by_type": json.dumps(train["entity_spans_by_type"], ensure_ascii=False),
        "train_entity_tokens_by_type": json.dumps(train["entity_tokens_by_type"], ensure_ascii=False),
        "eval_n_sentences": eval_["n_sentences"],
        "eval_n_tokens": eval_["n_tokens"],
        "eval_n_entity_spans": eval_["n_entity_spans"],
        "eval_entity_spans_by_type": json.dumps(eval_["entity_spans_by_type"], ensure_ascii=False),
        "eval_entity_tokens_by_type": json.dumps(eval_["entity_tokens_by_type"], ensure_ascii=False),
    }


def dataset_details_from_meta(
    meta: dict[str, Any],
    *,
    benchmark_label: str = "",
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Flatten ``split_meta.json`` into one row per (regime, variant, seed)."""
    benchmark_key = str(meta.get("benchmark_key", ""))
    label = benchmark_label or benchmark_key
    split_ratio = float(meta.get("split_ratio", 0.7))
    rows: list[dict[str, Any]] = []

    def _load_split(rel: str) -> list[dict]:
        if not data_root or not rel:
            return []
        path = data_root / "splits" / str(rel).replace("\\", "/")
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [{"text": str(x.get("text", "")), "labels": list(x.get("labels", []))} for x in raw]

    for regime, block in (meta.get("regimes") or {}).items():
        for vm in block.get("variants") or []:
            variant = str(vm.get("variant", ""))
            variant_label = str(vm.get("label", variant))
            seed_files = vm.get("seed_files") or {}
            for seed_str, files in seed_files.items():
                stats = files.get("stats") or {}
                train = stats.get("train")
                eval_ = stats.get("eval")
                if not train or not eval_:
                    train_sents = _load_split(str(files.get("train_file", "")))
                    eval_sents = _load_split(str(files.get("eval_file", "")))
                    if train_sents and eval_sents:
                        train = summarize_sentences(train_sents)
                        eval_ = summarize_sentences(eval_sents)
                if not train or not eval_:
                    continue
                rows.append(
                    _stats_row(
                        benchmark_key=benchmark_key,
                        benchmark_label=label,
                        regime=regime,
                        variant=variant,
                        variant_label=variant_label,
                        seed=int(seed_str),
                        pool_n_sentences=int(files.get("pool_n_sentences", meta.get("small_pool_size", 0))),
                        split_ratio=float(files.get("split_ratio", split_ratio)),
                        train=train,
                        eval_=eval_,
                    )
                )
    return rows


def build_dataset_details_df(
    *,
    benchmark_configs: list[Any],
    data_root_fn,
) -> pd.DataFrame:
    """Build dataset detail rows for all benchmarks that have ``split_meta.json``."""
    all_rows: list[dict[str, Any]] = []
    for cfg in benchmark_configs:
        data_root = data_root_fn(cfg.key)
        meta_path = data_root / "split_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        all_rows.extend(
            dataset_details_from_meta(meta, benchmark_label=cfg.display_name, data_root=data_root)
        )
    return pd.DataFrame(all_rows)
