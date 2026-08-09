"""
Cross-benchmark comparison runner (public NER corpora).

Per seed (default design):
* Baseline — ``experiment_01_regular_ner``: non-augmented train, simple random split.
* Treatment — ``10_svm_ready``: LLM-augmented train, paper-style multilabel stratified split
  (trains Exp10 CRF bases on that condition, then SVM router fusion).

Also supports custom ``--experiments`` with the full split × train-mode grid from ``build_conditions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]

# Benchmark-local modules (configs, splits, corpus_loaders) must precede repo root on sys.path.
for _p in (PROJECT_ROOT, BENCHMARK_ROOT):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

# Public NER benchmarks (CoNLL, NEMO, BC5CDR) are not Hebrew corpora.
os.environ.setdefault("THESIS_SKIP_HEBREW_TEXT_VALIDATION", "1")

# Network / Colab env (proxy bypass) — must run before HF downloads
import experiments.common  # noqa: F401, E402

import run_cross_data_model_comparison as cross  # noqa: E402

from configs import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_BASELINE_EXPERIMENT,
    BENCHMARK_TREATMENT_EXPERIMENT,
    DEFAULT_BASE_SEED,
    DEFAULT_NUM_SEEDS,
    DEFAULT_SEED_START,
    DEFAULT_TRAIN_MODES,
    EXPERIMENT_IDS,
    REGIMES,
    SPLIT_VARIANT_PAPER,
    SPLIT_VARIANT_RANDOM,
    TRAIN_MODE_AUGMENTED,
    TRAIN_MODE_BASELINE,
    TRAIN_MODES,
    BenchmarkConfig,
)
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from exp07_split_artifacts import THESIS_LABELS  # noqa: E402
from augmentation import augmentation_covers_meta, prepare_augmented_train_splits  # noqa: E402
from splits import build_conditions, load_split_meta, prepare_all_splits  # noqa: E402
from split_stats import build_dataset_details_df  # noqa: E402

COMPARISON_DIR = BENCHMARK_ROOT / "outputs" / "cross_comparison"
BASE_CRF_INDEX_PATH = COMPARISON_DIR / "cross_comparison_base_crf_ready_index.json"
CHECKPOINT_SCHEMA_VERSION = 2

EXP_NAMES = {
    "01": "Regular NER (Exp01 baseline)",
    "10_regular": "Regular NER (BERT-CRF)",
    "10_cascade": "Cascaded Pipeline (CRF + Consistency)",
    "10_svm_ready": "SVM Router Fusion CRF (Ready)",
}

DEFAULT_EXPERIMENT_PROFILE = (
    BENCHMARK_BASELINE_EXPERIMENT,
    BENCHMARK_TREATMENT_EXPERIMENT,
)


def _condition_matches_experiment(cond: dict[str, Any], exp_id: str) -> bool:
    """Map each experiment to its intended split variant and train mode."""
    exp_id = str(exp_id).strip()
    variant = str(cond.get("variant", ""))
    train_mode = str(cond.get("train_mode", TRAIN_MODE_BASELINE))
    if exp_id == BENCHMARK_BASELINE_EXPERIMENT:
        return variant == SPLIT_VARIANT_RANDOM and train_mode == TRAIN_MODE_BASELINE
    if exp_id == BENCHMARK_TREATMENT_EXPERIMENT:
        return variant == SPLIT_VARIANT_PAPER and train_mode == TRAIN_MODE_AUGMENTED
    return True


def _expand_run_plan(
    conditions: list[dict[str, Any]], experiment_ids: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    plan: list[tuple[str, dict[str, Any]]] = []
    for exp_id in experiment_ids:
        for cond in conditions:
            if _condition_matches_experiment(cond, exp_id):
                plan.append((exp_id, cond))
    return plan


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(msg: str) -> None:
    cross._log(msg)


def _resolve_benchmarks(keys: list[str] | None) -> list[BenchmarkConfig]:
    if not keys:
        return list(BENCHMARKS)
    out = [b for b in BENCHMARKS if b.key in keys]
    if not out:
        raise ValueError(f"No benchmarks matched: {keys}")
    return out


def _set_benchmark_model_env(cfg: BenchmarkConfig) -> None:
    cross._set_model_env(cfg.model_id)
    os.environ["THESIS_SKIP_HEBREW_TEXT_VALIDATION"] = "1"
    os.environ["THESIS_NER_CSV"] = str(BENCHMARK_ROOT / "data" / cfg.key / "corpus.csv")


def _meta_covers_prepare(meta: dict[str, Any], seeds: list[int], regimes: list[str]) -> bool:
    """True if split_meta already matches the requested seeds and regimes."""
    if meta.get("small_pool_sampling") != "per_seed":
        return False
    prepared = {int(s) for s in (meta.get("seeds") or [])}
    if prepared != set(seeds):
        return False
    meta_regimes = set((meta.get("regimes") or {}).keys())
    return set(regimes).issubset(meta_regimes)


def _run_key(benchmark_key: str, exp_id: str, condition_key: str) -> str:
    return f"{benchmark_key}||exp{exp_id}||{condition_key}"


def _row_run_key(row: dict[str, Any]) -> str | None:
    bk = str(row.get("benchmark_key", "")).strip()
    exp = str(row.get("experiment_id", "")).strip().replace("exp", "")
    ck = str(row.get("condition_key", "")).strip()
    if bk and exp and ck:
        return _run_key(bk, exp, ck)
    return None


def _run_plan_keys(run_plan: list[tuple[BenchmarkConfig, str, dict[str, Any]]]) -> set[str]:
    return {_run_key(cfg.key, exp_id, cond["key"]) for cfg, exp_id, cond in run_plan}


def _run_plan_fingerprint(
    *,
    run_plan: list[tuple[BenchmarkConfig, str, dict[str, Any]]],
    experiment_ids: list[str],
    regimes: list[str],
    seeds: list[int],
    train_modes: list[str],
) -> str:
    payload = {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "experiments": sorted(experiment_ids),
        "regimes": sorted(regimes),
        "seeds": sorted(int(s) for s in seeds),
        "train_modes": sorted(train_modes),
        "run_keys": sorted(_run_plan_keys(run_plan)),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _filter_rows_to_plan(rows: list[dict[str, Any]], plan_keys: set[str]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rk = _row_run_key(row)
        if rk and rk in plan_keys:
            kept.append(row)
    return kept


def _completed_keys_from_rows(rows: list[dict[str, Any]], plan_keys: set[str]) -> set[str]:
    done: set[str] = set()
    for row in rows:
        if str(row.get("status", "")).startswith("error"):
            continue
        rk = _row_run_key(row)
        if rk and rk in plan_keys:
            done.add(rk)
    return done


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    cross._atomic_write_json(path, payload)


def _save_checkpoint(
    path: Path,
    rows: list[dict],
    benchmarks: list[BenchmarkConfig],
    experiment_ids: list[str],
    started_at: str,
    run_counter: int,
    total_runs: int,
    *,
    run_plan_fingerprint: str,
    regimes: list[str],
    seeds: list[int],
    train_modes: list[str],
) -> None:
    payload = {
        "name": "benchmark_cross_comparison_checkpoint",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_plan_fingerprint": run_plan_fingerprint,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(),
        "run_counter": run_counter,
        "total_runs": total_runs,
        "benchmarks": [b.key for b in benchmarks],
        "experiments": experiment_ids,
        "regimes": regimes,
        "seeds": seeds,
        "train_modes": train_modes,
        "rows": rows,
    }
    _atomic_write_json(path, payload)


def _build_treatment_vs_baseline_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per seed: treatment (10_svm_ready + aug + paper) minus baseline (exp01 + random)."""
    if results_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (bench, regime, seed), grp in results_df.groupby(
        ["benchmark_key", "regime", "seed"], dropna=False
    ):
        base = grp[
            (grp["experiment_id"] == f"exp{BENCHMARK_BASELINE_EXPERIMENT}")
            & (grp["variant"] == SPLIT_VARIANT_RANDOM)
            & (grp["train_mode"] == TRAIN_MODE_BASELINE)
        ]
        treat = grp[
            (grp["experiment_id"] == f"exp{BENCHMARK_TREATMENT_EXPERIMENT}")
            & (grp["variant"] == SPLIT_VARIANT_PAPER)
            & (grp["train_mode"] == TRAIN_MODE_AUGMENTED)
        ]
        if base.empty or treat.empty:
            continue
        b_f1 = pd.to_numeric(base["f1"], errors="coerce").iloc[0]
        t_f1 = pd.to_numeric(treat["f1"], errors="coerce").iloc[0]
        if pd.isna(b_f1) or pd.isna(t_f1):
            continue
        rows.append(
            {
                "benchmark_key": bench,
                "regime": regime,
                "seed": seed,
                "f1_exp01_baseline": float(b_f1),
                "f1_svm_fusion_paper_aug": float(t_f1),
                "delta_f1_treatment_minus_baseline": float(t_f1 - b_f1),
            }
        )
    return pd.DataFrame(rows)


def _paired_summary_treatment(deltas_df: pd.DataFrame) -> pd.DataFrame:
    if deltas_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in deltas_df.groupby(["benchmark_key", "regime"]):
        bench, regime = keys
        vals = pd.to_numeric(grp["delta_f1_treatment_minus_baseline"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "benchmark_key": bench,
                "regime": regime,
                "n_seeds": len(vals),
                "delta_mean": float(vals.mean()),
                "delta_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_deltas_paper_vs_random(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    if "train_mode" not in results_df.columns:
        results_df = results_df.copy()
        results_df["train_mode"] = TRAIN_MODE_BASELINE
    rows: list[dict[str, Any]] = []
    for (bench, regime, exp_id, seed, train_mode), grp in results_df.groupby(
        ["benchmark_key", "regime", "experiment_id", "seed", "train_mode"], dropna=False
    ):
        base = grp[grp["variant"] == SPLIT_VARIANT_RANDOM]
        paper = grp[grp["variant"] == "after_multilabel_iterative_paper"]
        if base.empty or paper.empty:
            continue
        b_f1 = pd.to_numeric(base["f1"], errors="coerce").iloc[0]
        p_f1 = pd.to_numeric(paper["f1"], errors="coerce").iloc[0]
        if pd.isna(b_f1) or pd.isna(p_f1):
            continue
        rows.append(
            {
                "benchmark_key": bench,
                "regime": regime,
                "experiment_id": exp_id,
                "seed": seed,
                "train_mode": train_mode,
                "f1_random": float(b_f1),
                "f1_paper_stratified": float(p_f1),
                "delta_f1_paper_minus_random": float(p_f1 - b_f1),
            }
        )
    return pd.DataFrame(rows)


def _paired_summary(deltas_df: pd.DataFrame) -> pd.DataFrame:
    if deltas_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in deltas_df.groupby(["benchmark_key", "regime", "experiment_id"]):
        bench, regime, exp_id = keys
        vals = pd.to_numeric(grp["delta_f1_paper_minus_random"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "benchmark_key": bench,
                "regime": regime,
                "experiment_id": exp_id,
                "n_seeds": len(vals),
                "delta_mean": float(vals.mean()),
                "delta_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _export_workbook(
    *,
    results_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    treatment_deltas_df: pd.DataFrame,
    treatment_paired_df: pd.DataFrame,
    dataset_details_df: pd.DataFrame,
    ts: str,
    exp10_error_path: Path | None,
    benchmarks: list[BenchmarkConfig],
    experiment_ids: list[str],
    seeds: list[int],
) -> Path:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = COMPARISON_DIR / f"cross_comparison_{ts}.xlsx"

    pivot_rows = []
    if not results_df.empty:
        for keys, grp in results_df.groupby(
            ["benchmark_label", "experiment_id", "regime", "variant_label"], dropna=False
        ):
            f1s = pd.to_numeric(grp["f1"], errors="coerce").dropna()
            pivot_rows.append(
                {
                    "benchmark": keys[0],
                    "experiment_id": keys[1],
                    "regime": keys[2],
                    "split_variant": keys[3],
                    "f1_mean": float(f1s.mean()) if not f1s.empty else None,
                    "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else None,
                    "n_seeds": len(f1s),
                }
            )
    pivot_df = pd.DataFrame(pivot_rows)

    doc_rows = [
        {"Section": "Design", "Key": "Benchmarks", "Value": ", ".join(b.display_name for b in benchmarks)},
        {"Section": "Design", "Key": "Split variants", "Value": "Simple random; Multilabel stratified (paper-style)"},
        {"Section": "Design", "Key": "Regimes", "Value": "small_300 (per seed: sample 300 from official train, then 70% train / 30% eval); full (all official train, same 70/30 per seed)"},
        {"Section": "Design", "Key": "Dataset details sheet", "Value": "Per seed: train/eval sentences, tokens, entity spans, tokens per entity type (JSON columns)"},
        {"Section": "Design", "Key": "Train modes", "Value": "baseline (original train split); augmented (exp08 LLM mask-fill on train only, same eval)"},
        {"Section": "Design", "Key": "Augmentation", "Value": "THESIS_EXP08_MULTIPLIER (default 3); fill-mask model = benchmark encoder unless THESIS_AUGMENTATION_MODEL_NAME set"},
        {"Section": "Design", "Key": "Seeds", "Value": f"{seeds[0]}..{seeds[-1]} ({len(seeds)} paired seeds)"},
        {"Section": "Design", "Key": "Default comparison", "Value": (
            f"Baseline: exp{BENCHMARK_BASELINE_EXPERIMENT} (random split, no augmentation); "
            f"Treatment: exp{BENCHMARK_TREATMENT_EXPERIMENT} (paper stratified split, augmented train)"
        )},
        {"Section": "Design", "Key": "Experiments", "Value": ", ".join(experiment_ids)},
        {"Section": "Interpretation", "Key": "delta_f1_treatment_minus_baseline",
         "Value": "Positive => SVM fusion (paper + aug) beat Exp01 baseline (random, no aug) on the same seed/regime"},
        {"Section": "Interpretation", "Key": "delta_f1_paper_minus_random",
         "Value": "Only when both split variants are run for the same experiment (legacy grid mode)"},
    ]
    doc_df = pd.DataFrame(doc_rows)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not dataset_details_df.empty:
            dataset_details_df.to_excel(writer, sheet_name="dataset_details", index=False)
        if not pivot_df.empty:
            pivot_df.to_excel(writer, sheet_name="summary_pivot", index=False)
        results_df.to_excel(writer, sheet_name="all_runs", index=False)
        if not deltas_df.empty:
            deltas_df.to_excel(writer, sheet_name="deltas_split_variants", index=False)
        if not paired_df.empty:
            paired_df.to_excel(writer, sheet_name="paired_summary", index=False)
        if not treatment_deltas_df.empty:
            treatment_deltas_df.to_excel(writer, sheet_name="deltas_treatment_vs_baseline", index=False)
        if not treatment_paired_df.empty:
            treatment_paired_df.to_excel(writer, sheet_name="paired_treatment_summary", index=False)
        doc_df.to_excel(writer, sheet_name="documentation", index=False)
        if exp10_error_path and exp10_error_path.exists():
            pd.DataFrame(
                [{"item": "consolidated_error_analysis_exp10", "path": str(exp10_error_path)}]
            ).to_excel(writer, sheet_name="exp10_error_analysis", index=False)

    latest = COMPARISON_DIR / "cross_comparison_latest.xlsx"
    if latest.exists():
        latest.unlink()
    shutil.copy2(xlsx_path, latest)
    return xlsx_path


def run_comparison(
    *,
    benchmark_keys: list[str] | None,
    experiment_ids: list[str],
    regimes: list[str],
    seeds: list[int],
    train_modes: list[str],
    cache_dir: Path,
    pool_seed: int,
    base_mode: str,
    resume: bool,
    checkpoint_file: Path | None,
    prepare_only: bool,
    prepare_augmentation_only: bool,
    dry_run: bool,
    force_augmentation: bool = False,
    fresh: bool = False,
) -> None:
    os.chdir(PROJECT_ROOT)
    cross.COMPARISON_DIR = COMPARISON_DIR  # noqa: SLF001 — consolidate error analysis output dir

    benchmarks = _resolve_benchmarks(benchmark_keys)
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_aug = TRAIN_MODE_AUGMENTED in train_modes

    for cfg in benchmarks:
        data_root = BENCHMARK_ROOT / "data" / cfg.key
        meta_path = data_root / "split_meta.json"
        meta: dict[str, Any] | None = None
        if meta_path.exists():
            try:
                meta = load_split_meta(data_root)
            except Exception:
                meta = None

        need_baseline_prepare = not prepare_augmentation_only and not (
            meta and _meta_covers_prepare(meta, seeds, regimes)
        )

        if need_baseline_prepare:
            if meta and not _meta_covers_prepare(meta, seeds, regimes):
                _log(
                    f"Re-preparing {cfg.display_name}: split_meta has seeds "
                    f"{sorted(int(s) for s in (meta.get('seeds') or []))}, requested {seeds}"
                )
            if (not dry_run) or prepare_only or prepare_augmentation_only:
                _log(f"Preparing baseline splits for {cfg.display_name}...")
                prepare_all_splits(
                    benchmark_key=cfg.key,
                    dataset_key=cfg.dataset_key,
                    data_root=data_root,
                    cache_dir=cache_dir,
                    seeds=seeds,
                    pool_seed=pool_seed,
                    regimes=regimes,
                )
                meta = load_split_meta(data_root)
        elif prepare_only and not prepare_augmentation_only and meta:
            _log(f"Baseline splits skipped (already on Drive): {cfg.display_name} → {data_root}")

        if use_aug and not dry_run:
            meta = load_split_meta(data_root) if meta_path.exists() else None
            if meta is None:
                raise FileNotFoundError(
                    f"Missing baseline splits for {cfg.key}. Run --prepare-only first."
                )
            if force_augmentation or not augmentation_covers_meta(meta, data_root):
                _log(f"Preparing LLM augmentation (exp08) for {cfg.display_name}...")
                cross._set_model_env(cfg.model_id)
                prepare_augmented_train_splits(
                    data_root=data_root,
                    ner_model_id=cfg.model_id,
                    benchmark_display=cfg.display_name,
                    force=force_augmentation,
                    log_fn=_log,
                )
            elif prepare_only or prepare_augmentation_only:
                _log(f"Augmentation skipped (already on Drive): {cfg.display_name}")

        if prepare_only or prepare_augmentation_only:
            _log(f"Prepared {data_root}")
            continue

    if prepare_only or prepare_augmentation_only:
        return

    all_conditions: list[dict[str, Any]] = []
    for cfg in benchmarks:
        data_root = BENCHMARK_ROOT / "data" / cfg.key
        meta_path = data_root / "split_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path}. Run --prepare-only for {cfg.key} first."
            )
        meta = load_split_meta(data_root)
        prepared_seeds = set(int(s) for s in (meta.get("seeds") or []))
        requested_seeds = set(seeds)
        if not requested_seeds.issubset(prepared_seeds):
            raise ValueError(
                f"{cfg.key}: prepared seeds {sorted(prepared_seeds)} do not include all "
                f"requested {sorted(requested_seeds)}. Re-run --prepare-only with --num-seeds "
                f"{len(seeds)} (or matching --seeds)."
            )
        if use_aug and not augmentation_covers_meta(meta, data_root):
            raise ValueError(
                f"{cfg.key}: augmented train splits missing. Run --prepare-only (or "
                f"--prepare-augmentation-only) with --train-modes baseline,augmented."
            )
        all_conditions.extend(
            build_conditions(
                cfg_key=cfg.key,
                cfg_display=cfg.display_name,
                data_root=data_root,
                regimes=regimes,
                train_modes=train_modes,
            )
        )

    n_aug_cond = sum(1 for c in all_conditions if c.get("train_mode") == TRAIN_MODE_AUGMENTED)
    if TRAIN_MODE_AUGMENTED in train_modes and n_aug_cond == 0:
        raise ValueError(
            "No augmented conditions in the run plan (0 *_augmented_train.json files found). "
            "Run --prepare-augmentation-only for your benchmarks, then --resume."
        )

    if BENCHMARK_TREATMENT_EXPERIMENT in experiment_ids and not use_aug:
        raise ValueError(
            f"Experiment {BENCHMARK_TREATMENT_EXPERIMENT} requires augmented train splits. "
            "Use --train-modes baseline,augmented (default) and do not pass --skip-augmentation."
        )

    run_plan: list[tuple[BenchmarkConfig, str, dict[str, Any]]] = []
    for cfg in benchmarks:
        conds = [c for c in all_conditions if c.get("benchmark_key") == cfg.key]
        for exp_id, cond in _expand_run_plan(conds, experiment_ids):
            run_plan.append((cfg, exp_id, cond))

    if BENCHMARK_TREATMENT_EXPERIMENT in experiment_ids:
        n_treat = sum(1 for _, e, _ in run_plan if e == BENCHMARK_TREATMENT_EXPERIMENT)
        if n_treat == 0:
            raise ValueError(
                "No treatment runs in plan (paper stratified + augmented). "
                "Run --prepare-augmentation-only, then retry."
            )

    if dry_run:
        n_runs = len(run_plan)
        print(f"Planned: {len(benchmarks)} benchmarks × {n_runs} matched (experiment, condition) pairs")
        for _cfg, exp_id, cond in run_plan[:20]:
            print(f"  exp{exp_id} | {cond['key']}")
        if len(run_plan) > 20:
            print(f"  ... +{len(run_plan) - 20} more")
        return

    total_runs = len(run_plan)
    n_base_runs = sum(1 for _, e, c in run_plan if e == BENCHMARK_BASELINE_EXPERIMENT)
    n_treat_runs = sum(1 for _, e, c in run_plan if e == BENCHMARK_TREATMENT_EXPERIMENT)
    _log(
        f"Run plan: {total_runs} runs "
        f"(exp{BENCHMARK_BASELINE_EXPERIMENT}: {n_base_runs}, "
        f"exp{BENCHMARK_TREATMENT_EXPERIMENT}: {n_treat_runs})"
    )
    plan_keys = _run_plan_keys(run_plan)
    plan_fingerprint = _run_plan_fingerprint(
        run_plan=run_plan,
        experiment_ids=experiment_ids,
        regimes=regimes,
        seeds=seeds,
        train_modes=train_modes,
    )
    checkpoint_path = checkpoint_file or (COMPARISON_DIR / "benchmark_cross_comparison_checkpoint.json")
    rows: list[dict] = []
    completed: set[str] = set()
    started_at = datetime.now().isoformat()

    if fresh and checkpoint_path.exists():
        backup = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(checkpoint_path, backup)
        _log(f"--fresh: checkpoint backed up to {backup} (starting empty for this run plan)")
    elif fresh:
        _log("--fresh: starting with no checkpoint resume")

    if resume and not fresh and checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cp_fp = str(cp.get("run_plan_fingerprint") or "")
        cp_rows = [r for r in cp.get("rows", []) if isinstance(r, dict)]
        if cp_fp and cp_fp != plan_fingerprint:
            backup = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_stale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            shutil.copy2(checkpoint_path, backup)
            _log(
                f"Checkpoint run plan differs from current CLI (experiments/regimes/seeds/conditions). "
                f"Not reusing {len(cp_rows)} prior rows. Backup: {backup}"
            )
            rows = []
            completed = set()
        else:
            if not cp_fp:
                _log(
                    f"Checkpoint has no run_plan_fingerprint (older runner). "
                    f"Keeping only rows that match the current plan ({total_runs} runs)."
                )
            rows = _filter_rows_to_plan(cp_rows, plan_keys)
            dropped = len(cp_rows) - len(rows)
            if dropped:
                _log(f"Resume: dropped {dropped} checkpoint rows outside the current run plan")
            completed = _completed_keys_from_rows(rows, plan_keys)
            started_at = str(cp.get("started_at") or started_at)
            _log(
                f"Resume: {len(completed)}/{total_runs} runs complete for this plan "
                f"({checkpoint_path})"
            )
    elif resume and not fresh:
        _log(f"Resume requested but checkpoint not found: {checkpoint_path}. Starting fresh.")

    progress_done = len(completed)

    base_crf_mem: dict[str, dict[str, Any]] = {}
    base_crf_index = cross._load_base_index(BASE_CRF_INDEX_PATH)

    for cfg, exp_id, cond in run_plan:
        _set_benchmark_model_env(cfg)
        exp_name = EXP_NAMES.get(exp_id, exp_id)
        rk = _run_key(cfg.key, exp_id, cond["key"])
        if rk in completed:
            continue

        progress_done += 1
        t0 = time.time()
        _log(
            f"Run {progress_done}/{total_runs} | {cfg.display_name} | exp{exp_id} | {cond['short_label']}"
        )

        os.environ["THESIS_NER_CSV"] = str(cond["corpus_csv"])
        os.environ["THESIS_SPLIT_SEED"] = str(cond["seed"])
        os.environ["THESIS_CURRENT_CONDITION_KEY"] = cond["key"]
        os.environ["THESIS_CURRENT_EXP_ID"] = f"exp{exp_id}"

        payload: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        reused = False
        try:
            if exp_id in cross.EXP10_READY_DEPENDENT_EXP_IDS:
                base_entry, reused = cross._ensure_base_artifacts_crf(
                    model_id=cfg.model_id,
                    model_display=cfg.display_name,
                    condition=cond,
                    base_mode=base_mode,
                    base_mem=base_crf_mem,
                    base_index=base_crf_index,
                    base_index_path=BASE_CRF_INDEX_PATH,
                )
                cross._set_ready_env_crf(
                    base_entry["exp10_regular_metrics_file"],
                    base_entry["exp10_cascade_metrics_file"],
                )
                try:
                    payload = cross._import_experiment(exp_id).run()
                    metrics = cross._extract_metrics(payload)
                finally:
                    cross._clear_ready_env_crf()
            elif exp_id == BENCHMARK_BASELINE_EXPERIMENT:
                cross._set_presplit_env(cond["train_path"], cond["eval_path"])
                try:
                    payload = cross._import_experiment(BENCHMARK_BASELINE_EXPERIMENT).run()
                    metrics = cross._extract_metrics(payload)
                finally:
                    cross._clear_presplit_env()
            elif exp_id in {"10_regular", "10_cascade"}:
                base_entry, reused = cross._ensure_base_artifacts_crf(
                    model_id=cfg.model_id,
                    model_display=cfg.display_name,
                    condition=cond,
                    base_mode=base_mode,
                    base_mem=base_crf_mem,
                    base_index=base_crf_index,
                    base_index_path=BASE_CRF_INDEX_PATH,
                )
                result_key = "exp10_regular" if exp_id == "10_regular" else "exp10_cascade"
                payload = cross._load_result_payload(base_entry[f"{result_key}_result_file"])
                metrics = cross._extract_metrics(payload)
                if reused:
                    metrics["status"] = "ok_reused_base"
            else:
                raise ValueError(f"Unsupported experiment id: {exp_id}")
        except Exception as exc:
            traceback.print_exc()
            metrics = {"f1": None, "precision": None, "recall": None, "status": f"error: {exc}"}
        finally:
            os.environ.pop("THESIS_SPLIT_SEED", None)
            os.environ.pop("THESIS_CURRENT_CONDITION_KEY", None)
            os.environ.pop("THESIS_CURRENT_EXP_ID", None)
            try:
                from core.model_cleanup import cleanup_training_artifacts_if_enabled

                cleanup_training_artifacts_if_enabled()
            except Exception:
                pass

        elapsed = time.time() - t0
        _log(f"  F1={cross._fmt(metrics.get('f1'))} ({elapsed:.1f}s)")

        variant_label = THESIS_LABELS.get(cond["variant"], cond["variant"])
        rows.append(
            {
                "benchmark_key": cfg.key,
                "benchmark_label": cfg.display_name,
                "model_id": cfg.model_id,
                "model_name": cfg.display_name,
                "experiment_id": f"exp{exp_id}",
                "experiment_name": exp_name,
                "data_source": cond["regime"],
                "regime": cond["regime"],
                "variant": cond["variant"],
                "variant_label": variant_label,
                "condition_key": cond["key"],
                "condition_group_key": cond.get("base_condition_key", cond["key"]),
                "condition_group_short": cond.get("base_condition_short", cond["short_label"]),
                "condition_label": cond["label"],
                "condition_short": cond["short_label"],
                "condition_description": cond["description"],
                "seed": cond["seed"],
                "train_mode": cond.get("train_mode", TRAIN_MODE_BASELINE),
                "is_baseline": cond["is_baseline"],
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "status": metrics.get("status"),
                "result_file": payload.get("result_file", ""),
                "metrics_file": payload.get("metrics_file", ""),
                "base_artifacts_reused": reused,
                "base_mode": base_mode,
                "elapsed_seconds": round(elapsed, 1),
            }
        )
        completed.add(rk)
        _save_checkpoint(
            checkpoint_path,
            rows,
            benchmarks,
            experiment_ids,
            started_at,
            len(completed),
            total_runs,
            run_plan_fingerprint=plan_fingerprint,
            regimes=regimes,
            seeds=seeds,
            train_modes=train_modes,
        )

    rows = _filter_rows_to_plan(rows, plan_keys)
    results_df = pd.DataFrame(rows)
    deltas_df = _build_deltas_paper_vs_random(results_df)
    paired_df = _paired_summary(deltas_df)
    treatment_deltas_df = _build_treatment_vs_baseline_deltas(results_df)
    treatment_paired_df = _paired_summary_treatment(treatment_deltas_df)
    ts = _now_ts()

    dataset_details_df = build_dataset_details_df(
        benchmark_configs=benchmarks,
        data_root_fn=lambda key: BENCHMARK_ROOT / "data" / key,
    )

    exp10_error_path = None
    if BENCHMARK_TREATMENT_EXPERIMENT in experiment_ids:
        cross.COMPARISON_DIR = COMPARISON_DIR
        exp10_rows = [
            r
            for r in rows
            if str(r.get("experiment_id", "")).strip() in {"exp10_svm_ready", "exp10_fusion_ready"}
            or str(r.get("experiment_id", "")).strip().startswith("exp10_")
        ]
        if exp10_rows:
            exp10_error_path = cross._consolidate_exp10_error_analysis(exp10_rows, ts)

    xlsx_path = _export_workbook(
        results_df=results_df,
        deltas_df=deltas_df,
        paired_df=paired_df,
        treatment_deltas_df=treatment_deltas_df,
        treatment_paired_df=treatment_paired_df,
        dataset_details_df=dataset_details_df,
        ts=ts,
        exp10_error_path=exp10_error_path,
        benchmarks=benchmarks,
        experiment_ids=experiment_ids,
        seeds=seeds,
    )

    if not dataset_details_df.empty:
        details_json = COMPARISON_DIR / f"dataset_details_{ts}.json"
        details_payload = dataset_details_df.to_dict(orient="records")
        details_json.write_text(json.dumps(details_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        latest_details = COMPARISON_DIR / "dataset_details_latest.json"
        if latest_details.exists():
            latest_details.unlink()
        shutil.copy2(details_json, latest_details)

    json_path = COMPARISON_DIR / f"cross_comparison_{ts}.json"
    json_path.write_text(json.dumps({"rows": rows, "exported_at": ts}, indent=2, default=str), encoding="utf-8")
    latest_json = COMPARISON_DIR / "cross_comparison_latest.json"
    if latest_json.exists():
        latest_json.unlink()
    shutil.copy2(json_path, latest_json)

    print(f"\nResults: {xlsx_path}")
    if exp10_error_path:
        print(f"Exp10 error analysis: {exp10_error_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-benchmark Exp10 comparison (public NER corpora).")
    p.add_argument("--benchmarks", default="", help="Comma-separated benchmark keys (default: all).")
    p.add_argument(
        "--experiments",
        default=",".join(EXPERIMENT_IDS),
        help=(
            f"Default: {BENCHMARK_BASELINE_EXPERIMENT} (Exp01 baseline) + "
            f"{BENCHMARK_TREATMENT_EXPERIMENT} (SVM fusion, paper split + aug)."
        ),
    )
    p.add_argument("--regimes", default=",".join(REGIMES))
    p.add_argument("--seeds", default="")
    p.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    p.add_argument("--pool-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--cache-dir", type=Path, default=BENCHMARK_ROOT / "hf_cache")
    p.add_argument("--base-mode", choices=["auto", "reuse", "retrain"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint progress for this run plan (backs up existing checkpoint if present).",
    )
    p.add_argument("--checkpoint-file", type=Path, default=None)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--prepare-augmentation-only",
        action="store_true",
        help="Only run exp08 LLM augmentation on existing baseline splits (GPU recommended).",
    )
    p.add_argument(
        "--force-augmentation",
        action="store_true",
        help="Rebuild all augmented_train JSON files even if they exist.",
    )
    p.add_argument(
        "--train-modes",
        default=",".join(DEFAULT_TRAIN_MODES),
        help="Comma-separated: baseline, augmented (default: both).",
    )
    p.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Train/evaluate baseline splits only (same as --train-modes baseline).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--error-analysis-scope",
        choices=["exp10", "split", "both"],
        default="exp10",
        help="Consolidated error analysis export (same semantics as main cross-comparison).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_keys = [x.strip() for x in args.benchmarks.split(",") if x.strip()] or None
    experiment_ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    if args.skip_augmentation:
        train_modes = [TRAIN_MODE_BASELINE]
    else:
        train_modes = [x.strip() for x in args.train_modes.split(",") if x.strip()]
    bad = [m for m in train_modes if m not in TRAIN_MODES]
    if bad:
        raise ValueError(f"Unknown train modes: {bad}. Use: {list(TRAIN_MODES)}")

    run_comparison(
        benchmark_keys=benchmark_keys,
        experiment_ids=experiment_ids,
        regimes=regimes,
        seeds=seeds,
        train_modes=train_modes,
        cache_dir=args.cache_dir,
        pool_seed=args.pool_seed,
        base_mode=args.base_mode,
        resume=args.resume,
        checkpoint_file=args.checkpoint_file,
        prepare_only=args.prepare_only,
        prepare_augmentation_only=args.prepare_augmentation_only,
        dry_run=args.dry_run,
        force_augmentation=args.force_augmentation,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
