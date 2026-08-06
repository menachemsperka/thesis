"""
Cross-benchmark comparison runner (Exp10 on public NER corpora).

Mirrors ``run_cross_data_model_comparison.py`` for:
* Split conditions: simple random vs paper-style multilabel stratified (exp07 variants only)
* Regimes: 300-sentence pool (small) vs full official train (full)
* Experiments: ``10_regular``, ``10_cascade``, ``10_svm_ready``
* Outputs: ``cross_comparison_<ts>.xlsx/json``, checkpoint resume, consolidated Exp10 error analysis

Colab: set ``os.environ["THESIS_RUN_ENV"] = "colab"`` before running (see README).
"""

from __future__ import annotations

import argparse
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Network / Colab env (proxy bypass) — must run before HF downloads
import experiments.common  # noqa: F401, E402

import run_cross_data_model_comparison as cross  # noqa: E402

from configs import (  # noqa: E402
    BENCHMARKS,
    DEFAULT_BASE_SEED,
    DEFAULT_NUM_SEEDS,
    DEFAULT_SEED_START,
    EXPERIMENT_IDS,
    REGIMES,
    SPLIT_VARIANT_RANDOM,
    BenchmarkConfig,
)
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from exp07_split_artifacts import THESIS_LABELS  # noqa: E402
from splits import build_conditions, prepare_all_splits  # noqa: E402

COMPARISON_DIR = BENCHMARK_ROOT / "outputs" / "cross_comparison"
BASE_CRF_INDEX_PATH = COMPARISON_DIR / "cross_comparison_base_crf_ready_index.json"

EXP_NAMES = {
    "10_regular": "Regular NER (BERT-CRF)",
    "10_cascade": "Cascaded Pipeline (CRF + Consistency)",
    "10_svm_ready": "SVM Router Fusion CRF (Ready)",
}


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


def _run_key(benchmark_key: str, exp_id: str, condition_key: str) -> str:
    return f"{benchmark_key}||exp{exp_id}||{condition_key}"


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
) -> None:
    payload = {
        "name": "benchmark_cross_comparison_checkpoint",
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(),
        "run_counter": run_counter,
        "total_runs": total_runs,
        "benchmarks": [b.key for b in benchmarks],
        "experiments": experiment_ids,
        "rows": rows,
    }
    _atomic_write_json(path, payload)


def _build_deltas_paper_vs_random(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (bench, regime, exp_id, seed), grp in results_df.groupby(
        ["benchmark_key", "regime", "experiment_id", "seed"], dropna=False
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
        {"Section": "Design", "Key": "Regimes", "Value": "small_300 (300-sentence pool); full (official train)"},
        {"Section": "Design", "Key": "Split ratio", "Value": "70% train / 30% eval (exp07 split functions)"},
        {"Section": "Design", "Key": "Seeds", "Value": f"{seeds[0]}..{seeds[-1]} ({len(seeds)} paired seeds)"},
        {"Section": "Design", "Key": "Experiments", "Value": ", ".join(experiment_ids)},
        {"Section": "Interpretation", "Key": "delta_f1_paper_minus_random",
         "Value": "Positive => paper-style stratified split improved F1 vs simple random (same seed)"},
    ]
    doc_df = pd.DataFrame(doc_rows)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not pivot_df.empty:
            pivot_df.to_excel(writer, sheet_name="summary_pivot", index=False)
        results_df.to_excel(writer, sheet_name="all_runs", index=False)
        if not deltas_df.empty:
            deltas_df.to_excel(writer, sheet_name="deltas_split_variants", index=False)
        if not paired_df.empty:
            paired_df.to_excel(writer, sheet_name="paired_summary", index=False)
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
    cache_dir: Path,
    pool_seed: int,
    base_mode: str,
    resume: bool,
    checkpoint_file: Path | None,
    prepare_only: bool,
    dry_run: bool,
) -> None:
    os.chdir(PROJECT_ROOT)
    cross.COMPARISON_DIR = COMPARISON_DIR  # noqa: SLF001 — consolidate error analysis output dir

    benchmarks = _resolve_benchmarks(benchmark_keys)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for cfg in benchmarks:
        data_root = BENCHMARK_ROOT / "data" / cfg.key
        meta_path = data_root / "split_meta.json"
        if prepare_only or ((not dry_run) and not meta_path.exists()):
            _log(f"Preparing splits for {cfg.display_name}...")
            prepare_all_splits(
                benchmark_key=cfg.key,
                dataset_key=cfg.dataset_key,
                data_root=data_root,
                cache_dir=cache_dir,
                seeds=seeds,
                pool_seed=pool_seed,
                regimes=regimes,
            )
        if prepare_only:
            _log(f"Prepared {data_root}")
            continue

    if prepare_only:
        return

    all_conditions: list[dict[str, Any]] = []
    for cfg in benchmarks:
        data_root = BENCHMARK_ROOT / "data" / cfg.key
        if (data_root / "split_meta.json").exists():
            all_conditions.extend(
                build_conditions(
                    cfg_key=cfg.key,
                    cfg_display=cfg.display_name,
                    data_root=data_root,
                    regimes=regimes,
                )
            )

    if dry_run:
        if all_conditions:
            n_runs = len(all_conditions) * len(experiment_ids)
            print(
                f"Planned: {len(benchmarks)} benchmarks × {len(all_conditions)} conditions "
                f"× {len(experiment_ids)} experiments = {n_runs} runs"
            )
            for c in all_conditions[:15]:
                print(f"  {c['key']}")
            if len(all_conditions) > 15:
                print(f"  ... +{len(all_conditions) - 15} more")
        else:
            est = len(benchmarks) * len(regimes) * 2 * len(seeds)
            print(
                f"Dry-run (splits not materialized): ~{est} conditions/benchmark after --prepare-only "
                f"(2 split variants × {len(seeds)} seeds × {len(regimes)} regimes)"
            )
        return

    total_runs = len(all_conditions) * len(experiment_ids)
    checkpoint_path = checkpoint_file or (COMPARISON_DIR / "benchmark_cross_comparison_checkpoint.json")
    rows: list[dict] = []
    completed: set[str] = set()
    started_at = datetime.now().isoformat()
    run_counter = 0

    if resume and checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = [r for r in cp.get("rows", []) if isinstance(r, dict)]
        for r in rows:
            if str(r.get("status", "")).startswith("error"):
                continue
            bk = str(r.get("benchmark_key", ""))
            exp = str(r.get("experiment_id", "")).replace("exp", "")
            ck = str(r.get("condition_key", ""))
            if bk and exp and ck:
                completed.add(_run_key(bk, exp, ck))
        run_counter = len(completed)
        started_at = str(cp.get("started_at") or started_at)
        _log(f"Resume: {len(completed)} completed runs loaded from {checkpoint_path}")

    base_crf_mem: dict[str, dict[str, Any]] = {}
    base_crf_index = cross._load_base_index(BASE_CRF_INDEX_PATH)

    for cfg in benchmarks:
        _set_benchmark_model_env(cfg)
        conditions = build_conditions(
            cfg_key=cfg.key,
            cfg_display=cfg.display_name,
            data_root=BENCHMARK_ROOT / "data" / cfg.key,
            regimes=regimes,
        )

        for exp_id in experiment_ids:
            exp_name = EXP_NAMES.get(exp_id, exp_id)
            for cond in conditions:
                rk = _run_key(cfg.key, exp_id, cond["key"])
                if rk in completed:
                    continue

                run_counter += 1
                t0 = time.time()
                _log(
                    f"Run {run_counter}/{total_runs} | {cfg.display_name} | exp{exp_id} | {cond['short_label']}"
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
                    run_counter,
                    total_runs,
                )

    results_df = pd.DataFrame(rows)
    deltas_df = _build_deltas_paper_vs_random(results_df)
    paired_df = _paired_summary(deltas_df)
    ts = _now_ts()

    exp10_error_path = None
    if any(str(e).startswith("10") for e in experiment_ids):
        cross.COMPARISON_DIR = COMPARISON_DIR
        exp10_error_path = cross._consolidate_exp10_error_analysis(rows, ts)

    xlsx_path = _export_workbook(
        results_df=results_df,
        deltas_df=deltas_df,
        paired_df=paired_df,
        ts=ts,
        exp10_error_path=exp10_error_path,
        benchmarks=benchmarks,
        experiment_ids=experiment_ids,
        seeds=seeds,
    )

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
    p.add_argument("--experiments", default=",".join(EXPERIMENT_IDS))
    p.add_argument("--regimes", default=",".join(REGIMES))
    p.add_argument("--seeds", default="")
    p.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    p.add_argument("--pool-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--cache-dir", type=Path, default=BENCHMARK_ROOT / "hf_cache")
    p.add_argument("--base-mode", choices=["auto", "reuse", "retrain"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-file", type=Path, default=None)
    p.add_argument("--prepare-only", action="store_true")
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

    run_comparison(
        benchmark_keys=benchmark_keys,
        experiment_ids=experiment_ids,
        regimes=regimes,
        seeds=seeds,
        cache_dir=args.cache_dir,
        pool_seed=args.pool_seed,
        base_mode=args.base_mode,
        resume=args.resume,
        checkpoint_file=args.checkpoint_file,
        prepare_only=args.prepare_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
