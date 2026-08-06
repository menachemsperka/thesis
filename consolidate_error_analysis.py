"""
consolidate_error_analysis.py — Low-RAM cross-run error-analysis consolidation.

Reads many per-run metrics workbooks (one seed × condition × model × experiment),
aggregates across seeds, and writes a single summary workbook. Does **not** stack
full ``detailed_results`` or other heavy sheets.

Skipped sheets (not copied verbatim):
    detailed_results, documentation, confusion_matrix,
    regular_from_exp01, cascade_from_source

Seed-aggregated sheets (mean / std + n_seeds):
    per_type_metrics, error_type_summary, confidence_analysis,
    disagreement_analysis, entity_length_analysis

error_examples: reservoir sample (default 100,000 rows max).

detailed_results: replaced by thesis-style summary tables (overall + per model),
split by CRF vs non-CRF experiment families.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
_EXPERIMENTS = PROJECT_ROOT / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from error_analysis import classify_error  # noqa: E402

SKIP_SHEETS = frozenset({
    "detailed_results",
    "documentation",
    "confusion_matrix",
    "regular_from_exp01",
    "cascade_from_source",
})

SEED_AGG_SHEETS = (
    "per_type_metrics",
    "error_type_summary",
    "confidence_analysis",
    "disagreement_analysis",
    "entity_length_analysis",
)

DEFAULT_MAX_ERROR_EXAMPLES = 100_000

ERROR_ROW_ORDER = (
    ("false_positive", "FP (False Positive)"),
    ("false_negative", "FN (False Negative)"),
    ("type_error", "Type Error"),
    ("boundary_error", "Boundary Error"),
)

METHOD_SPECS = (
    ("Regular NER", ("regular_pred_label", "pred_label", "predicted_label")),
    ("Cascade NER", ("cascade_pred_label",)),
    ("SVM Fused", ("fused_pred_label",)),
)


def _split_bio(label: str) -> tuple[str, str]:
    label = str(label)
    if "-" in label:
        prefix, etype = label.split("-", 1)
        return prefix, etype
    return label, ""


def _is_exp10_experiment_id(experiment_id: str) -> bool:
    e = str(experiment_id).strip()
    return e.startswith("exp10_") or e == "exp10"


def crf_family(experiment_id: str) -> str:
    return "CRF" if _is_exp10_experiment_id(experiment_id) else "NON-CRF"


def _parse_split_and_aug(condition_short: str, data_source: str) -> tuple[str, str]:
    short = str(condition_short or "").strip()
    aug = "Yes" if str(data_source or "").strip().lower() == "exp07+aug" or "+ Aug" in short else "No"
    strategy = short.replace(" + Aug", "").strip() or short or "unknown"
    return strategy, aug


def _summarise_series(values: pd.Series) -> tuple[float | None, float | None]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None, None
    mean_v = float(vals.mean())
    std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean_v, std_v


def _aggregate_across_seeds(
    frames: list[pd.DataFrame],
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""
    rows: list[dict[str, Any]] = []
    for keys, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        if "seed" in gdf.columns:
            row["n_seeds"] = int(gdf["seed"].nunique())
        elif "source_file" in gdf.columns:
            row["n_seeds"] = int(gdf["source_file"].nunique())
        else:
            row["n_seeds"] = int(len(gdf))
        for metric in metric_cols:
            if metric not in gdf.columns:
                continue
            mean_v, std_v = _summarise_series(gdf[metric])
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_std"] = std_v
        rows.append(row)
    return pd.DataFrame(rows)


class _Reservoir:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self.items: list[dict[str, Any]] = []
        self._seen = 0

    def add_frame(self, df: pd.DataFrame) -> None:
        if self.capacity == 0 or df is None or df.empty:
            return
        for rec in df.to_dict(orient="records"):
            self._seen += 1
            if len(self.items) < self.capacity:
                self.items.append(rec)
            else:
                j = random.randint(0, self._seen - 1)
                if j < self.capacity:
                    self.items[j] = rec


@dataclass
class _DetailAccumulator:
    """Counts derived from detailed_results (streaming, no row storage)."""

    # (family, model_scope, method) -> error_type -> count
    error_by_method: Counter = field(default_factory=Counter)
    # (family, model_scope, method) -> "TRUE → PRED" -> count
    type_confusion: Counter = field(default_factory=Counter)
    tokens_by_slice: Counter = field(default_factory=Counter)
    # SVM routing: (family, model_scope, route) -> (correct, error)
    svm_route: Counter = field(default_factory=Counter)
    svm_route_errors: Counter = field(default_factory=Counter)
    # split table: (family, split, aug, method) -> error_type -> count
    split_method_errors: Counter = field(default_factory=Counter)
    split_tokens: Counter = field(default_factory=Counter)


def _pick_pred_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _cascade_label_from_row(row: pd.Series) -> str | None:
    if "cascade_pred_label" in row.index and pd.notna(row.get("cascade_pred_label")):
        return str(row["cascade_pred_label"])
    if "pred_bio" in row.index and "pred_etype" in row.index:
        bio = str(row.get("pred_bio", "O"))
        etype = row.get("pred_etype")
        if bio == "O":
            return "O"
        if etype is None or (isinstance(etype, float) and np.isnan(etype)):
            return "O"
        return f"{bio}-{etype}"
    return None


def _true_label_series(df: pd.DataFrame) -> pd.Series:
    if "true_label" in df.columns:
        return df["true_label"].astype(str)
    if "true_bio" in df.columns and "true_etype" in df.columns:
        return df.apply(
            lambda r: "O"
            if str(r.get("true_bio", "O")) == "O"
            else f"{r['true_bio']}-{r['true_etype']}",
            axis=1,
        )
    return pd.Series(["O"] * len(df), index=df.index)


def _normalize_svm_route(disagree: bool, selected_source: str) -> str | None:
    src = str(selected_source or "").strip().lower()
    if not disagree or src == "agree":
        return "Agree (no routing needed)"
    if "regular" in src or src.startswith("svm_regular") or src == "fallback_regular":
        return "SVM → Regular"
    if "cascade" in src or "exp05" in src or "exp10" in src or "svm_cascade" in src:
        return "SVM → Cascade"
    return None


def _accumulate_detailed(
    acc: _DetailAccumulator,
    df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    if df.empty:
        return

    family = crf_family(str(meta.get("experiment_id", "")))
    model = str(meta.get("model_name") or meta.get("model_id") or "unknown")
    split_strategy, aug = _parse_split_and_aug(
        str(meta.get("condition_group_short") or meta.get("condition_short") or ""),
        str(meta.get("data_source") or ""),
    )
    true_labels = _true_label_series(df)
    n_tokens = int(len(df))

    for model_scope in ("__overall__", model):
        acc.tokens_by_slice[(family, model_scope, "__all__")] += n_tokens
        acc.split_tokens[(family, split_strategy, aug, "__all__")] += n_tokens

    exp_id = str(meta.get("experiment_id", ""))
    is_svm = "svm" in exp_id.lower()

    for method_name, col_candidates in METHOD_SPECS:
        if method_name == "SVM Fused" and "fused_pred_label" not in df.columns:
            continue
        preds: list[str] = []
        if method_name == "Cascade NER":
            for _, row in df.iterrows():
                lab = _cascade_label_from_row(row)
                preds.append(lab if lab is not None else "O")
        else:
            col = _pick_pred_column(df, col_candidates)
            if col is None:
                continue
            preds = df[col].astype(str).tolist()

        for model_scope in ("__overall__", model):
            for true_l, pred_l in zip(true_labels, preds):
                err = classify_error(true_l, pred_l)
                key = (family, model_scope, method_name)
                acc.error_by_method[(key, err)] += 1
                if err == "type_error":
                    _, te = _split_bio(true_l)
                    _, pe = _split_bio(pred_l)
                    conf = f"{te or 'O'} → {pe or 'O'}"
                    acc.type_confusion[(key, conf)] += 1

        for true_l, pred_l in zip(true_labels, preds):
            err = classify_error(true_l, pred_l)
            acc.split_method_errors[(family, split_strategy, aug, method_name, err)] += 1

    if is_svm and "fused_pred_label" in df.columns and "selected_source" in df.columns:
        disagree = df["disagree"].astype(bool) if "disagree" in df.columns else pd.Series(False, index=df.index)
        fused = df["fused_pred_label"].astype(str)
        for model_scope in ("__overall__", model):
            for i in range(len(df)):
                route = _normalize_svm_route(bool(disagree.iloc[i]), str(df["selected_source"].iloc[i]))
                if route is None:
                    continue
                err = classify_error(str(true_labels.iloc[i]), str(fused.iloc[i]))
                correct = err == "correct"
                acc.svm_route[(family, model_scope, route, "correct" if correct else "error")] += 1
                acc.svm_route_errors[(family, model_scope, route, err)] += 1


def _error_type_table(acc: _DetailAccumulator, family: str, model_scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for method_name, _ in METHOD_SPECS:
        col_sum = 0
        for err_key, label in ERROR_ROW_ORDER:
            n = acc.error_by_method.get(((family, model_scope, method_name), err_key), 0)
            counts[(method_name, err_key)] = n
            col_sum += n
        counts[(method_name, "__total__")] = col_sum

    totals = {m: counts.get((m, "__total__"), 0) for m, _ in METHOD_SPECS}
    for err_key, label in ERROR_ROW_ORDER:
        row: dict[str, Any] = {"Error Type": label}
        for method_name, _ in METHOD_SPECS:
            n = counts.get((method_name, err_key), 0)
            row[method_name] = n
            tot = totals.get(method_name) or 0
            row[f"{method_name} %"] = (100.0 * n / tot) if tot else None
        rows.append(row)

    total_row: dict[str, Any] = {"Error Type": "TOTAL ERRORS"}
    for method_name, _ in METHOD_SPECS:
        tot = totals.get(method_name) or 0
        total_row[method_name] = tot
        total_row[f"{method_name} %"] = 100.0 if tot else None
    rows.append(total_row)
    return pd.DataFrame(rows)


def _type_confusion_table(acc: _DetailAccumulator, family: str, model_scope: str, top_n: int = 14) -> pd.DataFrame:
    """Top entity confusions with Regular / Cascade / SVM Fused counts."""
    pooled: Counter = Counter()
    for (slice_key, conf), n in acc.type_confusion.items():
        fam, scope, _meth = slice_key
        if fam == family and scope == model_scope:
            pooled[conf] += n

    if not pooled:
        return pd.DataFrame()

    top_labels = [c for c, _ in pooled.most_common(top_n)]
    rows: list[dict[str, Any]] = []
    for conf_label in top_labels + (["Other"] if sum(pooled.values()) > sum(pooled[l] for l in top_labels) else []):
        if conf_label == "Other":
            other_labels = set(pooled) - set(top_labels)
            if not other_labels:
                continue
        row: dict[str, Any] = {"Entity Confusion": conf_label}
        for method_name, _ in METHOD_SPECS:
            n = sum(
                cnt
                for (slice_key, c), cnt in acc.type_confusion.items()
                if slice_key == (family, model_scope, method_name)
                and (c == conf_label if conf_label != "Other" else c not in top_labels)
            )
            row[method_name] = n
        rows.append(row)

    df = pd.DataFrame(rows)
    for method_name, _ in METHOD_SPECS:
        if method_name not in df.columns:
            continue
        tot = int(df[method_name].sum())
        df[f"{method_name} %"] = df[method_name].apply(lambda x: (100.0 * x / tot) if tot else None)
    total_row: dict[str, Any] = {"Entity Confusion": "TOTAL TYPE ERRORS"}
    for method_name, _ in METHOD_SPECS:
        if method_name in df.columns:
            total_row[method_name] = int(df[method_name].sum())
            total_row[f"{method_name} %"] = 100.0 if total_row[method_name] else None
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def _svm_router_table(acc: _DetailAccumulator, family: str, model_scope: str, title: str) -> pd.DataFrame:
    routes = [
        "Agree (no routing needed)",
        "SVM → Regular",
        "SVM → Cascade",
    ]
    rows: list[dict[str, Any]] = []
    total_n = 0
    total_correct = 0
    total_error = 0
    for route in routes:
        correct = acc.svm_route.get((family, model_scope, route, "correct"), 0)
        error = acc.svm_route.get((family, model_scope, route, "error"), 0)
        count = correct + error
        if count == 0 and route != "Agree (no routing needed)":
            continue
        total_n += count
        total_correct += correct
        total_error += error
        rows.append({
            "Routing Decision": route,
            "Count": count,
            "%": None,
            "Correct": correct,
            "Error": error,
            "Accuracy": (correct / count) if count else None,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["%"] = df["Count"].apply(lambda x: (100.0 * x / total_n) if total_n else None)
    df.loc[len(df)] = {
        "Routing Decision": "TOTAL",
        "Count": total_n,
        "%": 100.0 if total_n else None,
        "Correct": total_correct,
        "Error": total_error,
        "Accuracy": (total_correct / total_n) if total_n else None,
    }
    df.attrs["section_title"] = title
    return df


def _split_strategy_table(acc: _DetailAccumulator, family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    triples = sorted({
        (k[1], k[2], k[3])
        for k in acc.split_method_errors
        if k[0] == family
    })
    for split_strategy, aug, method in triples:
        fp = acc.split_method_errors.get((family, split_strategy, aug, method, "false_positive"), 0)
        fn = acc.split_method_errors.get((family, split_strategy, aug, method, "false_negative"), 0)
        te = acc.split_method_errors.get((family, split_strategy, aug, method, "type_error"), 0)
        be = acc.split_method_errors.get((family, split_strategy, aug, method, "boundary_error"), 0)
        total = fp + fn + te + be
        tokens = acc.split_tokens.get((family, split_strategy, aug, "__all__"), 0)
        err_rate = (100.0 * total / tokens) if tokens else None
        rows.append({
            "Split Strategy": split_strategy,
            "Aug.": aug,
            "Method": method.replace("SVM Fused", "SVM Fusion"),
            "FP": fp,
            "FN": fn,
            "Type": te,
            "Boundary": be,
            "Total": total,
            "Error Rate": err_rate,
        })
    return pd.DataFrame(rows)


def _write_section_blocks(writer: pd.ExcelWriter, sheet_name: str, sections: list[tuple[str, pd.DataFrame]]) -> None:
    """Write multiple titled sections into one sheet (vertical stack)."""
    start_row = 0
    placeholder = pd.DataFrame({"": ["Consolidated error-analysis summaries"]})
    placeholder.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(placeholder) + 2
    for title, frame in sections:
        if frame is None or frame.empty:
            continue
        hdr = pd.DataFrame({title: [""]})
        hdr.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, header=False)
        start_row += 2
        frame.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        start_row += len(frame) + 3


def _build_summary_sections(acc: _DetailAccumulator, family: str) -> list[tuple[str, pd.DataFrame]]:
    sections: list[tuple[str, pd.DataFrame]] = []
    sections.append((
        f"Error Analysis: Regular vs Cascade NER ({family}) — OVERALL",
        _error_type_table(acc, family, "__overall__"),
    ))
    tc_all = _type_confusion_table(acc, family, "__overall__")
    if not tc_all.empty:
        sections.append((f"Type Error by Entity ({family}) — OVERALL", tc_all))

    model_names: set[str] = set()
    for key in acc.error_by_method:
        fam, scope, _meth = key[0]
        if fam == family and scope != "__overall__":
            model_names.add(scope)
    for model in sorted(model_names):
        sections.append((
            f"Error Analysis ({family}) — {model}",
            _error_type_table(acc, family, model),
        ))
        tc = _type_confusion_table(acc, family, model)
        if not tc.empty:
            sections.append((f"Type Error by Entity ({family}) — {model}", tc))

    svm_overall = _svm_router_table(acc, family, "__overall__", "SVM Router Results")
    if not svm_overall.empty:
        sections.append(("SVM Router Results", svm_overall.drop(columns=[], errors="ignore")))
        dis = _svm_router_table(acc, family, "__overall__", "SVM Routing on Disagreements")
        # Filter to disagreement routes only for second table
        if not dis.empty:
            dis = dis[dis["Routing Decision"].astype(str).str.contains("SVM →", na=False)]
            if not dis.empty:
                sections.append(("SVM Routing on Disagreements", dis))

    split_df = _split_strategy_table(acc, family)
    if not split_df.empty:
        sections.append((f"ERROR ANALYSIS BY SPLIT STRATEGY AND METHOD ({family})", split_df))

    thesis = pd.DataFrame({
        "Thesis Statements": [
            "• Add your thesis observations here...",
            "• Key findings from the error analysis...",
            "• Conclusions about model performance...",
        ]
    })
    sections.append(("Thesis Statements", thesis))
    return sections


def _row_matches_filters(
    row: dict[str, Any],
    experiment_id_prefixes: tuple[str, ...] | None,
    exclude_exp10: bool,
) -> bool:
    status = str(row.get("status", "")).strip().lower()
    if status.startswith("error"):
        return False
    exp_id = str(row.get("experiment_id", "")).strip()
    if exclude_exp10 and _is_exp10_experiment_id(exp_id):
        return False
    if experiment_id_prefixes and not any(exp_id == p for p in experiment_id_prefixes):
        return False
    mf = str(row.get("metrics_file", "")).strip()
    return bool(mf) and Path(mf).exists()


def consolidate_workbooks_from_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    experiment_id_prefixes: tuple[str, ...] | None = None,
    exclude_exp10: bool = False,
    progress_every: int = 25,
    max_error_examples: int = DEFAULT_MAX_ERROR_EXAMPLES,
) -> dict[str, int]:
    """Build consolidated workbook; return scan stats."""
    output_path = Path(output_path)
    jobs: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _row_matches_filters(row, experiment_id_prefixes, exclude_exp10):
            continue
        mf = str(row["metrics_file"]).strip()
        if mf in seen_paths:
            continue
        seen_paths.add(mf)
        meta = {
            "experiment_id": row.get("experiment_id"),
            "model_id": row.get("model_id"),
            "model_name": row.get("model_name"),
            "data_source": row.get("data_source"),
            "condition_group_short": row.get("condition_group_short") or row.get("condition_short"),
            "condition_short": row.get("condition_short"),
            "seed": row.get("seed"),
            "source_file": mf,
        }
        jobs.append((Path(mf), meta))

    if not jobs:
        raise ValueError("No metrics workbooks to consolidate.")

    sheet_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in SEED_AGG_SHEETS}
    reservoir = _Reservoir(max_error_examples)
    detail_acc = _DetailAccumulator()
    n_files = len(jobs)
    print(f"[consolidate] processing {n_files} workbook(s)...", flush=True)

    for idx, (path, meta) in enumerate(jobs):
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or (idx + 1) == n_files):
            print(f"[consolidate] file {idx + 1}/{n_files}: {path.name}", flush=True)
        try:
            xl = pd.ExcelFile(path)
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}", flush=True)
            continue

        for sheet in SEED_AGG_SHEETS:
            if sheet not in xl.sheet_names:
                continue
            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["source_file"] = path.name
            df["consolidated_experiment_id"] = meta.get("experiment_id")
            df["crf_family"] = crf_family(str(meta.get("experiment_id", "")))
            for k, v in meta.items():
                col = f"run_{k}"
                if col not in df.columns:
                    df[col] = v
            sheet_frames[sheet].append(df)

        if "error_examples" in xl.sheet_names:
            try:
                ex = pd.read_excel(path, sheet_name="error_examples")
                if not ex.empty:
                    ex = ex.copy()
                    ex["source_file"] = path.name
                    reservoir.add_frame(ex.head(5000))
            except Exception:
                pass

        if "detailed_results" in xl.sheet_names:
            try:
                usecols = None
                dr = pd.read_excel(path, sheet_name="detailed_results")
                _accumulate_detailed(detail_acc, dr, meta)
            except Exception as exc:
                print(f"[warn] detailed_results summary skipped for {path.name}: {exc}", flush=True)

    stats = {"workbooks": n_files, "error_example_sample": len(reservoir.items)}

    out_docs = pd.DataFrame([
        {"section": "ABOUT", "item": "consolidation_mode", "description": "Seed-aggregated + detailed_results summaries (low RAM)"},
        {"section": "ABOUT", "item": "skipped_sheets", "description": ", ".join(sorted(SKIP_SHEETS))},
        {"section": "ABOUT", "item": "workbooks_processed", "description": str(n_files)},
        {"section": "ABOUT", "item": "max_error_examples", "description": str(max_error_examples)},
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_docs.to_excel(writer, sheet_name="documentation", index=False)

        group_base = ["consolidated_experiment_id", "crf_family", "run_model_name", "run_condition_group_short"]
        if sheet_frames["per_type_metrics"]:
            pt = _aggregate_across_seeds(
                sheet_frames["per_type_metrics"],
                group_base + ["entity_type"],
                ["precision", "recall", "f1", "support"],
            )
            pt.to_excel(writer, sheet_name="per_type_metrics", index=False)

        if sheet_frames["error_type_summary"]:
            et = _aggregate_across_seeds(
                sheet_frames["error_type_summary"],
                group_base + ["error_type"],
                ["count", "pct_of_tokens"],
            )
            et.to_excel(writer, sheet_name="error_type_summary", index=False)

        if sheet_frames["confidence_analysis"]:
            ca = _aggregate_across_seeds(
                sheet_frames["confidence_analysis"],
                group_base + ["confidence_bucket"],
                ["tokens", "correct", "accuracy", "mean_confidence"],
            )
            ca.to_excel(writer, sheet_name="confidence_analysis", index=False)

        if sheet_frames["disagreement_analysis"]:
            da = _aggregate_across_seeds(
                sheet_frames["disagreement_analysis"],
                group_base,
                [
                    "disagreement_tokens", "regular_correct", "cascade_correct",
                    "fused_correct", "oracle_correct", "fused_accuracy",
                    "oracle_accuracy", "router_recovery_rate",
                ],
            )
            da.to_excel(writer, sheet_name="disagreement_analysis", index=False)

        if sheet_frames["entity_length_analysis"]:
            el = _aggregate_across_seeds(
                sheet_frames["entity_length_analysis"],
                group_base + ["entity_length_tokens"],
                ["true_entities", "correctly_detected", "recall"],
            )
            el.to_excel(writer, sheet_name="entity_length_analysis", index=False)

        if reservoir.items:
            pd.DataFrame(reservoir.items).to_excel(writer, sheet_name="error_examples", index=False)

        for family in ("NON-CRF", "CRF"):
            sections = _build_summary_sections(detail_acc, family)
            if not sections:
                continue
            sheet = f"summary_{family.lower().replace('-', '_')}"
            _write_section_blocks(writer, sheet[:31], sections)

    print(f"[done] consolidated -> {output_path}", flush=True)
    return stats
