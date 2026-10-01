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

Journal / thesis method focus (eight columns): Regular NER (exp01), Cascade NER (exp04),
Confidence Fusion (exp06_ready), Linear SVM Fusion (exp06_svm_oof primary; exp06_svm_ready
appendix), RBF SVM Fusion (exp06_svm_kernel_oof / exp06_svm_kernel_ready), Naive Bayes Fusion
(exp06_nb_oof / exp06_nb_ready), Logistic Regression Fusion (exp06_lr_oof / exp06_lr_ready),
RF Fusion (exp06_rf_oof / exp06_rf_ready). MLP Fusion (exp06_mlp_oof / exp06_mlp_ready) is also
recognized if that experiment is run. Designed for 1–3 training seeds + 5-fold OOF router CV.
Other experiment IDs are omitted from summary tabs.
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

from error_analysis import classify_error, model_display_name  # noqa: E402

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

# One worksheet per scope; slug is used in Excel tab names (31-char limit).
CANONICAL_MODEL_SHEETS: tuple[tuple[str, str], ...] = (
    ("overall", "__overall__"),
    ("dictabert", "DictaBERT"),
    ("berel", "BEREL 3.0"),
    ("hero", "HeRo"),
    ("alephbertgimmel", "AlephBERT-Gimmel"),
)

ROUTER_ROUTES = (
    "Agree (no routing needed)",
    "Router → Regular",
    "Router → Cascade",
)

# Thesis error-analysis focus: direct NER, cascade, and every fusion/router variant.
FOCUS_THESIS_ERROR_ANALYSIS_EXP_IDS = frozenset({
    "exp01",
    "exp04",
    "exp06_ready",
    "exp06_svm_oof",
    "exp06_svm_ready",
    "exp06_svm_kernel_oof",
    "exp06_svm_kernel_ready",
    "exp06_nb_oof",
    "exp06_nb_ready",
    "exp06_lr_oof",
    "exp06_lr_ready",
    "exp06_rf_oof",
    "exp06_rf_ready",
    "exp06_mlp_oof",
    "exp06_mlp_ready",
    "exp10_regular",
    "exp10_cascade",
    "exp10_fusion_ready",
    "exp10_svm_ready",
    "exp10_svm_kernel_ready",
    "exp10_nb_ready",
    "exp10_lr_ready",
    "exp10_rf_ready",
    "exp10_mlp_ready",
})

# exp01 / exp10_regular write a *sentence*-level "detailed_results" sheet
# (one row per sentence; true/predicted labels are space-joined strings in
# "true_labels" / "predicted_labels"). The real per-token "true_label" /
# "pred_label" columns used for token-level error analysis live in the
# "token_predictions" sheet instead — use that one for these experiment ids.
SENTENCE_LEVEL_DETAILED_RESULTS_EXP_IDS = frozenset({"exp01", "exp10_regular"})


def _in_thesis_error_analysis_focus(experiment_id: str) -> bool:
    return str(experiment_id or "").strip().lower() in FOCUS_THESIS_ERROR_ANALYSIS_EXP_IDS


# Each fusion method name maps to the set of (OOF, ready, CRF-ready) experiment ids
# that should be pooled into that column. Add new rows here as new routers are run.
_FUSION_METHOD_EXP_IDS: dict[str, tuple[str, ...]] = {
    "Confidence Fusion": ("exp06_ready", "exp10_fusion_ready"),
    "Linear SVM Fusion": ("exp06_svm_oof", "exp06_svm_ready", "exp10_svm_ready"),
    "RBF SVM Fusion": ("exp06_svm_kernel_oof", "exp06_svm_kernel_ready", "exp10_svm_kernel_ready"),
    "Naive Bayes Fusion": ("exp06_nb_oof", "exp06_nb_ready", "exp10_nb_ready"),
    "Logistic Regression Fusion": ("exp06_lr_oof", "exp06_lr_ready", "exp10_lr_ready"),
    "RF Fusion": ("exp06_rf_oof", "exp06_rf_ready", "exp10_rf_ready"),
    "MLP Fusion": ("exp06_mlp_oof", "exp06_mlp_ready", "exp10_mlp_ready"),
}

# Router label used in the routing-decision sections (confidence fusion has no
# trained classifier router, so it is intentionally excluded here).
_ROUTER_LABEL_BY_METHOD: dict[str, str] = {
    "Linear SVM Fusion": "Linear SVM",
    "RBF SVM Fusion": "RBF SVM",
    "Naive Bayes Fusion": "Naive Bayes",
    "Logistic Regression Fusion": "Logistic Regression",
    "RF Fusion": "RF",
    "MLP Fusion": "MLP",
}


def _method_applies(experiment_id: str, method_name: str) -> bool:
    """Map each run to at most one comparison column (no duplicate Regular/Cascade from fusion runs)."""
    e = str(experiment_id or "").strip().lower()
    if method_name == "Regular NER":
        return e in ("exp01", "exp10_regular")
    if method_name == "Cascade NER":
        return e in ("exp04", "exp10_cascade")
    return e in _FUSION_METHOD_EXP_IDS.get(method_name, ())


def _router_label_for_experiment(experiment_id: str) -> str | None:
    e = str(experiment_id or "").strip().lower()
    for method_name, router_label in _ROUTER_LABEL_BY_METHOD.items():
        if e in _FUSION_METHOD_EXP_IDS.get(method_name, ()):
            return router_label
    return None

ROUTING_ERROR_TYPES = (
    "correct",
    "type_error",
    "boundary_error",
    "false_positive",
    "false_negative",
)

ERROR_ROW_ORDER = (
    ("false_positive", "FP (False Positive)"),
    ("false_negative", "FN (False Negative)"),
    ("type_error", "Type Error"),
    ("boundary_error", "Boundary Error"),
)

METHOD_SPECS = (
    ("Regular NER", ("regular_pred_label", "pred_label", "predicted_label")),
    ("Cascade NER", ("cascade_pred_label",)),
    ("Confidence Fusion", ("fused_pred_label",)),
    ("Linear SVM Fusion", ("fused_pred_label",)),
    ("RBF SVM Fusion", ("fused_pred_label",)),
    ("Naive Bayes Fusion", ("fused_pred_label",)),
    ("Logistic Regression Fusion", ("fused_pred_label",)),
    ("RF Fusion", ("fused_pred_label",)),
    ("MLP Fusion", ("fused_pred_label",)),
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


def _canonical_model_display(meta: dict[str, Any]) -> str:
    """Map run metadata to one of the canonical model display names when possible."""
    raw = str(meta.get("model_name") or meta.get("model_id") or "").strip()
    low = raw.lower()
    for slug, display in CANONICAL_MODEL_SHEETS:
        if slug == "overall":
            continue
        if slug in low or display.lower() in low:
            return display
    return model_display_name(raw) if raw else "unknown"


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
    # Linear SVM / RF routers: (family, model_scope, router_label, route, outcome) -> count
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


def _true_label_series(df: pd.DataFrame, context: str = "") -> pd.Series:
    if "true_label" in df.columns:
        return df["true_label"].astype(str)
    if "true_bio" in df.columns and "true_etype" in df.columns:
        return df.apply(
            lambda r: "O"
            if str(r.get("true_bio", "O")) == "O"
            else f"{r['true_bio']}-{r['true_etype']}",
            axis=1,
        )
    print(
        f"[warn] _true_label_series: no true_label/true_bio+true_etype columns found "
        f"({context}); columns={list(df.columns)[:12]} — defaulting every row to 'O' "
        f"(this silently zeroes out error counts for this sheet).",
        flush=True,
    )
    return pd.Series(["O"] * len(df), index=df.index)


def _normalize_router_route(disagree: bool, selected_source: str) -> str | None:
    src = str(selected_source or "").strip().lower()
    if not disagree or src == "agree":
        return "Agree (no routing needed)"
    if "regular" in src or src == "fallback_regular":
        return "Router → Regular"
    if "cascade" in src or "exp05" in src or src == "fallback_cascade":
        return "Router → Cascade"
    return None


def _accumulate_detailed(
    acc: _DetailAccumulator,
    df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    if df.empty:
        return

    family = crf_family(str(meta.get("experiment_id", "")))
    model = _canonical_model_display(meta)
    split_strategy, aug = _parse_split_and_aug(
        str(meta.get("condition_group_short") or meta.get("condition_short") or ""),
        str(meta.get("data_source") or ""),
    )
    exp_id = str(meta.get("experiment_id", ""))
    e_low = exp_id.strip().lower()
    true_labels = _true_label_series(
        df, context=f"experiment_id={exp_id} source_file={meta.get('source_file')}"
    )
    n_tokens = int(len(df))
    router_label = _router_label_for_experiment(exp_id)

    # Token denominators: one baseline NER run per (model, split), not repeated per fusion experiment.
    if e_low in ("exp01", "exp10_regular"):
        for model_scope in ("__overall__", model):
            acc.tokens_by_slice[(family, model_scope, "__all__")] += n_tokens
            acc.split_tokens[(family, model_scope, split_strategy, aug, "__all__")] += n_tokens

    for method_name, col_candidates in METHOD_SPECS:
        if not _method_applies(exp_id, method_name):
            continue
        if col_candidates == ("fused_pred_label",) and "fused_pred_label" not in df.columns:
            # Fusion columns are all keyed off a single "fused_pred_label" column;
            # skip quietly instead of warning when a run simply predates that column.
            continue
        preds: list[str] = []
        if method_name == "Cascade NER":
            for _, row in df.iterrows():
                lab = _cascade_label_from_row(row)
                preds.append(lab if lab is not None else "O")
        else:
            col = _pick_pred_column(df, col_candidates)
            if col is None:
                print(
                    f"[warn] _accumulate_detailed: '{method_name}' expected one of "
                    f"{col_candidates} but found none in columns={list(df.columns)[:12]} "
                    f"(experiment_id={exp_id}, source_file={meta.get('source_file')}) — "
                    "skipping this method for this sheet.",
                    flush=True,
                )
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

        for model_scope in ("__overall__", model):
            for true_l, pred_l in zip(true_labels, preds):
                err = classify_error(true_l, pred_l)
                acc.split_method_errors[
                    (family, model_scope, split_strategy, aug, method_name, err)
                ] += 1

    if (
        router_label
        and "fused_pred_label" in df.columns
        and "selected_source" in df.columns
    ):
        disagree = df["disagree"].astype(bool) if "disagree" in df.columns else pd.Series(False, index=df.index)
        fused = df["fused_pred_label"].astype(str)
        for model_scope in ("__overall__", model):
            for i in range(len(df)):
                route = _normalize_router_route(bool(disagree.iloc[i]), str(df["selected_source"].iloc[i]))
                if route is None:
                    continue
                err = classify_error(str(true_labels.iloc[i]), str(fused.iloc[i]))
                correct = err == "correct"
                outcome = "correct" if correct else "error"
                acc.svm_route[(family, model_scope, router_label, route, outcome)] += 1
                acc.svm_route_errors[(family, model_scope, router_label, route, err)] += 1


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
    """Top entity confusions across Regular / Cascade / all fusion-method columns."""
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


def _router_table(
    acc: _DetailAccumulator,
    family: str,
    model_scope: str,
    title: str,
    *,
    router_label: str,
) -> pd.DataFrame:
    routes = list(ROUTER_ROUTES)
    rows: list[dict[str, Any]] = []
    total_n = 0
    total_correct = 0
    total_error = 0
    for route in routes:
        correct = acc.svm_route.get((family, model_scope, router_label, route, "correct"), 0)
        error = acc.svm_route.get((family, model_scope, router_label, route, "error"), 0)
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


def _split_strategy_table(
    acc: _DetailAccumulator,
    family: str,
    model_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    triples = sorted({
        (k[2], k[3], k[4])
        for k in acc.split_method_errors
        if k[0] == family and k[1] == model_scope
    })
    for split_strategy, aug, method in triples:
        fp = acc.split_method_errors.get(
            (family, model_scope, split_strategy, aug, method, "false_positive"), 0
        )
        fn = acc.split_method_errors.get(
            (family, model_scope, split_strategy, aug, method, "false_negative"), 0
        )
        te = acc.split_method_errors.get(
            (family, model_scope, split_strategy, aug, method, "type_error"), 0
        )
        be = acc.split_method_errors.get(
            (family, model_scope, split_strategy, aug, method, "boundary_error"), 0
        )
        total = fp + fn + te + be
        tokens = acc.split_tokens.get((family, model_scope, split_strategy, aug, "__all__"), 0)
        err_rate = (100.0 * total / tokens) if tokens else None
        rows.append({
            "Split Strategy": split_strategy,
            "Aug.": aug,
            "Method": method,
            "FP": fp,
            "FN": fn,
            "Type": te,
            "Boundary": be,
            "Total": total,
            "Error Rate": err_rate,
        })
    return pd.DataFrame(rows)


def _empty_section_note(section: str) -> pd.DataFrame:
    return pd.DataFrame([{"note": f"No data for: {section}"}])


def _error_types_by_routing_table(
    acc: _DetailAccumulator,
    family: str,
    model_scope: str,
    *,
    router_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    route_cols = {
        "Agree (no routing needed)": "Agree",
        "Router → Regular": "Router→Regular",
        "Router → Cascade": "Router→Cascade",
    }
    for err in ROUTING_ERROR_TYPES:
        row: dict[str, Any] = {"Error Type": err}
        row_total = 0
        for route, col in route_cols.items():
            n = acc.svm_route_errors.get((family, model_scope, router_label, route, err), 0)
            row[col] = n
            row_total += n
        row["Total"] = row_total
        rows.append(row)
    if not any(r.get("Total", 0) for r in rows):
        return pd.DataFrame()
    total_row: dict[str, Any] = {"Error Type": "TOTAL"}
    for col in ("Agree", "Router→Regular", "Router→Cascade", "Total"):
        total_row[col] = sum(int(r.get(col, 0) or 0) for r in rows)
    rows.append(total_row)
    return pd.DataFrame(rows)


def _thesis_statements_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Journal / discussion notes (edit in Excel)": [
            "• Main F1: cross_comparison → journal_oof_fold_summary (5-fold OOF) or journal_main_table (1–3 seeds).",
            "• Loss weights: journal_lambda_grid + loss_config_summary; cite validation grid, not test tuning.",
            "• Error patterns: compare Boundary vs Type columns across methods on summary_non_crf_* tabs.",
            "• Routing: Router → Regular/Cascade accuracy on each ML-router OOF run "
            "(Linear SVM / RBF SVM / Naive Bayes / Logistic Regression / RF / MLP); "
            "Confidence Fusion has no trained router, so it has no routing section.",
        ]
    })


def _build_scope_summary_sections(
    acc: _DetailAccumulator,
    family: str,
    model_scope: str,
    scope_label: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Fixed section order — same blocks on every model tab and on overall."""
    sections: list[tuple[str, pd.DataFrame]] = []

    err_tbl = _error_type_table(acc, family, model_scope)
    method_list = " vs ".join(m for m, _ in METHOD_SPECS)
    sections.append((
        f"Error Analysis: {method_list} ({family} — {scope_label})",
        err_tbl if not err_tbl.empty else _empty_section_note("error analysis"),
    ))

    tc = _type_confusion_table(acc, family, model_scope)
    sections.append((
        f"Type Error by Entity ({scope_label})",
        tc if not tc.empty else _empty_section_note("type error by entity"),
    ))

    for router_label in _ROUTER_LABEL_BY_METHOD.values():
        router_tbl = _router_table(
            acc, family, model_scope, f"{router_label} Router Results", router_label=router_label,
        )
        sections.append((
            f"{router_label} Router Results",
            router_tbl if not router_tbl.empty else _empty_section_note(f"{router_label} router"),
        ))

        dis = _router_table(
            acc, family, model_scope, f"{router_label} Routing on Disagreements", router_label=router_label,
        )
        if not dis.empty:
            dis = dis[dis["Routing Decision"].astype(str).str.contains("Router →", na=False)]
        sections.append((
            f"{router_label} Routing on Disagreements",
            dis if not dis.empty else _empty_section_note(f"{router_label} routing on disagreements"),
        ))

        route_err = _error_types_by_routing_table(acc, family, model_scope, router_label=router_label)
        sections.append((
            f"{router_label} — Error Types by Routing Decision",
            route_err if not route_err.empty else _empty_section_note(f"{router_label} routing errors"),
        ))

    split_df = _split_strategy_table(acc, family, model_scope)
    sections.append((
        f"ERROR ANALYSIS BY SPLIT STRATEGY AND METHOD ({scope_label})",
        split_df if not split_df.empty else _empty_section_note("split strategy breakdown"),
    ))

    sections.append(("Thesis Statements", _thesis_statements_frame()))
    return sections


def _write_section_blocks(writer: pd.ExcelWriter, sheet_name: str, sections: list[tuple[str, pd.DataFrame]]) -> None:
    """Write multiple titled sections into one sheet (vertical stack)."""
    start_row = 0
    for title, frame in sections:
        hdr = pd.DataFrame({title: [""]})
        hdr.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, header=False)
        start_row += 2
        out = frame if frame is not None and not frame.empty else _empty_section_note(title)
        out.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        start_row += len(out) + 3


def _write_family_summary_tabs(writer: pd.ExcelWriter, acc: _DetailAccumulator, family: str) -> None:
    """One tab per model (+ overall), identical section layout on each tab."""
    prefix = f"summary_{family.lower().replace('-', '_')}"
    for slug, scope in CANONICAL_MODEL_SHEETS:
        if slug == "overall":
            scope_label = "OVERALL"
        else:
            scope_label = scope
        sheet_name = f"{prefix}_{slug}"[:31]
        sections = _build_scope_summary_sections(acc, family, scope, scope_label)
        _write_section_blocks(writer, sheet_name, sections)


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

        exp_id = str(meta.get("experiment_id", ""))
        e_low = exp_id.strip().lower()
        in_focus = _in_thesis_error_analysis_focus(exp_id)

        if in_focus:
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

            # exp01 / exp10_regular: "detailed_results" is sentence-level (space-joined
            # label strings); the real per-token true_label/pred_label columns are in
            # "token_predictions" instead. Everything else (exp04/exp10_cascade,
            # exp06_*/exp10_* fusion runs) already has a token-level "detailed_results".
            detail_sheet_name = "detailed_results"
            if e_low in SENTENCE_LEVEL_DETAILED_RESULTS_EXP_IDS and "token_predictions" in xl.sheet_names:
                detail_sheet_name = "token_predictions"

            if detail_sheet_name in xl.sheet_names:
                try:
                    dr = pd.read_excel(path, sheet_name=detail_sheet_name)
                    _accumulate_detailed(detail_acc, dr, meta)
                except Exception as exc:
                    print(f"[warn] {detail_sheet_name} summary skipped for {path.name}: {exc}", flush=True)

    stats = {"workbooks": n_files, "error_example_sample": len(reservoir.items)}

    out_docs = pd.DataFrame([
        {"section": "ABOUT", "item": "consolidation_mode", "description": "Seed-aggregated + detailed_results summaries (low RAM)"},
        {"section": "ABOUT", "item": "skipped_sheets", "description": ", ".join(sorted(SKIP_SHEETS))},
        {"section": "ABOUT", "item": "workbooks_processed", "description": str(n_files)},
        {"section": "ABOUT", "item": "max_error_examples", "description": str(max_error_examples)},
        {"section": "ABOUT", "item": "thesis_method_focus",
         "description": (
             "Regular NER (exp01), Cascade NER (exp04), Confidence Fusion (exp06_ready), "
             "Linear SVM Fusion (exp06_svm_oof), RBF SVM Fusion (exp06_svm_kernel_oof), "
             "Naive Bayes Fusion (exp06_nb_oof), Logistic Regression Fusion (exp06_lr_oof), "
             "RF Fusion (exp06_rf_oof), MLP Fusion (exp06_mlp_oof); CRF analogs exp10_* when present"
         )},
        {"section": "ABOUT", "item": "journal_guide",
         "description": "thesis_overview.md Part IV — how to use this workbook + cross_comparison_*.xlsx"},
        {"section": "ABOUT", "item": "focus_experiment_ids",
         "description": ", ".join(sorted(FOCUS_THESIS_ERROR_ANALYSIS_EXP_IDS))},
        {"section": "ABOUT", "item": "summary_tabs",
         "description": (
             "Per family (summary_non_crf_* / summary_crf_*): overall, dictabert, berel, hero, "
             "alephbertgimmel — identical section layout on each tab"
         )},
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
            _write_family_summary_tabs(writer, detail_acc, family)

        try:
            from journal_results_export import (
                collect_loss_config_rows,
                collect_oof_fold_long,
                journal_documentation_rows,
                load_lambda_grid_selection,
                summarize_oof_for_paper,
            )

            rows_df = pd.DataFrame([r for r in rows if isinstance(r, dict)])
            loss_cfg = collect_loss_config_rows(rows_df)
            if not loss_cfg.empty:
                loss_cfg.to_excel(writer, sheet_name="loss_config_summary", index=False)
            oof_long = collect_oof_fold_long(rows_df)
            if not oof_long.empty:
                oof_long.to_excel(writer, sheet_name="oof_fold_metrics_long", index=False)
                oof_sum = summarize_oof_for_paper(oof_long)
                if not oof_sum.empty:
                    oof_sum.to_excel(writer, sheet_name="oof_fold_summary", index=False)
            grid_sel = load_lambda_grid_selection(PROJECT_ROOT)
            if not grid_sel.empty:
                grid_sel.to_excel(writer, sheet_name="lambda_grid_selection", index=False)
            pd.DataFrame(journal_documentation_rows()).to_excel(
                writer, sheet_name="journal_paper_guide", index=False
            )
        except Exception as exc:
            print(f"[warn] journal supplement sheets skipped: {exc}", flush=True)

    print(f"[done] consolidated -> {output_path}", flush=True)
    return stats
