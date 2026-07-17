"""
fusion_ready_sources.py — Shared module for ready-results fusion experiments.

Loads pre-computed outputs from Exp01 (Regular NER) and Exp04 (Cascaded Pipeline),
merges them, and provides a generic entry point for applying any fusion strategy
without retraining.

Requires:
    - Exp01 output with a "token_predictions" sheet (sentence_id, token_idx, token,
      true_label, pred_label, prob, entropy, margin).
    - Exp04 output with a "detailed_results" sheet (sentence_id, token_idx, token,
      true_bio, true_etype, pred_bio, pred_etype, entity_prob, bio_prob).

Environment variables (optional — auto-resolved from latest.json if not set):
    THESIS_READY_EXP01_XLSX  — path to Exp01 result xlsx
    THESIS_READY_EXP04_XLSX  — path to Exp04 result xlsx
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from common import write_result_excel, write_result_json


# ---------------------------------------------------------------------------
# Run-context resolution (model / split condition / seed)
# ---------------------------------------------------------------------------

# Maps a substring of THESIS_MODEL_NAME to a human-friendly display name.
_MODEL_DISPLAY_NAMES: tuple[tuple[str, str], ...] = (
    ("berel", "BEREL 3.0"),
    ("dictabert", "DictaBERT"),
    ("alephbertgimmel", "AlephBERT-Gimmel"),
    ("hero", "HeRo"),
)


def _resolve_model_display() -> str:
    """Map THESIS_MODEL_NAME (a model id or local path) to a friendly name."""
    raw = (os.environ.get("THESIS_MODEL_NAME") or "").strip()
    if not raw:
        return "unknown"
    lowered = raw.lower()
    for needle, display in _MODEL_DISPLAY_NAMES:
        if needle in lowered:
            return display
    # Fall back to the last path/id segment (e.g. "dicta-il/foo" -> "foo").
    return raw.replace("\\", "/").rstrip("/").split("/")[-1] or raw



# ---------------------------------------------------------------------------
# Source file resolution
# ---------------------------------------------------------------------------

def _resolve_source(exp_id: str, env_var: str) -> Path:
    """Find the latest output xlsx for *exp_id*, or use an explicit env var."""
    explicit = (os.environ.get(env_var) or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points to a missing file: {p}")
        return p

    latest_json = Path("outputs") / exp_id / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(
            f"Cannot auto-resolve {exp_id} output.  "
            f"Either set {env_var} or run experiment {exp_id} first so that "
            f"{latest_json} exists."
        )

    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    metrics_file = payload.get("metrics_file")
    if not metrics_file:
        raise ValueError(f"metrics_file key missing in {latest_json}")

    p = Path(metrics_file)
    if not p.exists():
        raise FileNotFoundError(f"metrics_file from {latest_json} not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _load_regular_from_exp01_legacy_token_level(xlsx_path: Path) -> pd.DataFrame:
    """Compatibility path for older Exp01 files that only have token_level sheet.

    Legacy files do not store per-token probabilities. We synthesize neutral
    confidence features so ready fusion can still execute without retraining.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name="token_level")
    except ValueError:
        raise ValueError(
            f"Exp01 output {xlsx_path} does not have a 'token_predictions' sheet "
            "or a compatible legacy 'token_level' sheet."
        )

    required = {"sentence_id", "token_id", "token", "true_label", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Exp01 legacy token_level in {xlsx_path} is missing columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "sentence_id": df["sentence_id"],
            "token_idx": df["token_id"],
            "token": df["token"],
            "true_label": df["true_label"],
            "regular_pred_label": df["predicted_label"],
            # Unknown confidence in legacy sheet -> neutral defaults.
            "regular_prob": 0.5,
            "regular_entropy": 1.0,
            "regular_margin": 0.0,
        }
    )

    out["sentence_id"] = pd.to_numeric(out["sentence_id"], errors="coerce").fillna(-1).astype(int)
    out["token_idx"] = pd.to_numeric(out["token_idx"], errors="coerce").fillna(-1).astype(int)
    out = out[(out["sentence_id"] >= 0) & (out["token_idx"] >= 0)].copy()
    out["token"] = out["token"].astype(str)
    out["true_label"] = out["true_label"].astype(str)
    out["regular_pred_label"] = out["regular_pred_label"].astype(str)
    return out


def load_regular_from_exp01(xlsx_path: Path) -> pd.DataFrame:
    """Load regular token predictions from Exp01, with legacy compatibility."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="token_predictions")
    except ValueError:
        return _load_regular_from_exp01_legacy_token_level(xlsx_path)

    required = {"sentence_id", "token_idx", "token", "true_label", "pred_label", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp01 token_predictions missing columns: {sorted(missing)}")

    out = df.copy()
    out["sentence_id"] = out["sentence_id"].astype(int)
    out["token_idx"] = out["token_idx"].astype(int)
    out["token"] = out["token"].astype(str)
    out["true_label"] = out["true_label"].astype(str)
    # Rename for fusion consistency
    out.rename(columns={
        "pred_label": "regular_pred_label",
        "prob": "regular_prob",
    }, inplace=True)
    # Optional columns (entropy, margin) — fill with defaults if missing
    if "entropy" in out.columns:
        out.rename(columns={"entropy": "regular_entropy"}, inplace=True)
    else:
        out["regular_entropy"] = 0.0
    if "margin" in out.columns:
        out.rename(columns={"margin": "regular_margin"}, inplace=True)
    else:
        out["regular_margin"] = 1.0
    out["regular_pred_label"] = out["regular_pred_label"].astype(str)
    out["regular_prob"] = pd.to_numeric(out["regular_prob"], errors="coerce").fillna(0.0)
    out["regular_entropy"] = pd.to_numeric(out["regular_entropy"], errors="coerce").fillna(0.0)
    out["regular_margin"] = pd.to_numeric(out["regular_margin"], errors="coerce").fillna(1.0)
    return out


def load_cascade_from_exp04(xlsx_path: Path) -> pd.DataFrame:
    """Load token-level cascaded predictions from an Exp04 (or Exp05) output file."""
    df = pd.read_excel(xlsx_path, sheet_name="detailed_results")

    if "eval_mode" in df.columns:
        df = df[df["eval_mode"].astype(str) == "predicted"].copy()

    required = {"sentence_id", "token_idx", "token", "true_bio", "true_etype",
                "pred_bio", "pred_etype", "entity_prob", "bio_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp04 detailed_results missing columns: {sorted(missing)}")

    df["sentence_id"] = df["sentence_id"].astype(int)
    df["token_idx"] = df["token_idx"].astype(int)

    cascade_true_label = [
        _bio_type_to_label(b, t) for b, t in zip(df["true_bio"], df["true_etype"])
    ]
    cascade_pred_label = [
        _bio_type_to_label(b, t) for b, t in zip(df["pred_bio"], df["pred_etype"])
    ]

    entity_prob = pd.to_numeric(df["entity_prob"], errors="coerce").fillna(0.0).values
    bio_prob = pd.to_numeric(df["bio_prob"], errors="coerce").fillna(0.0).values
    pred_bio = df["pred_bio"].astype(str).values

    cascade_prob = np.where(pred_bio == "O", 1.0 - entity_prob, entity_prob * bio_prob)
    cascade_entropy = np.array([
        -p * math.log(p + 1e-10) - (1 - p) * math.log(1 - p + 1e-10) for p in cascade_prob
    ])
    cascade_margin = np.abs(cascade_prob - 0.5) * 2.0

    out = pd.DataFrame({
        "sentence_id": df["sentence_id"].values,
        "token_idx": df["token_idx"].values,
        "token_cascade": df["token"].astype(str).values,
        "cascade_true_label": cascade_true_label,
        "cascade_pred_label": cascade_pred_label,
        "cascade_prob": cascade_prob,
        "cascade_entropy": cascade_entropy,
        "cascade_margin": cascade_margin,
        "entity_prob": entity_prob,
        "bio_prob": bio_prob,
        "cascade_bio": df["pred_bio"].astype(str).values,
        "cascade_etype": df["pred_etype"].astype(str).fillna("None").values,
    })
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_regular_cascade(regular_df: pd.DataFrame, cascade_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on (sentence_id, token_idx) and add derived columns."""
    merged = regular_df.merge(cascade_df, on=["sentence_id", "token_idx"], how="inner")
    if merged.empty:
        raise RuntimeError("No aligned tokens between Exp01 and Exp04 outputs.  "
                           "Were they run with the same data split?")

    merged["true_label"] = merged["true_label"].astype(str)
    merged["cascade_true_label"] = merged["cascade_true_label"].astype(str)
    merged["disagree"] = (
        merged["regular_pred_label"].astype(str) != merged["cascade_pred_label"].astype(str)
    )

    # Derived features used by some strategies
    merged["prob_diff"] = merged["regular_prob"] - merged["cascade_prob"]
    merged["abs_prob_diff"] = merged["prob_diff"].abs()
    merged["max_prob"] = merged[["regular_prob", "cascade_prob"]].max(axis=1)

    # BIO/etype parts for SVM features
    def _split_label(lbl):
        lbl = str(lbl)
        if lbl == "O":
            return "O", "None"
        parts = lbl.split("-", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "None")

    reg_parts = merged["regular_pred_label"].map(_split_label)
    merged["regular_bio"] = reg_parts.map(lambda t: t[0])
    merged["regular_etype"] = reg_parts.map(lambda t: t[1])

    return merged


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def to_seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str):
    y_true, y_pred = [], []
    for sid in sorted(df["sentence_id"].unique()):
        sdf = df[df["sentence_id"] == sid].sort_values("token_idx")
        y_true.append(sdf[true_col].astype(str).tolist())
        y_pred.append(sdf[pred_col].astype(str).tolist())
    return y_true, y_pred


def compute_metrics(df: pd.DataFrame, true_col: str = "true_label", pred_col: str = "fused_pred_label"):
    y_true, y_pred = to_seqeval_lists(df, true_col, pred_col)
    if not y_true:
        return None, None, None
    return (
        float(f1_score(y_true, y_pred)),
        float(precision_score(y_true, y_pred)),
        float(recall_score(y_true, y_pred)),
    )


# ---------------------------------------------------------------------------
# Error-analysis helpers (confusion matrix, per-type metrics, error taxonomy)
# ---------------------------------------------------------------------------

def _split_bio(label: str) -> tuple[str, str]:
    """Split a BIO tag into (prefix, entity_type). 'B-PER' -> ('B', 'PER')."""
    label = str(label)
    if "-" in label:
        prefix, etype = label.split("-", 1)
        return prefix, etype
    return label, ""


def classify_error(true_label: str, pred_label: str) -> str:
    """Row-level NER error taxonomy for a single token."""
    true_label = str(true_label)
    pred_label = str(pred_label)
    if true_label == pred_label:
        return "correct"
    true_o = true_label == "O"
    pred_o = pred_label == "O"
    if true_o and not pred_o:
        return "false_positive"   # spurious entity predicted on a non-entity token
    if pred_o and not true_o:
        return "false_negative"   # real entity token predicted as O (missed)
    # Both are non-O but differ.
    _, true_etype = _split_bio(true_label)
    _, pred_etype = _split_bio(pred_label)
    if true_etype != pred_etype:
        return "type_error"       # correct that it's an entity, wrong entity type
    return "boundary_error"       # same entity type, wrong B/I boundary


def build_confusion_matrix(
    df: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "fused_pred_label",
) -> pd.DataFrame:
    """Token-level confusion matrix as a grid (rows = true tag, cols = predicted tag)."""
    ct = pd.crosstab(
        df[true_col].astype(str),
        df[pred_col].astype(str),
        rownames=["true \\ pred"],
        colnames=[""],
        dropna=False,
    )
    return ct.reset_index()


def build_per_type_metrics(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
    true_col: str = "true_label",
    pred_col: str = "fused_pred_label",
) -> pd.DataFrame:
    """Entity-level precision/recall/F1/support per entity type (seqeval), long format."""
    from seqeval.metrics import classification_report

    y_true, y_pred = to_seqeval_lists(df, true_col, pred_col)
    rows: list[dict] = []
    if y_true:
        try:
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        except TypeError:
            report = classification_report(y_true, y_pred, output_dict=True)
        for label, metrics in report.items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "model": model,
                "split_condition": split_condition,
                "seed": seed,
                "entity_type": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1-score"),
                "support": metrics.get("support"),
            })
    return pd.DataFrame(rows)


def build_error_type_summary(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
    error_col: str = "error_type",
) -> pd.DataFrame:
    """Count of each token-level error type (long format)."""
    counts = df[error_col].value_counts()
    total = int(len(df))
    rows = [{
        "model": model,
        "split_condition": split_condition,
        "seed": seed,
        "error_type": err,
        "count": int(n),
        "pct_of_tokens": (float(n) / total) if total else 0.0,
    } for err, n in counts.items()]
    return pd.DataFrame(rows)


def build_error_examples(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
    max_examples: int = 300,
    context_window: int = 5,
) -> pd.DataFrame:
    """Qualitative sample of misclassified tokens with surrounding sentence context.

    The target token is wrapped in >>> <<< inside the ``context`` column.
    """
    errors = df[df["error_type"] != "correct"]
    if errors.empty:
        return pd.DataFrame()

    # Pre-build an ordered token list per sentence for context windows.
    sent_lookup: dict = {}
    for sid, sdf in df.sort_values("token_idx").groupby("sentence_id"):
        sent_lookup[sid] = (
            sdf["token"].astype(str).tolist(),
            sdf["token_idx"].tolist(),
        )

    rows: list[dict] = []
    for _, r in errors.head(max_examples).iterrows():
        sid = r["sentence_id"]
        tokens, idxs = sent_lookup.get(sid, ([], []))
        try:
            pos = idxs.index(r["token_idx"])
        except ValueError:
            pos = None
        if pos is not None:
            lo = max(0, pos - context_window)
            hi = min(len(tokens), pos + context_window + 1)
            ctx = (
                tokens[lo:pos]
                + [f">>> {tokens[pos]} <<<"]
                + tokens[pos + 1:hi]
            )
            context = " ".join(ctx)
        else:
            context = str(r["token"])
        rows.append({
            "model": model,
            "split_condition": split_condition,
            "seed": seed,
            "sentence_id": sid,
            "token_idx": r["token_idx"],
            "token": r["token"],
            "true_label": r["true_label"],
            "fused_pred_label": r["fused_pred_label"],
            "error_type": r["error_type"],
            "selected_source": r.get("selected_source", ""),
            "selected_confidence": r.get("selected_confidence", ""),
            "context": context,
        })
    return pd.DataFrame(rows)


def build_confidence_analysis(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
) -> pd.DataFrame:
    """Reliability table: accuracy vs. selected_confidence bucket (calibration check)."""
    d = df.copy()
    d["is_correct"] = (d["error_type"] == "correct")
    d["conf"] = pd.to_numeric(d["selected_confidence"], errors="coerce")
    d = d[d["conf"].notna()]
    if d.empty:
        return pd.DataFrame()

    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0"]
    d["conf_bucket"] = pd.cut(d["conf"], bins=bins, labels=labels, right=False)

    rows: list[dict] = []
    for bucket, gdf in d.groupby("conf_bucket", observed=True):
        n = int(len(gdf))
        n_correct = int(gdf["is_correct"].sum())
        rows.append({
            "model": model,
            "split_condition": split_condition,
            "seed": seed,
            "confidence_bucket": str(bucket),
            "tokens": n,
            "correct": n_correct,
            "accuracy": (n_correct / n) if n else 0.0,
            "mean_confidence": float(gdf["conf"].mean()),
        })
    # Overall correct-vs-incorrect mean confidence (calibration gap indicator).
    correct_conf = d.loc[d["is_correct"], "conf"]
    wrong_conf = d.loc[~d["is_correct"], "conf"]
    rows.append({
        "model": model,
        "split_condition": split_condition,
        "seed": seed,
        "confidence_bucket": "ALL_correct",
        "tokens": int(len(correct_conf)),
        "correct": int(len(correct_conf)),
        "accuracy": 1.0,
        "mean_confidence": float(correct_conf.mean()) if len(correct_conf) else 0.0,
    })
    rows.append({
        "model": model,
        "split_condition": split_condition,
        "seed": seed,
        "confidence_bucket": "ALL_incorrect",
        "tokens": int(len(wrong_conf)),
        "correct": 0,
        "accuracy": 0.0,
        "mean_confidence": float(wrong_conf.mean()) if len(wrong_conf) else 0.0,
    })
    return pd.DataFrame(rows)


def build_disagreement_analysis(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
) -> pd.DataFrame:
    """On disagreement tokens: how well the fusion router selected the correct source.

    Reports fused accuracy vs. the *oracle* (best-possible) accuracy where at least
    one source was correct — this quantifies how much fusion recovered.
    """
    dis = df[df["disagree"].astype(bool)].copy()
    if dis.empty:
        return pd.DataFrame([{
            "model": model,
            "split_condition": split_condition,
            "seed": seed,
            "disagreement_tokens": 0,
            "note": "no disagreements",
        }])

    reg_ok = dis["regular_pred_label"].astype(str) == dis["true_label"].astype(str)
    cas_ok = dis["cascade_pred_label"].astype(str) == dis["true_label"].astype(str)
    fused_ok = dis["fused_pred_label"].astype(str) == dis["true_label"].astype(str)
    oracle_ok = reg_ok | cas_ok
    n = int(len(dis))

    rows = [{
        "model": model,
        "split_condition": split_condition,
        "seed": seed,
        "disagreement_tokens": n,
        "regular_correct": int(reg_ok.sum()),
        "cascade_correct": int(cas_ok.sum()),
        "fused_correct": int(fused_ok.sum()),
        "oracle_correct": int(oracle_ok.sum()),
        "fused_accuracy": float(fused_ok.mean()),
        "oracle_accuracy": float(oracle_ok.mean()),
        "router_recovery_rate": (
            float(fused_ok.sum() / oracle_ok.sum()) if oracle_ok.sum() else 0.0
        ),
    }]
    return pd.DataFrame(rows)


def _extract_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Extract entity spans (start, end_exclusive, entity_type) from BIO labels."""
    spans: list[tuple[int, int, str]] = []
    cur_start = None
    cur_type = None
    for i, lab in enumerate(labels):
        prefix, etype = _split_bio(str(lab))
        if prefix == "B":
            if cur_start is not None:
                spans.append((cur_start, i, cur_type))  # type: ignore[arg-type]
            cur_start, cur_type = i, etype
        elif prefix == "I" and cur_start is not None and etype == cur_type:
            continue
        else:
            if cur_start is not None:
                spans.append((cur_start, i, cur_type))  # type: ignore[arg-type]
            cur_start, cur_type = None, None
    if cur_start is not None:
        spans.append((cur_start, len(labels), cur_type))  # type: ignore[arg-type]
    return spans


def build_entity_length_analysis(
    df: pd.DataFrame,
    model: str,
    split_condition: str,
    seed: str,
) -> pd.DataFrame:
    """Entity-level recall grouped by true entity span length (in tokens)."""
    rows_by_len: dict[int, list[int]] = {}
    for sid, sdf in df.sort_values("token_idx").groupby("sentence_id"):
        true_labels = sdf["true_label"].astype(str).tolist()
        pred_labels = sdf["fused_pred_label"].astype(str).tolist()
        true_spans = _extract_spans(true_labels)
        pred_spans = set(_extract_spans(pred_labels))
        for span in true_spans:
            length = span[1] - span[0]
            matched = 1 if span in pred_spans else 0
            rows_by_len.setdefault(length, []).append(matched)

    out: list[dict] = []
    for length in sorted(rows_by_len):
        matches = rows_by_len[length]
        support = len(matches)
        recovered = sum(matches)
        out.append({
            "model": model,
            "split_condition": split_condition,
            "seed": seed,
            "entity_length_tokens": length,
            "true_entities": support,
            "correctly_detected": recovered,
            "recall": (recovered / support) if support else 0.0,
        })
    return pd.DataFrame(out)


# Ordered description of every sheet + key columns written to the workbook.
_SHEET_DOCS: list[tuple[str, str]] = [
    ("metrics",
     "One row summarising this run. Entity-level seqeval F1/precision/recall over the "
     "fused predictions, plus token counts, disagreement counts and how often each "
     "source was selected. Columns: model, split_condition, seed identify the run."),
    ("detailed_results",
     "One row per aligned token. true_label vs fused_pred_label with each source's "
     "prediction and probability. 'disagree' = the two sources disagreed. "
     "'selected_source' / 'selected_confidence' = what fusion chose. "
     "'error_type' = per-token error taxonomy (see error_type_summary)."),
    ("confusion_matrix",
     "Token-level confusion matrix. Rows = true BIO tag, columns = predicted fused tag; "
     "cell = token count. Diagonal = correct. Read off which tags get confused."),
    ("per_type_metrics",
     "Entity-level (seqeval) precision/recall/F1/support per entity type, plus "
     "micro/macro/weighted averages. 'support' = number of true entities of that type."),
    ("error_type_summary",
     "Count and % of each token-level error category: correct, false_positive "
     "(spurious entity on a true-O token), false_negative (real entity token predicted O), "
     "type_error (right that it is an entity, wrong type), boundary_error "
     "(right type, wrong B/I boundary)."),
    ("error_examples",
     "Up to 300 misclassified tokens with sentence context. The offending token is "
     "wrapped in >>> <<< in the 'context' column for qualitative reading."),
    ("confidence_analysis",
     "Reliability / calibration table: token accuracy per selected_confidence bucket, "
     "plus mean confidence for correct vs incorrect tokens. A large gap between "
     "ALL_correct and ALL_incorrect mean_confidence indicates useful confidence signal."),
    ("disagreement_analysis",
     "Only tokens where the two sources disagreed. Compares fused accuracy to the "
     "'oracle' (best possible when at least one source was right). "
     "router_recovery_rate = fused_correct / oracle_correct."),
    ("entity_length_analysis",
     "Entity-level recall grouped by true entity span length (in tokens). "
     "Shows whether longer multi-token entities are harder to detect."),
    ("regular_from_exp01",
     "Raw Exp01 (Regular NER) token predictions used as the first fusion source."),
    ("cascade_from_source",
     "Raw Exp04/Exp05 (Cascaded Pipeline) token predictions used as the second source."),
]


def build_documentation_sheet(
    experiment_name: str,
    model: str,
    split_condition: str,
    seed: str,
    cascade_source: str,
) -> pd.DataFrame:
    """Human-readable 'read me' sheet describing every other sheet in the workbook."""
    header = [
        {"section": "ABOUT", "item": "experiment", "description": experiment_name},
        {"section": "ABOUT", "item": "model", "description": model},
        {"section": "ABOUT", "item": "split_condition", "description": split_condition},
        {"section": "ABOUT", "item": "seed", "description": str(seed)},
        {"section": "ABOUT", "item": "cascade_source", "description": cascade_source},
        {"section": "ABOUT", "item": "scope",
         "description": "This workbook = ONE run (one model x one split_condition x one seed). "
                        "Aggregate across seeds with aggregate_fusion_error_analysis.py."},
        {"section": "ABOUT", "item": "metric_note",
         "description": "F1/precision/recall are entity-level (seqeval). The confusion "
                        "matrix and error_type_summary are token-level (BIO tags)."},
    ]
    sheets = [
        {"section": "SHEET", "item": name, "description": desc}
        for name, desc in _SHEET_DOCS
    ]
    return pd.DataFrame(header + sheets)


# ---------------------------------------------------------------------------
# Generic ready-fusion entry point
# ---------------------------------------------------------------------------

def run_ready_fusion(
    *,
    strategy_fn: Callable[[pd.DataFrame], pd.DataFrame],
    experiment_id: str,
    experiment_name: str,
    description: str,
    result_basename: str,
    cascade_source: str = "exp04",
    extra_info: dict | None = None,
) -> dict:
    """
    Load Exp01 + Exp04/05 ready outputs, apply *strategy_fn*, compute metrics, save.

    Parameters
    ----------
    strategy_fn : callable(df) -> df
        Receives the merged DataFrame and must add columns:
        ``fused_pred_label``, ``selected_source``, ``selected_confidence``.
    cascade_source : str
        ``"exp04"`` (default) or ``"exp05"`` — which experiment to load cascade
        predictions from.
    extra_info : dict
        Additional keys to include in the result JSON.
    """
    exp01_xlsx = _resolve_source("exp01", "THESIS_READY_EXP01_XLSX")

    cascade_env_var = "THESIS_READY_EXP05_XLSX" if cascade_source == "exp05" else "THESIS_READY_EXP04_XLSX"
    cascade_xlsx = _resolve_source(cascade_source, cascade_env_var)

    regular_df = load_regular_from_exp01(exp01_xlsx)
    cascade_df = load_cascade_from_exp04(cascade_xlsx)

    merged = merge_regular_cascade(regular_df, cascade_df)

    mismatched_truth = int((merged["true_label"] != merged["cascade_true_label"]).sum())

    # Apply the strategy
    merged = strategy_fn(merged)

    # Validate required output columns
    for col in ("fused_pred_label", "selected_source", "selected_confidence"):
        if col not in merged.columns:
            raise RuntimeError(f"strategy_fn must add column '{col}' to the merged DataFrame")

    f1, precision, recall = compute_metrics(merged)

    disagreement_count = int(merged["disagree"].sum())

    model_display = _resolve_model_display()
    split_condition = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
    split_seed = os.environ.get("THESIS_SPLIT_SEED", "42")

    metrics_df = pd.DataFrame([{
        "dataset_name": "ready_results_merge",
        "model": model_display,
        "split_condition": split_condition,
        "seed": split_seed,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tokens_aligned": len(merged),
        "disagreements": disagreement_count,
        "selected_regular": int(merged["selected_source"].astype(str).str.contains("regular").sum()),
        "selected_cascade": int(merged["selected_source"].astype(str).str.contains("cascade").sum()),
        "agreements": int((~merged["disagree"]).sum()),
        "truth_label_mismatch_between_sources": mismatched_truth,
        "cascade_source": cascade_source,
    }])

    if extra_info:
        for k, v in extra_info.items():
            metrics_df[k] = v

    merged["model"] = model_display
    merged["split_condition"] = split_condition
    merged["seed"] = split_seed
    merged["error_type"] = [
        classify_error(t, p)
        for t, p in zip(merged["true_label"], merged["fused_pred_label"])
    ]

    detailed_cols = [
        "model", "split_condition", "seed",
        "sentence_id", "token_idx", "token", "true_label",
        "regular_pred_label", "regular_prob",
        "cascade_pred_label", "cascade_prob",
        "disagree", "selected_source", "selected_confidence", "fused_pred_label",
        "error_type",
    ]
    # Include any extra columns the strategy added
    for col in merged.columns:
        if col not in detailed_cols and col.startswith(("calibrated_", "entropy_", "learned_", "svm_")):
            detailed_cols.append(col)
    detailed_df = merged[[c for c in detailed_cols if c in merged.columns]]

    # Error-analysis artifacts (per this model / split condition / seed run).
    confusion_df = build_confusion_matrix(merged)
    per_type_df = build_per_type_metrics(merged, model_display, split_condition, split_seed)
    error_summary_df = build_error_type_summary(merged, model_display, split_condition, split_seed)
    error_examples_df = build_error_examples(merged, model_display, split_condition, split_seed)
    confidence_df = build_confidence_analysis(merged, model_display, split_condition, split_seed)
    disagreement_df = build_disagreement_analysis(merged, model_display, split_condition, split_seed)
    entity_length_df = build_entity_length_analysis(merged, model_display, split_condition, split_seed)
    documentation_df = build_documentation_sheet(
        experiment_name, model_display, split_condition, split_seed, cascade_source
    )

    metrics_file = write_result_excel(
        experiment_id,
        f"{result_basename}_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "documentation": documentation_df,
            "confusion_matrix": confusion_df,
            "per_type_metrics": per_type_df,
            "error_type_summary": error_summary_df,
            "error_examples": error_examples_df,
            "confidence_analysis": confidence_df,
            "disagreement_analysis": disagreement_df,
            "entity_length_analysis": entity_length_df,
            "regular_from_exp01": regular_df,
            "cascade_from_source": cascade_df,
        },
    )

    result = {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "description": description,
        "mode": "ready",
        "model": model_display,
        "split_condition": split_condition,
        "seed": split_seed,
        "cascade_source": cascade_source,
        "source_exp01_xlsx": str(exp01_xlsx),
        "source_cascade_xlsx": str(cascade_xlsx),
        "metrics_file": str(metrics_file),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "status": "ok",
    }
    if extra_info:
        result.update(extra_info)

    out_path = write_result_json(experiment_id, result_basename, result)
    result["result_file"] = str(out_path)
    return result
