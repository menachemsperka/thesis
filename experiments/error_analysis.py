"""
error_analysis.py — Shared error-analysis artifacts for NER experiments.

Produces a consistent set of Excel sheets (with an in-workbook documentation sheet)
from any token-level DataFrame that has a true-label column and a predicted-label
column. Used by the ready-fusion experiments and by experiments 01 / 03 / 05 so a
single error-analysis format can be reported across the whole thesis.

Sheets produced by :func:`build_error_analysis_sheets`:
    documentation, confusion_matrix, per_type_metrics, error_type_summary,
    error_examples, confidence_analysis (optional), entity_length_analysis.

Error taxonomy (:func:`classify_error`, token level):
    correct / false_positive / false_negative / type_error / boundary_error.
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Run-context helpers
# ---------------------------------------------------------------------------

# Maps a substring of a model id / path to a human-friendly display name.
_MODEL_DISPLAY_NAMES: tuple[tuple[str, str], ...] = (
    ("berel", "BEREL 3.0"),
    ("dictabert", "DictaBERT"),
    ("alephbertgimmel", "AlephBERT-Gimmel"),
    ("hero", "HeRo"),
)


def model_display_name(raw: str | None) -> str:
    """Map a model id or local path to a friendly name (e.g. 'DictaBERT')."""
    raw = (raw or "").strip()
    if not raw:
        return "unknown"
    lowered = raw.lower()
    for needle, display in _MODEL_DISPLAY_NAMES:
        if needle in lowered:
            return display
    return raw.replace("\\", "/").rstrip("/").split("/")[-1] or raw


# ---------------------------------------------------------------------------
# Label helpers + per-token error taxonomy
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
    _, true_etype = _split_bio(true_label)
    _, pred_etype = _split_bio(pred_label)
    if true_etype != pred_etype:
        return "type_error"       # correct that it's an entity, wrong entity type
    return "boundary_error"       # same entity type, wrong B/I boundary


def annotate_error_types(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    """Return a copy of *df* with an added ``error_type`` column."""
    out = df.copy()
    out["error_type"] = [
        classify_error(t, p) for t, p in zip(out[true_col], out[pred_col])
    ]
    return out


def _to_seqeval_lists(df, sentence_col, token_idx_col, true_col, pred_col):
    y_true, y_pred = [], []
    for sid in sorted(df[sentence_col].unique()):
        sdf = df[df[sentence_col] == sid].sort_values(token_idx_col)
        y_true.append(sdf[true_col].astype(str).tolist())
        y_pred.append(sdf[pred_col].astype(str).tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Individual sheet builders
# ---------------------------------------------------------------------------

def build_confusion_matrix(df, true_col, pred_col) -> pd.DataFrame:
    """Token-level confusion matrix as a grid (rows = true tag, cols = predicted tag)."""
    ct = pd.crosstab(
        df[true_col].astype(str),
        df[pred_col].astype(str),
        rownames=["true \\ pred"],
        colnames=[""],
        dropna=False,
    )
    return ct.reset_index()


def build_per_type_metrics(df, model, split_condition, seed,
                           sentence_col, token_idx_col, true_col, pred_col) -> pd.DataFrame:
    """Entity-level precision/recall/F1/support per entity type (seqeval), long format."""
    from seqeval.metrics import classification_report

    y_true, y_pred = _to_seqeval_lists(df, sentence_col, token_idx_col, true_col, pred_col)
    rows: list[dict] = []
    if y_true:
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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


def build_error_type_summary(df, model, split_condition, seed, error_col="error_type") -> pd.DataFrame:
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


def build_error_examples(df, model, split_condition, seed,
                         sentence_col, token_idx_col, token_col, true_col, pred_col,
                         confidence_col=None, source_col=None,
                         max_examples=300, context_window=5) -> pd.DataFrame:
    """Qualitative sample of misclassified tokens with surrounding sentence context.

    The target token is wrapped in >>> <<< inside the ``context`` column.
    """
    errors = df[df["error_type"] != "correct"]
    if errors.empty:
        return pd.DataFrame()

    sent_lookup: dict = {}
    for sid, sdf in df.sort_values(token_idx_col).groupby(sentence_col):
        sent_lookup[sid] = (
            sdf[token_col].astype(str).tolist(),
            sdf[token_idx_col].tolist(),
        )

    rows: list[dict] = []
    for _, r in errors.head(max_examples).iterrows():
        sid = r[sentence_col]
        tokens, idxs = sent_lookup.get(sid, ([], []))
        try:
            pos = idxs.index(r[token_idx_col])
        except ValueError:
            pos = None
        if pos is not None:
            lo = max(0, pos - context_window)
            hi = min(len(tokens), pos + context_window + 1)
            ctx = tokens[lo:pos] + [f">>> {tokens[pos]} <<<"] + tokens[pos + 1:hi]
            context = " ".join(ctx)
        else:
            context = str(r[token_col])
        row = {
            "model": model,
            "split_condition": split_condition,
            "seed": seed,
            "sentence_id": sid,
            "token_idx": r[token_idx_col],
            "token": r[token_col],
            "true_label": r[true_col],
            "pred_label": r[pred_col],
            "error_type": r["error_type"],
        }
        if source_col and source_col in r:
            row["selected_source"] = r.get(source_col, "")
        if confidence_col and confidence_col in r:
            row["confidence"] = r.get(confidence_col, "")
        row["context"] = context
        rows.append(row)
    return pd.DataFrame(rows)


def build_confidence_analysis(df, model, split_condition, seed, confidence_col) -> pd.DataFrame:
    """Reliability table: accuracy vs. confidence bucket (calibration check)."""
    d = df.copy()
    d["is_correct"] = (d["error_type"] == "correct")
    d["conf"] = pd.to_numeric(d[confidence_col], errors="coerce")
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
    correct_conf = d.loc[d["is_correct"], "conf"]
    wrong_conf = d.loc[~d["is_correct"], "conf"]
    rows.append({
        "model": model, "split_condition": split_condition, "seed": seed,
        "confidence_bucket": "ALL_correct", "tokens": int(len(correct_conf)),
        "correct": int(len(correct_conf)), "accuracy": 1.0,
        "mean_confidence": float(correct_conf.mean()) if len(correct_conf) else 0.0,
    })
    rows.append({
        "model": model, "split_condition": split_condition, "seed": seed,
        "confidence_bucket": "ALL_incorrect", "tokens": int(len(wrong_conf)),
        "correct": 0, "accuracy": 0.0,
        "mean_confidence": float(wrong_conf.mean()) if len(wrong_conf) else 0.0,
    })
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


def build_entity_length_analysis(df, model, split_condition, seed,
                                 sentence_col, token_idx_col, true_col, pred_col) -> pd.DataFrame:
    """Entity-level recall grouped by true entity span length (in tokens)."""
    rows_by_len: dict[int, list[int]] = {}
    for sid, sdf in df.sort_values(token_idx_col).groupby(sentence_col):
        true_labels = sdf[true_col].astype(str).tolist()
        pred_labels = sdf[pred_col].astype(str).tolist()
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


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

# Canonical descriptions for the sheets this module produces.
CANONICAL_SHEET_DOCS: dict[str, str] = {
    "confusion_matrix":
        "Token-level confusion matrix. Rows = true BIO tag, columns = predicted tag; "
        "cell = token count. Diagonal = correct. Shows which tags get confused.",
    "per_type_metrics":
        "Entity-level (seqeval) precision/recall/F1/support per entity type, plus "
        "micro/macro/weighted averages. 'support' = number of true entities of that type.",
    "error_type_summary":
        "Count and % of each token-level error category: correct, false_positive "
        "(spurious entity on a true-O token), false_negative (real entity token predicted O), "
        "type_error (right that it is an entity, wrong type), boundary_error "
        "(right type, wrong B/I boundary).",
    "error_examples":
        "Up to 300 misclassified tokens with sentence context. The offending token is "
        "wrapped in >>> <<< in the 'context' column for qualitative reading.",
    "confidence_analysis":
        "Reliability / calibration table: token accuracy per confidence bucket, plus mean "
        "confidence for correct vs incorrect tokens (ALL_correct / ALL_incorrect rows).",
    "entity_length_analysis":
        "Entity-level recall grouped by true entity span length (in tokens). "
        "Shows whether longer multi-token entities are harder to detect.",
}


def build_documentation_sheet(about_rows: list[dict], sheet_docs: list[tuple[str, str]]) -> pd.DataFrame:
    """Assemble the 'read me' sheet from ABOUT rows and per-sheet descriptions."""
    header = [
        {"section": "ABOUT", "item": item, "description": desc}
        for item, desc in about_rows
    ]
    sheets = [
        {"section": "SHEET", "item": name, "description": desc}
        for name, desc in sheet_docs
    ]
    return pd.DataFrame(header + sheets)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_error_analysis_sheets(
    df: pd.DataFrame,
    *,
    experiment_name: str,
    model: str,
    split_condition: str,
    seed: str,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    sentence_col: str = "sentence_id",
    token_idx_col: str = "token_idx",
    token_col: str = "token",
    confidence_col: str | None = None,
    source_col: str | None = None,
    extra_about: list[tuple[str, str]] | None = None,
    extra_sheet_docs: list[tuple[str, str]] | None = None,
) -> "dict[str, pd.DataFrame]":
    """Build the full ordered set of error-analysis sheets for one run.

    Returns an ordered dict ``{sheet_name: DataFrame}`` beginning with
    ``documentation``. The input *df* must be token-level with the given columns.
    """
    work = df.copy()
    work[true_col] = work[true_col].astype(str)
    work[pred_col] = work[pred_col].astype(str)
    work["error_type"] = [
        classify_error(t, p) for t, p in zip(work[true_col], work[pred_col])
    ]

    sheets: dict[str, pd.DataFrame] = {}
    sheets["confusion_matrix"] = build_confusion_matrix(work, true_col, pred_col)
    sheets["per_type_metrics"] = build_per_type_metrics(
        work, model, split_condition, seed, sentence_col, token_idx_col, true_col, pred_col)
    sheets["error_type_summary"] = build_error_type_summary(work, model, split_condition, seed)

    examples = build_error_examples(
        work, model, split_condition, seed,
        sentence_col, token_idx_col, token_col, true_col, pred_col,
        confidence_col=confidence_col, source_col=source_col)
    if not examples.empty:
        sheets["error_examples"] = examples

    if confidence_col and confidence_col in work.columns:
        conf = build_confidence_analysis(work, model, split_condition, seed, confidence_col)
        if not conf.empty:
            sheets["confidence_analysis"] = conf

    sheets["entity_length_analysis"] = build_entity_length_analysis(
        work, model, split_condition, seed, sentence_col, token_idx_col, true_col, pred_col)

    about = [
        ("experiment", experiment_name),
        ("model", model),
        ("split_condition", split_condition),
        ("seed", str(seed)),
        ("scope",
         "This workbook = ONE run (one model x one split_condition x one seed). "
         "Aggregate across seeds with aggregate_fusion_error_analysis.py."),
        ("metric_note",
         "F1/precision/recall are entity-level (seqeval). The confusion matrix and "
         "error_type_summary are token-level (BIO tags)."),
    ]
    if extra_about:
        about.extend(extra_about)

    sheet_docs = [(name, CANONICAL_SHEET_DOCS[name]) for name in sheets
                  if name in CANONICAL_SHEET_DOCS]
    if extra_sheet_docs:
        sheet_docs.extend(extra_sheet_docs)

    ordered: dict[str, pd.DataFrame] = {
        "documentation": build_documentation_sheet(about, sheet_docs)
    }
    ordered.update(sheets)
    return ordered
