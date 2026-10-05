"""
build_unified_protocol_table.py — Re-score every method on the SAME nested-CV
outer-test tokens, so fusion is comparable to the base models under one protocol.

No retraining is required. The ``detailed_results`` sheet of each ``06_*_oof``
result file already holds, for every pooled outer-test token: the gold label, the
Exp01 (direct NER) label, the Exp04 (cascade) label, both confidences, the
cascade BIO/type parts, and the router's fused label. This script recomputes
strict entity-level F1 for each method over those identical tokens and writes one
workbook containing the unified main table, fusion-vs-base paired deltas,
per-fold F1 and paired fold tests.

Usage (from the repo root, e.g. on Colab):

    python build_unified_protocol_table.py \
      --oof-dir outputs/cross_comparison_150sent \
      --output  outputs/unified_protocol_table.xlsx

Add the corpus JSONs to unlock the per-fold sheets (fold membership is
recomputed deterministically from the same stratifier and seed the runner used):

    python build_unified_protocol_table.py \
      --oof-dir outputs/cross_comparison_150sent \
      --corpus-train-json outputs/exp07/splits/<condition>_train.json \
      --corpus-eval-json  outputs/exp07/splits/<condition>_eval.json \
      --output outputs/unified_protocol_table.xlsx

Pass the runner's workbook to copy its paper sheets in verbatim, so the output is
the single file you write the paper from:

      --reference-xlsx outputs/cross_comparison_150sent/cross_comparison_latest.xlsx
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_COLUMNS = {
    "sentence_id",
    "token_idx",
    "true_label",
    "regular_pred_label",
    "cascade_pred_label",
    "regular_prob",
    "cascade_prob",
    "fused_pred_label",
}

# (sheet label, prediction column) in reporting order. The router row is appended
# per file because its label depends on which classifier produced the file.
# Sheets copied verbatim from the runner's cross_comparison_latest.xlsx so that the
# unified workbook is self-contained and nothing else has to be opened to write the paper.
REFERENCE_SHEETS: tuple[str, ...] = (
    "journal_main_table",
    "journal_oof_fold_summary",
    "journal_oof_folds_long",
    "journal_paired_fold_deltas",
    "journal_lambda_grid",
    "journal_loss_config",
    "paired_tests",
)

BASE_METHODS: tuple[tuple[str, str], ...] = (
    ("Direct NER (Exp01)", "regular_pred_label"),
    ("Three-step cascade (Exp04)", "cascade_pred_label"),
    ("Cascade + repair (Exp05)", "repaired_cascade_pred_label"),
    ("Confidence fusion (Exp06)", "conf_fusion_pred_label"),
)


# ---------------------------------------------------------------------------
# Metrics — identical grouping and scorer to experiments/fusion_ready_sources.py
# ---------------------------------------------------------------------------

def _seqeval_lists(df: pd.DataFrame, true_col: str, pred_col: str):
    # One sort + one groupby. Filtering per sentence id instead would be O(sentences x rows),
    # which at ~600 workbooks x 30 scorings each dominates the whole run.
    work = df[["sentence_id", "token_idx", true_col, pred_col]].sort_values(
        ["sentence_id", "token_idx"], kind="stable"
    )
    y_true, y_pred = [], []
    for _, group in work.groupby("sentence_id", sort=True):
        y_true.append(group[true_col].astype(str).tolist())
        y_pred.append(group[pred_col].astype(str).tolist())
    return y_true, y_pred


def _score(df: pd.DataFrame, pred_col: str) -> tuple[float, float, float]:
    y_true, y_pred = _seqeval_lists(df, "true_label", pred_col)
    if not y_true:
        return (float("nan"),) * 3
    return (
        float(f1_score(y_true, y_pred)),
        float(precision_score(y_true, y_pred)),
        float(recall_score(y_true, y_pred)),
    )


# ---------------------------------------------------------------------------
# Token error taxonomy
# ---------------------------------------------------------------------------

ERROR_CATEGORIES: tuple[str, ...] = (
    "correct",
    "false_positive",
    "false_negative",
    "type_error",
    "boundary_error",
)


def _load_error_classifier():
    """Import the thesis-wide token error taxonomy.

    Reusing experiments/error_analysis.py keeps these counts directly comparable to
    the error_type_summary sheets the individual experiments already emit, instead of
    defining a second taxonomy that drifts from them.
    """
    experiments_dir = PROJECT_ROOT / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    try:
        from error_analysis import classify_error
    except Exception as exc:
        print(f"  ! cannot import classify_error, error breakdown skipped ({exc})")
        return None
    return classify_error


def _error_counts(df: pd.DataFrame, pred_col: str, classify) -> dict[str, int]:
    """Token counts per error category.

    Classifies each distinct (true, pred) label pair once instead of each token. With
    9 component types there are at most a few hundred pairs against tens of thousands
    of tokens per run, so this keeps the breakdown off the hot path.
    """
    pairs = pd.DataFrame({
        "true_label": df["true_label"].astype(str),
        "pred_label": df[pred_col].astype(str),
    }).value_counts(["true_label", "pred_label"], sort=False)

    counts = dict.fromkeys(ERROR_CATEGORIES, 0)
    for (true_label, pred_label), n in pairs.items():
        counts[classify(true_label, pred_label)] += int(n)
    return counts


# ---------------------------------------------------------------------------
# Derived prediction columns
# ---------------------------------------------------------------------------

def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _repaired_cascade_labels(work: pd.DataFrame) -> list[str]:
    """Exp05 B/I type consistency: on B-X followed by I-Y, keep the higher bio_prob type."""
    bio = work["cascade_bio"].astype(str).tolist()
    etype = work["cascade_etype"].astype(str).tolist()
    bio_prob = pd.to_numeric(work["bio_prob"], errors="coerce").fillna(0.0).tolist()

    # Positional order per sentence from a single sort + groupby; `work` has a 0..n-1 index,
    # so group indices double as offsets into the bio/etype/bio_prob lists.
    ordered = work[["sentence_id", "token_idx"]].sort_values(
        ["sentence_id", "token_idx"], kind="stable"
    )
    for _, group in ordered.groupby("sentence_id", sort=False):
        positions = group.index.tolist()
        for curr, nxt in zip(positions, positions[1:]):
            if bio[curr] == "B" and bio[nxt] == "I" and etype[curr] != etype[nxt]:
                if bio_prob[curr] >= bio_prob[nxt]:
                    etype[nxt] = etype[curr]
                else:
                    etype[curr] = etype[nxt]

    return [_bio_type_to_label(b, t) for b, t in zip(bio, etype)]


def add_derived_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True)

    reg = out["regular_pred_label"].astype(str).to_numpy()
    cas = out["cascade_pred_label"].astype(str).to_numpy()
    reg_prob = pd.to_numeric(out["regular_prob"], errors="coerce").fillna(0.0).to_numpy()
    cas_prob = pd.to_numeric(out["cascade_prob"], errors="coerce").fillna(0.0).to_numpy()

    if "disagree" in out.columns:
        disagree = out["disagree"].astype(bool).to_numpy()
    else:
        disagree = reg != cas
    out["disagree"] = disagree

    # Mirrors the no-router fallback branch of _apply_router: ties go to direct NER.
    out["conf_fusion_pred_label"] = np.where(disagree & (cas_prob > reg_prob), cas, reg)

    if {"cascade_bio", "cascade_etype", "bio_prob"}.issubset(out.columns):
        # Guard: rebuilding the label from the BIO/type parts must reproduce
        # cascade_pred_label exactly, otherwise the repair row is not trustworthy.
        rebuilt = np.array([
            _bio_type_to_label(b, t)
            for b, t in zip(out["cascade_bio"], out["cascade_etype"])
        ])
        mismatches = int(np.sum(rebuilt != cas))
        if mismatches:
            print(f"  ! {mismatches}/{len(out)} tokens disagree when the cascade label is rebuilt "
                  "from cascade_bio/cascade_etype; the 'Cascade + repair' row is unreliable")
        out["repaired_cascade_pred_label"] = _repaired_cascade_labels(out)
    else:
        print("  ! cascade_bio/cascade_etype/bio_prob absent; "
              "'Cascade + repair' falls back to the unrepaired cascade")
        out["repaired_cascade_pred_label"] = cas

    return out


# ---------------------------------------------------------------------------
# OOF file discovery
# ---------------------------------------------------------------------------

# Directory names never worth walking. oof_router_cache holds one exp01/exp04 workbook per
# (outer x inner) fold per run — thousands of files that cannot match the pattern anyway, and
# walking them over a Drive FUSE mount costs minutes.
PRUNED_DIR_NAMES: frozenset[str] = frozenset({
    "oof_router_cache",
    "splits",
    "exp07",
    "exp07_augmented",
    "data",
    "exp04_lambda_grid_cache",
    ".git",
})


def _walk_for_pattern(root: Path, pattern: str) -> list[Path]:
    """Recursive glob that skips PRUNED_DIR_NAMES subtrees."""
    import fnmatch

    matches: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in PRUNED_DIR_NAMES:
                    stack.append(entry)
            elif fnmatch.fnmatch(entry.name, pattern):
                matches.append(entry.resolve())
    return matches


def discover_oof_files(oof_dirs: list[Path], explicit: list[Path], pattern: str) -> list[Path]:
    found: list[Path] = [p.resolve() for p in explicit]
    for directory in oof_dirs:
        if not directory.exists():
            print(f"  ! missing directory, skipped: {directory}")
            continue
        found.extend(sorted(_walk_for_pattern(directory, pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path not in seen and path.suffix.lower() == ".xlsx" and not path.name.startswith("~$"):
            seen.add(path)
            unique.append(path)
    return unique


def read_oof_file(path: Path) -> dict | None:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        print(f"  ! unreadable, skipped: {path.name} ({exc})")
        return None

    detailed = sheets.get("detailed_results")
    if detailed is None or detailed.empty:
        return None

    missing = REQUIRED_COLUMNS - set(detailed.columns)
    if missing:
        print(f"  ! not an OOF fusion file, skipped: {path.name} (missing {sorted(missing)})")
        return None

    metrics = sheets.get("metrics", pd.DataFrame())
    meta = metrics.iloc[0].to_dict() if not metrics.empty else {}
    protocol = str(meta.get("protocol", "")).strip()
    if protocol and protocol != "nested_stratified_cv_oof":
        print(f"  ! protocol is '{protocol}', not nested CV, skipped: {path.name}")
        return None

    return {
        "path": path,
        "meta": meta,
        "detailed": detailed,
        "fold_metrics": sheets.get("fold_metrics", pd.DataFrame()),
    }


def _run_key(meta: dict, path: Path) -> tuple:
    return (
        str(meta.get("model", "unknown")),
        str(meta.get("split_condition", "unknown")),
        str(meta.get("seed", "unknown")),
        str(meta.get("router", path.stem)),
    )


def dedupe_runs(runs: list[dict]) -> list[dict]:
    """Keep the newest file per (model, condition, seed, router)."""
    best: dict[tuple, dict] = {}
    for run in runs:
        key = _run_key(run["meta"], run["path"])
        incumbent = best.get(key)
        if incumbent is None or run["path"].stat().st_mtime > incumbent["path"].stat().st_mtime:
            best[key] = run
    return [best[k] for k in sorted(best)]


# ---------------------------------------------------------------------------
# Fold reconstruction
# ---------------------------------------------------------------------------

def _read_sentences(path: Path) -> list[dict]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{"text": str(i.get("text", "")), "labels": list(i.get("labels", []))} for i in raw]


def _load_stratifier():
    experiments_dir = PROJECT_ROOT / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    try:
        from exp07_split_artifacts import multilabel_stratified_kfold_assignments
    except Exception as exc:
        print(f"  ! cannot import the stratifier, per-fold sheets skipped ({exc})")
        return None
    return multilabel_stratified_kfold_assignments


def reconstruct_fold_map(
    train_json: Path,
    eval_json: Path,
    n_folds: int,
    split_seed: int,
) -> dict[int, int] | None:
    """Map sentence_id -> outer fold, reproducing the runner's fold assignment.

    sentence_id is the 1-based position in (train JSON + eval JSON), so the mapping
    is specific to one split condition: the same corpus ordered differently per
    condition yields different fold membership.
    """
    stratify = _load_stratifier()
    if stratify is None:
        return None
    if not train_json.exists() or not eval_json.exists():
        print(f"  ! split JSONs not found ({train_json.name} / {eval_json.name})")
        return None

    sentences = _read_sentences(train_json) + _read_sentences(eval_json)
    # Same call the runner makes: outer_assign = ...(sentences, outer_folds, split_seed + 1000)
    assignments = stratify(sentences, n_folds, split_seed + 1000)
    return {idx + 1: fold for idx, fold in enumerate(assignments)}


_SEED_SUFFIX_RE = re.compile(r"__seed-?\d+$")
_CONDITION_PREFIXES = ("exp07aug_", "exp07_", "exp08_")


def base_condition(condition: str) -> str:
    """Condition key without the training-seed suffix the runner appends."""
    return _SEED_SUFFIX_RE.sub("", str(condition))


def split_stems_for_condition(condition: str) -> list[str]:
    """Candidate split-JSON stems for a runner condition key, most specific first.

    The runner reports split_condition as f"{base_condition_key}__seed{seed}" where the
    base key prefixes the source (``exp07_after_label_aware_split``), but the split JSONs
    on disk are named after the bare variant (``after_label_aware_split_train.json``).
    """
    base = _SEED_SUFFIX_RE.sub("", condition)
    stems = [s for s in (condition, base) if s]
    for prefix in _CONDITION_PREFIXES:
        if base.startswith(prefix):
            stems.append(base[len(prefix):])
            break
    seen: set[str] = set()
    return [s for s in stems if s and not (s in seen or seen.add(s))]


def fold_map_for_run(
    meta: dict,
    args: argparse.Namespace,
) -> dict[int, int] | None:
    """Resolve the split JSONs for this run's condition, then rebuild its fold map."""
    condition = str(meta.get("split_condition", "")).strip()
    n_folds = args.outer_folds or int(meta.get("outer_folds", 5) or 5)
    try:
        seed = args.split_seed if args.split_seed is not None else int(meta.get("seed", 42))
    except (TypeError, ValueError):
        seed = 42

    if args.corpus_train_json and args.corpus_eval_json:
        train_json, eval_json = args.corpus_train_json, args.corpus_eval_json
    elif args.splits_dir:
        stems = split_stems_for_condition(condition)
        pairs = [
            (args.splits_dir / f"{stem}_train.json", args.splits_dir / f"{stem}_eval.json")
            for stem in stems
        ]
        found = next(((t, e) for t, e in pairs if t.is_file() and e.is_file()), None)
        if found is None:
            tried = ", ".join(t.name for t, _ in pairs)
            print(f"  ! no split JSONs for condition '{condition}' in {args.splits_dir} "
                  f"(tried: {tried})")
            return None
        train_json, eval_json = found
    else:
        return None

    return reconstruct_fold_map(train_json, eval_json, n_folds, seed)


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_tables(runs: list[dict], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    main_rows: list[dict] = []
    delta_rows: list[dict] = []
    fold_rows: list[dict] = []
    diag_rows: list[dict] = []
    error_rows: list[dict] = []
    fold_map_cache: dict[tuple, dict[int, int] | None] = {}
    classify = _load_error_classifier()
    # The four base methods are byte-identical across the five router workbooks of one
    # (model, condition, seed) because every router reuses the same cached Exp01/Exp04
    # predictions. Count them once so the sheet is not 5x inflated.
    base_errors_done: set[tuple[str, str, str]] = set()

    for run_idx, run in enumerate(runs, start=1):
        meta = run["meta"]
        path = run["path"]
        if run_idx % 25 == 0 or run_idx == 1 or run_idx == len(runs):
            print(f"  scoring run {run_idx}/{len(runs)}", flush=True)
        detailed = add_derived_predictions(run["detailed"])

        model = str(meta.get("model", "unknown"))
        condition = str(meta.get("split_condition", "unknown"))
        seed = str(meta.get("seed", "unknown"))
        router = str(meta.get("router", path.stem))
        outer_folds = int(meta.get("outer_folds", 5) or 5)
        inner_folds = int(meta.get("inner_folds", 4) or 4)

        router_label = f"{router} fusion (Exp06 OOF)"
        methods = list(BASE_METHODS) + [(router_label, "fused_pred_label")]

        n_sentences = int(detailed["sentence_id"].nunique())
        n_tokens = int(len(detailed))
        scores: dict[str, float] = {}

        for label, pred_col in methods:
            f1, precision, recall = _score(detailed, pred_col)
            scores[label] = f1
            main_rows.append({
                "model": model,
                "split_condition": condition,
                "seed": seed,
                "router": router,
                "method": label,
                "is_fusion_router": pred_col == "fused_pred_label",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "eval_sentences": n_sentences,
                "eval_tokens": n_tokens,
                "outer_folds": outer_folds,
                "inner_folds": inner_folds,
                "protocol": "nested_stratified_cv_oof",
                "source_file": path.name,
            })

        fused_f1 = scores[router_label]
        for label, _ in BASE_METHODS:
            delta_rows.append({
                "model": model,
                "split_condition": condition,
                "seed": seed,
                "router": router,
                "baseline_method": label,
                "baseline_f1": scores[label],
                "fusion_f1": fused_f1,
                "delta_f1": fused_f1 - scores[label],
                "same_tokens": True,
                "protocol": "nested_stratified_cv_oof",
            })

        disagree_tokens = int(detailed["disagree"].sum())
        diag_rows.append({
            "model": model,
            "split_condition": condition,
            "seed": seed,
            "router": router,
            "eval_tokens": n_tokens,
            "disagreement_tokens": disagree_tokens,
            "disagreement_rate": disagree_tokens / n_tokens if n_tokens else float("nan"),
            "agreement_rate": 1 - (disagree_tokens / n_tokens) if n_tokens else float("nan"),
            "fold_metrics_rows": int(len(run["fold_metrics"])),
        })

        if classify is not None:
            base_key = (model, condition, seed)
            first_run_for_base = base_key not in base_errors_done
            base_errors_done.add(base_key)
            entity_tokens = int((detailed["true_label"].astype(str) != "O").sum())
            for label, pred_col in methods:
                is_router = pred_col == "fused_pred_label"
                if not is_router and not first_run_for_base:
                    continue
                counts = _error_counts(detailed, pred_col, classify)
                errors_total = n_tokens - counts["correct"]
                error_rows.append({
                    "model": model,
                    "split_condition": condition,
                    "base_split_condition": base_condition(condition),
                    "seed": seed,
                    "router": router if is_router else "(shared by all routers)",
                    "method": label,
                    "is_fusion_router": is_router,
                    "eval_tokens": n_tokens,
                    "entity_tokens": entity_tokens,
                    **{name: counts[name] for name in ERROR_CATEGORIES},
                    "errors_total": errors_total,
                    "token_error_rate": errors_total / n_tokens if n_tokens else float("nan"),
                    "source_file": path.name,
                })

        cache_key = (condition, seed, outer_folds)
        if cache_key not in fold_map_cache:
            fold_map_cache[cache_key] = fold_map_for_run(meta, args)
            if fold_map_cache[cache_key]:
                print(f"  folds reconstructed for condition={condition} seed={seed}: "
                      f"{len(fold_map_cache[cache_key])} sentences")
        fold_map = fold_map_cache[cache_key]

        if fold_map is not None:
            folded = detailed.copy()
            folded["outer_fold"] = folded["sentence_id"].map(fold_map)
            unmapped = int(folded["outer_fold"].isna().sum())
            if unmapped:
                print(f"  ! {unmapped} tokens had no fold assignment in {path.name};"
                      " check that the corpus JSONs match this run's condition")
            for fold, chunk in folded.dropna(subset=["outer_fold"]).groupby("outer_fold"):
                for label, pred_col in methods:
                    f1, precision, recall = _score(chunk, pred_col)
                    fold_rows.append({
                        "model": model,
                        "split_condition": condition,
                        "seed": seed,
                        "router": router,
                        "outer_fold": int(fold),
                        "method": label,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "fold_sentences": int(chunk["sentence_id"].nunique()),
                        "fold_tokens": int(len(chunk)),
                    })

    tables = {
        "unified_main_table": pd.DataFrame(main_rows),
        "fusion_vs_base": pd.DataFrame(delta_rows),
        "router_diagnostics": pd.DataFrame(diag_rows),
        "per_fold_f1": pd.DataFrame(fold_rows),
        "token_error_breakdown": pd.DataFrame(error_rows),
    }
    tables["paired_fold_tests"] = build_paired_fold_tests(tables["per_fold_f1"])
    tables["token_error_summary"] = summarise_error_breakdown(tables["token_error_breakdown"])
    return tables


def summarise_error_breakdown(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Sum the token error categories over every training seed of one condition."""
    if breakdown.empty:
        return pd.DataFrame()

    group_cols = ["model", "base_split_condition", "method"]
    count_cols = ["eval_tokens", "entity_tokens", *ERROR_CATEGORIES, "errors_total"]
    rows: list[dict] = []
    for keys, group in breakdown.groupby(group_cols, sort=True):
        totals = {
            col: int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            for col in count_cols
        }
        tokens = totals["eval_tokens"]
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_seeds": int(group["seed"].nunique()),
            "n_runs": int(len(group)),
            **totals,
            "token_error_rate": totals["errors_total"] / tokens if tokens else float("nan"),
            **{
                f"{name}_pct_of_tokens": (totals[name] / tokens if tokens else float("nan"))
                for name in ERROR_CATEGORIES
            },
        })
    return pd.DataFrame(rows)


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment. NaN entries are left out of the family."""
    adjusted = np.full(len(pvals), np.nan, dtype=float)
    valid = np.flatnonzero(~np.isnan(pvals))
    if valid.size == 0:
        return adjusted

    m = valid.size
    order = valid[np.argsort(pvals[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(pvals[idx]))
        adjusted[idx] = min(running, 1.0)
    return adjusted


def apply_holm_within_families(tests: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust across every contrast sharing a (model, split_condition)."""
    if tests.empty:
        return tests

    out = tests.copy()
    family_cols = ["model", "split_condition"]
    out["holm_family"] = out[family_cols].astype(str).agg(" | ".join, axis=1)

    for raw_col, adj_col in (("wilcoxon_p", "wilcoxon_p_holm"), ("ttest_rel_p", "ttest_rel_p_holm")):
        out[adj_col] = np.nan
        for family, group in out.groupby("holm_family"):
            values = pd.to_numeric(group[raw_col], errors="coerce").to_numpy(dtype=float)
            out.loc[group.index, adj_col] = _holm_adjust(values)

    out["holm_family_size"] = out.groupby("holm_family")["wilcoxon_p"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").notna().sum()
    )
    out["sig_05_holm"] = out["wilcoxon_p_holm"] < 0.05
    out["sig_01_holm"] = out["wilcoxon_p_holm"] < 0.01
    return out


def build_paired_fold_tests(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Pair by (training_seed, outer_fold), matching paired_fold_method_comparison().

    With S training seeds and 5 outer folds each contrast has up to S x 5 paired
    observations, so seed count drives the power here rather than the 5 folds alone.
    """
    if per_fold.empty:
        return pd.DataFrame()
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception as exc:
        print(f"  ! scipy unavailable, paired fold tests skipped ({exc})")
        return pd.DataFrame()

    floor_cache: dict[int, float] = {}

    def _p_floor(n: int) -> float:
        """Smallest two-sided p this n can produce (all differences same sign)."""
        if n < 1:
            return float("nan")
        if n not in floor_cache:
            try:
                floor_cache[n] = float(wilcoxon(np.arange(1, n + 1, dtype=float)).pvalue)
            except Exception:
                floor_cache[n] = float("nan")
        return floor_cache[n]

    rows: list[dict] = []
    group_cols = ["model", "split_condition", "router"]

    for keys, group in per_fold.groupby(group_cols):
        wide = group.pivot_table(index=["seed", "outer_fold"], columns="method", values="f1")
        router_cols = [c for c in wide.columns if "Exp06 OOF" in c]
        if not router_cols:
            continue
        fusion_col = router_cols[0]

        for label, _ in BASE_METHODS:
            if label not in wide.columns:
                continue
            paired = wide[[fusion_col, label]].dropna()
            if paired.empty:
                continue
            diffs = (paired[fusion_col] - paired[label]).to_numpy(dtype=float)
            n = len(diffs)

            wilcoxon_p = float("nan")
            ttest_p = float("nan")
            if n >= 2 and np.any(diffs != 0):
                try:
                    wilcoxon_p = float(wilcoxon(diffs, alternative="two-sided").pvalue)
                except ValueError:
                    pass
                try:
                    ttest_p = float(ttest_rel(paired[fusion_col], paired[label]).pvalue)
                except ValueError:
                    pass

            rows.append({
                **dict(zip(group_cols, keys)),
                "contrast": f"{fusion_col} - {label}",
                "pairing": "training_seed x outer_fold",
                "n_seeds": int(paired.index.get_level_values("seed").nunique()),
                "n_folds": int(paired.index.get_level_values("outer_fold").nunique()),
                "n_pairs": n,
                "mean_delta_f1": float(np.mean(diffs)),
                "sd_delta_f1": float(np.std(diffs, ddof=1)) if n > 1 else float("nan"),
                "pairs_favouring_fusion": int(np.sum(diffs > 0)),
                "wilcoxon_p": wilcoxon_p,
                "ttest_rel_p": ttest_p,
                "min_attainable_wilcoxon_p": _p_floor(n),
            })

    return apply_holm_within_families(pd.DataFrame(rows))


def summarise_across_folds(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD over all (training_seed, outer_fold) units, as summarize_oof_for_paper does."""
    if per_fold.empty:
        return pd.DataFrame()

    group_cols = ["model", "split_condition", "router", "method"]
    rows: list[dict] = []
    for keys, group in per_fold.groupby(group_cols):
        f1 = pd.to_numeric(group["f1"], errors="coerce").dropna()
        sd = float(f1.std(ddof=1)) if len(f1) > 1 else 0.0
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_seeds": int(group["seed"].nunique()),
            "n_folds": int(group["outer_fold"].nunique()),
            "n_units": int(len(f1)),
            "mean_f1": float(f1.mean()) if len(f1) else float("nan"),
            "sd_f1": sd,
            "min_f1": float(f1.min()) if len(f1) else float("nan"),
            "max_f1": float(f1.max()) if len(f1) else float("nan"),
            "f1_mean_pm_sd": f"{f1.mean():.3f}±{sd:.3f}" if len(f1) else "",
        })
    return pd.DataFrame(rows)


def build_readme() -> pd.DataFrame:
    rows = [
        ("Purpose", "Why this file exists",
         "Scores every method on the SAME nested-CV outer-test tokens so fusion can be "
         "compared with the base models under one evaluation protocol."),
        ("Protocol", "Name", "nested_stratified_cv_oof"),
        ("Protocol", "Outer loop",
         "5 sentence-level multilabel-stratified folds; each sentence is in exactly one outer test."),
        ("Protocol", "Inner loop",
         "4 folds on outer-train produce out-of-fold Exp01/Exp04 predictions used to fit the router."),
        ("Protocol", "Router isolation",
         "Outer-test sentences are never used for NER training, inner OOF, or router fitting."),
        ("Protocol", "Base model training size",
         "~80% of the corpus per outer fold (approx. 120 of 150 sentences)."),
        ("Protocol", "Scored sentences", "All corpus sentences, pooled over outer tests."),
        ("Metric", "Definition",
         "Strict entity-level micro F1 (seqeval), exact span and component-type match."),
        ("Derivation", "Direct NER / cascade",
         "Read directly from regular_pred_label / cascade_pred_label in the OOF detailed_results sheet."),
        ("Derivation", "Cascade + repair",
         "Exp05 B/I type consistency re-applied to cascade_bio / cascade_etype using bio_prob."),
        ("Derivation", "Confidence fusion",
         "Agreements pass through; disagreements take the higher-confidence source, ties to direct NER."),
        ("Derivation", "Router fusion", "fused_pred_label as produced by the frozen per-fold router."),
        ("Derivation", "Retraining", "None. All methods are re-scored from cached OOF predictions."),
        ("Caveat", "Comparison with the 70/30 holdout tables",
         "Not directly comparable: holdout trains on ~105 sentences and scores 45, this protocol "
         "trains on ~120 and scores all 150. Report the two in separate tables."),
        ("Caveat", "Significance testing",
         "Pairing is by (training_seed, outer_fold), so S seeds give S x 5 paired observations. "
         "With a single seed n=5 and the two-sided signed-rank floor is p=0.0625; run more seeds "
         "to reach alpha=0.01."),
        ("Gap closed", "journal_paired_fold_deltas",
         "collect_oof_fold_long() keeps only experiment ids containing '_oof', so exp01 and exp04 "
         "never enter fold_long and the fusion-vs-baseline contrasts are dropped. This file supplies "
         "the missing per-fold base-model F1."),
        ("Sheets", "unified_main_table", "One row per run and method: pooled outer-test F1/P/R."),
        ("Sheets", "fold_summary", "Mean±SD F1 over all (seed, fold) units per method."),
        ("Sheets", "fusion_vs_base", "Fusion minus each baseline on identical tokens."),
        ("Sheets", "per_fold_f1", "Per-(seed, outer_fold) F1 for every method (paired units)."),
        ("Sheets", "paired_fold_tests",
         "Paired deltas with Wilcoxon and paired t-test p-values, Holm-adjusted within each "
         "(model, split_condition) family."),
        ("Statistics", "Dependency caveat",
         "Outer folds within one partition share training data, so paired CV tests are liberal "
         "rather than conservative (Dietterich 1998; Bengio & Grandvalet 2004). Because "
         "THESIS_SPLIT_SEED drives fold assignment, each training seed yields a different 5-fold "
         "partition, so the seed dimension is repeated CV rather than reruns of one partition."),
        ("Sheets", "token_error_summary",
         "Token error counts summed over all training seeds per (model, condition, method): "
         "false_positive, false_negative, type_error, boundary_error, correct, eval_tokens."),
        ("Sheets", "token_error_breakdown",
         "Same counts per individual run, so spread across seeds can be computed. Base-method "
         "rows appear once per (model, condition, seed) with router='(shared by all routers)' "
         "because all five routers reuse the same cached Exp01/Exp04 predictions."),
        ("Metric", "Token error taxonomy",
         "classify_error() from experiments/error_analysis.py, so these counts match the "
         "error_type_summary sheets of the individual experiments: false_positive = spurious "
         "entity on a true-O token, false_negative = real entity token predicted O, type_error = "
         "entity detected but wrong component type, boundary_error = right type but wrong B/I."),
        ("Caveat", "Token counts vs entity F1",
         "The error breakdown is token-level (BIO tags) while F1 is entity-level (exact span and "
         "type). They answer different questions and will not reconcile arithmetically."),
        ("Sheets", "router_diagnostics", "Agreement/disagreement token rates per run."),
        ("Sheets", "journal_* / paired_tests",
         "Copied verbatim from the runner's cross_comparison_latest.xlsx so this workbook is "
         "self-contained: lambda grid and loss config for Methods, journal_main_table for the "
         "seed-based holdout numbers, and the runner's own fold sheets for cross-checking."),
        ("Reading order", "Which sheet for what",
         "Main table cells -> fold_summary. Headline gain -> fusion_vs_base. p-values -> "
         "paired_fold_tests. Error analysis -> token_error_summary. Methods lambda -> "
         "journal_lambda_grid. Holdout comparison -> journal_main_table."),
    ]
    return pd.DataFrame(rows, columns=["Section", "Key", "Value"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof-dir", action="append", default=[], type=Path,
                        help="Directory searched recursively for OOF result files (repeatable).")
    parser.add_argument("--oof-xlsx", action="append", default=[], type=Path,
                        help="Explicit OOF result file (repeatable).")
    parser.add_argument("--pattern", default="*oof*.xlsx",
                        help="Glob used inside --oof-dir (default: %(default)s).")
    parser.add_argument("--splits-dir", type=Path,
                        help="Exp07 splits directory (e.g. <output-dir>/exp07/splits). Enables the "
                             "per-fold sheets and picks <split_condition>_train/eval.json per run. "
                             "Prefer this over the two --corpus-*-json flags when more than one "
                             "split condition was run, since fold membership is condition-specific.")
    parser.add_argument("--corpus-train-json", type=Path,
                        help="Single Exp07 train split JSON; use only for a one-condition run.")
    parser.add_argument("--corpus-eval-json", type=Path,
                        help="Single Exp07 eval split JSON; use only for a one-condition run.")
    parser.add_argument("--outer-folds", type=int, default=None,
                        help="Override the outer fold count used for fold reconstruction.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Override THESIS_SPLIT_SEED used for fold reconstruction.")
    parser.add_argument("--reference-xlsx", "--holdout-xlsx", dest="reference_xlsx", type=Path,
                        help="The runner's cross_comparison_latest.xlsx. Its paper sheets are copied "
                             "in verbatim so this workbook is the single file you write from.")
    parser.add_argument("--reference-sheets", default=",".join(REFERENCE_SHEETS),
                        help="Comma-separated sheets to carry over from --reference-xlsx "
                             "(default: %(default)s).")
    parser.add_argument("--output", type=Path, default=Path("outputs/unified_protocol_table.xlsx"),
                        help="Destination workbook (default: %(default)s).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.oof_dir and not args.oof_xlsx:
        args.oof_dir = [Path("outputs")]

    print("Discovering OOF result files...", flush=True)
    candidates = discover_oof_files(args.oof_dir, args.oof_xlsx, args.pattern)
    print(f"  {len(candidates)} candidate workbook(s)", flush=True)

    print(f"Reading {len(candidates)} workbook(s) (slow over a Drive mount)...")
    runs = []
    for idx, path in enumerate(candidates, start=1):
        run = read_oof_file(path)
        if run is not None:
            runs.append(run)
        if idx % 25 == 0 or idx == len(candidates):
            print(f"  read {idx}/{len(candidates)} ({len(runs)} usable)", flush=True)

    if not runs:
        print("\nNo OOF fusion results found. Point --oof-dir at the cross-comparison output "
              "directory (the one containing the 06_*_oof result files).")
        return 1

    runs = dedupe_runs(runs)
    print(f"  {len(runs)} unique (model, condition, seed, router) run(s) after de-duplication")
    if len(runs) <= 12:
        for run in runs:
            key = _run_key(run["meta"], run["path"])
            print(f"    - model={key[0]} condition={key[1]} seed={key[2]} router={key[3]}")
    else:
        keys = [_run_key(r["meta"], r["path"]) for r in runs]
        print(f"    models={sorted({k[0] for k in keys})}")
        print(f"    conditions={sorted({k[1] for k in keys})}")
        print(f"    seeds={len({k[2] for k in keys})}, routers={sorted({k[3] for k in keys})}")

    if args.splits_dir:
        print(f"\nResolving split JSONs per condition from {args.splits_dir}")
    elif args.corpus_train_json and args.corpus_eval_json:
        conditions = {str(r["meta"].get("split_condition", "")) for r in runs}
        if len(conditions) > 1:
            print(f"\n  ! {len(conditions)} split conditions present but a single pair of corpus "
                  "JSONs was given. Fold membership is condition-specific — pass --splits-dir "
                  "instead, or the per-fold sheets will be wrong.")
        print("\nUsing the supplied corpus JSONs for fold reconstruction")
    else:
        print("\nNo --splits-dir or corpus JSONs supplied; per-fold sheets will be empty.")

    print("\nScoring every method on the pooled outer-test tokens...")
    tables = build_tables(runs, args)
    tables["fold_summary"] = summarise_across_folds(tables["per_fold_f1"])
    tables["readme"] = build_readme()

    carried: list[str] = []
    if args.reference_xlsx:
        wanted = [s.strip() for s in args.reference_sheets.split(",") if s.strip()]
        try:
            available = pd.read_excel(args.reference_xlsx, sheet_name=None)
        except Exception as exc:
            available = {}
            print(f"  ! could not read {args.reference_xlsx} ({exc})")
        for name in wanted:
            frame = available.get(name)
            if frame is None or frame.empty:
                print(f"  - reference sheet '{name}' absent or empty; skipped")
                continue
            tables[name] = frame
            carried.append(name)
        if carried:
            print(f"  carried {len(carried)} sheet(s) from {args.reference_xlsx.name}: "
                  f"{', '.join(carried)}")

    sheet_order = [
        "readme",
        "unified_main_table",
        "fold_summary",
        "fusion_vs_base",
        "paired_fold_tests",
        "per_fold_f1",
        "token_error_summary",
        "token_error_breakdown",
        "router_diagnostics",
    ] + carried

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for name in sheet_order:
            frame = tables.get(name)
            if frame is None or frame.empty:
                continue
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\nWrote {args.output}")
    for name in sheet_order:
        frame = tables.get(name)
        if frame is not None and not frame.empty:
            print(f"  {name}: {len(frame)} row(s)")

    main_table = tables["unified_main_table"]
    if not main_table.empty:
        print("\nPooled outer-test F1 (one protocol, identical tokens):")
        preview = main_table[["model", "split_condition", "method", "f1", "eval_sentences"]]
        print(preview.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    deltas = tables["fusion_vs_base"]
    if not deltas.empty:
        print("\nFusion minus baseline on identical tokens:")
        preview = deltas[["model", "split_condition", "baseline_method", "baseline_f1",
                          "fusion_f1", "delta_f1"]]
        print(preview.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
