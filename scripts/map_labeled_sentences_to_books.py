#!/usr/bin/env python3
"""Map each labeled NER sentence to source ``.txt`` files under ``data/books``.

Labeled tokens often expand Hebrew abbreviations (e.g. ``אבן העזר``) while book
text keeps gershayim forms (e.g. ``אה\"ע``). Matching uses *anchor* words:
non-abbreviation tokens that must appear in order in the source file. Label
anchors that do not occur in a file (typical expansion pieces) are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from hebrew_text_io import read_ner_dataset_csv  # noqa: E402

# Labeled token data (sheet ``ner_dataset``).
DEFAULT_DATASET_XLSX = (
    Path(__file__).resolve().parents[1] / "data" / "ner_dataset.xlsx"
)
DEFAULT_XLSX_SHEET = "ner_dataset"

HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")
NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")
STRUCTURAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "[MASK]"})
ABBREV_INNER_RE = re.compile(r'[א-ת]["\u05f3\u05f4\'][א-ת]')
READ_ENCODINGS = ("utf-8-sig", "utf-8", "cp1255", "windows-1255")


def _hebrew_letter_count(text: str) -> int:
    return len(HEBREW_RE.findall(text))


def normalize_token(token: str) -> str:
    t = NIQQUD_RE.sub("", str(token).strip())
    t = t.replace("\u05f4", '"').replace("\u05f3", "'")
    t = t.strip(".,;:!?()[]{}«»""''\u201c\u201d/\\|")
    return t


def is_abbreviation_token(token: str) -> bool:
    """Hebrew abbreviation marker (gershayim) or very short quoted form."""
    t = normalize_token(token)
    if not t:
        return False
    if '"' in t or "'" in t:
        return True
    if ABBREV_INNER_RE.search(t):
        return True
    return False


def is_anchor_token(token: str, *, min_hebrew_letters: int = 2) -> bool:
    t = normalize_token(token)
    if not t or t in STRUCTURAL_TOKENS:
        return False
    if is_abbreviation_token(t):
        return False
    if _hebrew_letter_count(t) < min_hebrew_letters:
        return False
    return True


def tokenize_text(text: str) -> list[str]:
    """Split running text into normalized match tokens."""
    cleaned = NIQQUD_RE.sub("", text)
    cleaned = cleaned.replace("\n", " ")
    raw_parts = re.split(r"\s+", cleaned)
    tokens: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        # Further split on punctuation that often sticks to words.
        for piece in re.split(r"(?<=[,.;:!?])|(?=[,.;:!?])", part):
            piece = piece.strip()
            if not piece or piece in ".,;:!?":
                continue
            nt = normalize_token(piece)
            if nt:
                tokens.append(nt)
    return tokens


def merge_wordpieces(tokens: list[str]) -> list[str]:
    merged: list[str] = []
    for tok in tokens:
        if tok.startswith("##") and merged:
            merged[-1] += tok[2:]
        else:
            merged.append(tok)
    return merged


def load_book_text(path: Path) -> str:
    raw = path.read_bytes()
    text: str | None = None
    for enc in READ_ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, dict) and "text" in first:
                    return str(first["text"])
            if isinstance(payload, dict) and "text" in payload:
                return str(payload["text"])
        except json.JSONDecodeError:
            pass
    return text


@dataclass
class BookRecord:
    rel_path: str
    raw_text: str
    tokens: list[str]
    token_set: frozenset[str]
    text_flat: str


def build_book_index(books_dir: Path) -> tuple[list[BookRecord], dict[str, list[int]]]:
    """Load all ``*.txt`` under *books_dir* and invert rare tokens for lookup."""
    records: list[BookRecord] = []
    inverted: dict[str, list[int]] = defaultdict(list)

    txt_paths = sorted(books_dir.rglob("*.txt"))
    for path in txt_paths:
        try:
            body = load_book_text(path)
        except OSError:
            continue
        tokens = tokenize_text(body)
        if not tokens:
            continue
        rel = path.relative_to(books_dir).as_posix()
        flat = " ".join(tokens)
        rec = BookRecord(
            rel_path=rel,
            raw_text=body,
            tokens=tokens,
            token_set=frozenset(tokens),
            text_flat=flat,
        )
        idx = len(records)
        records.append(rec)
        seen_local: set[str] = set()
        for tok in tokens:
            if tok in seen_local:
                continue
            seen_local.add(tok)
            inverted[tok].append(idx)

    return records, inverted


def extract_label_anchors(tokens: list[str]) -> list[str]:
    anchors: list[str] = []
    for t in tokens:
        nt = normalize_token(t)
        if not nt or nt in {"(", ")", "[", "]"}:
            continue
        if is_anchor_token(t):
            anchors.append(nt)
    return anchors


def greedy_ordered_anchor_hits(anchors: list[str], book: BookRecord) -> int:
    """Count how many label anchors match in order (skip anchors absent from the file)."""
    return find_match_token_span(anchors, book)[2]


def find_match_token_span(
    anchors: list[str], book: BookRecord
) -> tuple[int | None, int | None, int]:
    """Return ``(start_idx, end_exclusive, matched_anchor_count)`` in *book.tokens*."""
    hi = 0
    first: int | None = None
    last: int | None = None
    matched = 0
    for needle in anchors:
        if needle not in book.token_set:
            continue
        while hi < len(book.tokens):
            if book.tokens[hi] == needle:
                if first is None:
                    first = hi
                last = hi
                matched += 1
                hi += 1
                break
            hi += 1
        else:
            break
    if first is None or last is None:
        return None, None, matched
    return first, last + 1, matched


def split_source_sentences(text: str) -> list[str]:
    """Rough sentence/paragraph splits for responsa plain text."""
    text = text.replace("\r\n", "\n")
    chunks = re.split(r"(?<=[.:!?])\s+|\n\s*\n+", text)
    return [c.strip() for c in chunks if c and c.strip()]


def _best_sentence_index(sentences: list[str], match_tokens: list[str]) -> int | None:
    if not sentences or not match_tokens:
        return None
    match_set = set(match_tokens)
    best_idx: int | None = None
    best_score = 0
    for i, sent in enumerate(sentences):
        sent_tokens = tokenize_text(sent)
        score = sum(1 for t in match_tokens if t in sent_tokens)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score < max(2, min(3, len(match_tokens))):
        return None
    return best_idx


def format_match_context(
    book: BookRecord,
    label_anchors: list[str],
    labeled_sentence: str,
    *,
    sentences_before: int = 1,
    sentences_after: int = 1,
) -> str:
    """Context line: prior/next source sentences plus labeled sentence in ``**``."""
    start, end, matched = find_match_token_span(label_anchors, book)
    if start is None or end is None or matched < 1:
        return f"{book.rel_path}: **{labeled_sentence}**"

    match_tokens = book.tokens[start:end]
    source_sents = split_source_sentences(book.raw_text)
    sent_idx = _best_sentence_index(source_sents, match_tokens)

    highlight = labeled_sentence.strip()
    if sent_idx is not None:
        before_parts = source_sents[max(0, sent_idx - sentences_before) : sent_idx]
        after_parts = source_sents[sent_idx + 1 : sent_idx + 1 + sentences_after]
        chunks: list[str] = []
        chunks.extend(before_parts)
        chunks.append(f"**{highlight}**")
        chunks.extend(after_parts)
        body = " ".join(chunks)
        return f"{book.rel_path}: {body}"

    # Token-window fallback when sentence boundaries are unclear.
    before = " ".join(book.tokens[max(0, start - 28) : start])
    after = " ".join(book.tokens[end : end + 28])
    parts = [p for p in (before, f"**{highlight}**", after) if p]
    return f"{book.rel_path}: {' '.join(parts)}"


def build_context_column(
    matched_files: list[str],
    books_by_path: dict[str, BookRecord],
    label_anchors: list[str],
    labeled_sentence: str,
    *,
    max_files: int = 5,
) -> str:
    blocks: list[str] = []
    for rel_path in matched_files[:max_files]:
        book = books_by_path.get(rel_path)
        if book is None:
            blocks.append(f"{rel_path}: **{labeled_sentence.strip()}**")
            continue
        blocks.append(
            format_match_context(book, label_anchors, labeled_sentence)
        )
    return "\n---\n".join(blocks)


def anchor_windows(anchors: list[str], *, min_len: int = 2) -> list[list[str]]:
    """Partial-sentence anchor runs, longest first (full sentence first)."""
    if not anchors:
        return []
    n = len(anchors)
    windows: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for size in range(n, min_len - 1, -1):
        for start in range(0, n - size + 1):
            chunk = tuple(anchors[start : start + size])
            if chunk in seen:
                continue
            seen.add(chunk)
            windows.append(list(chunk))
    return windows


def ordered_subsequence_match(anchors: list[str], haystack: list[str]) -> bool:
    hi = 0
    for needle in anchors:
        while hi < len(haystack):
            if haystack[hi] == needle:
                hi += 1
                break
            hi += 1
        else:
            return False
    return True


def score_match(
    label_anchors: list[str],
    book: BookRecord,
    *,
    min_ratio: float,
    min_matched: int,
    strict_all_relevant: bool = False,
) -> tuple[float, int, int] | None:
    """Return (matched_ratio, matched_count, relevant_count) or None if no match."""
    if not label_anchors:
        return None

    relevant = [a for a in label_anchors if a in book.token_set]
    if not relevant:
        return None

    if strict_all_relevant:
        if len(relevant) < min_matched:
            return None
        if not ordered_subsequence_match(relevant, book.tokens):
            return None
        matched_count = len(relevant)
    else:
        matched_count = greedy_ordered_anchor_hits(label_anchors, book)
        if matched_count < min_matched:
            return None

    ratio = matched_count / len(label_anchors)
    if ratio < min_ratio:
        return None

    return ratio, matched_count, len(relevant)


def score_match_strict(label_anchors: list[str], book: BookRecord) -> tuple[float, int, int] | None:
    if not label_anchors:
        return None
    min_hits = min(3, len(label_anchors))
    if len(label_anchors) <= 4:
        min_hits = max(2, len(label_anchors) - 1)
    min_ratio = 0.45 if len(label_anchors) >= 6 else 0.55
    return score_match(
        label_anchors,
        book,
        min_ratio=min_ratio,
        min_matched=min_hits,
        strict_all_relevant=True,
    )


def score_match_relaxed(label_anchors: list[str], book: BookRecord) -> tuple[float, int, int] | None:
    if not label_anchors:
        return None
    min_matched = max(2, min(4, len(label_anchors) // 3))
    if len(label_anchors) <= 3:
        min_matched = max(2, len(label_anchors))
    min_ratio = 0.22 if len(label_anchors) >= 8 else 0.28
    return score_match(
        label_anchors,
        book,
        min_ratio=min_ratio,
        min_matched=min_matched,
        strict_all_relevant=False,
    )


def substring_phrase_match(label_anchors: list[str], book: BookRecord) -> tuple[float, int, int] | None:
    """Last resort: contiguous anchor phrase appears in file token stream."""
    if len(label_anchors) < 2:
        return None
    for size in range(min(8, len(label_anchors)), 1, -1):
        for start in range(0, len(label_anchors) - size + 1):
            phrase_tokens = label_anchors[start : start + size]
            phrase = " ".join(phrase_tokens)
            if len(phrase) < 8:
                continue
            if phrase in book.text_flat:
                ratio = size / len(label_anchors)
                return ratio, size, size
    return None


def candidate_book_indices(
    label_anchors: list[str],
    inverted: dict[str, list[int]],
    n_books: int,
) -> list[int] | None:
    """Pick candidate books via rare anchor words; ``None`` means search all."""
    usable = [a for a in label_anchors if a in inverted]
    if not usable:
        return None

    usable.sort(key=lambda w: len(inverted[w]))
    seed_words = usable[: min(4, len(usable))]
    candidate_set: set[int] | None = None
    for word in seed_words:
        postings = set(inverted[word])
        if not postings:
            continue
        candidate_set = postings if candidate_set is None else candidate_set & postings
        if candidate_set is not None and len(candidate_set) <= max(50, n_books // 200):
            break

    if candidate_set is not None and len(candidate_set) == 0 and seed_words:
        candidate_set = set(inverted[seed_words[0]])

    if not candidate_set:
        return None
    if len(candidate_set) > 800:
        return sorted(candidate_set)
    return sorted(candidate_set)


def _collect_hits(
    indices: list[int] | range,
    books: list[BookRecord],
    label_anchors: list[str],
    scorer,
) -> list[tuple[str, float, int, int]]:
    hits: list[tuple[str, float, int, int]] = []
    for idx in indices:
        scored = scorer(label_anchors, books[idx])
        if scored is None:
            continue
        ratio, matched, relevant = scored
        hits.append((books[idx].rel_path, ratio, matched, relevant))
    hits.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return hits


def _best_ties(hits: list[tuple[str, float, int, int]]) -> list[tuple[str, float, int, int]]:
    if not hits:
        return []
    best_ratio = hits[0][1]
    best_matched = hits[0][2]
    return [h for h in hits if h[1] >= best_ratio - 1e-9 and h[2] >= best_matched]


def find_books_for_sentence(
    label_anchors: list[str],
    books: list[BookRecord],
    inverted: dict[str, list[int]],
) -> tuple[list[tuple[str, float, int, int]], str]:
    if not label_anchors:
        return [], "none"

    candidates = candidate_book_indices(label_anchors, inverted, len(books))
    index_lists: list[list[int] | range] = []
    if candidates is not None:
        index_lists.append(candidates)
    index_lists.append(range(len(books)))

    for indices in index_lists:
        hits = _collect_hits(indices, books, label_anchors, score_match_strict)
        if hits:
            return _best_ties(hits), "strict"

        hits = _collect_hits(indices, books, label_anchors, score_match_relaxed)
        if hits:
            return _best_ties(hits), "relaxed"

        for window in anchor_windows(label_anchors, min_len=2):
            if len(window) < 2:
                continue
            hits = _collect_hits(indices, books, window, score_match_relaxed)
            if hits:
                return _best_ties(hits), "partial_window"

        phrase_hits: list[tuple[str, float, int, int]] = []
        for idx in indices:
            scored = substring_phrase_match(label_anchors, books[idx])
            if scored is None:
                continue
            ratio, matched, relevant = scored
            phrase_hits.append((books[idx].rel_path, ratio, matched, relevant))
        phrase_hits.sort(key=lambda x: (-x[1], -x[2], x[0]))
        if phrase_hits:
            return _best_ties(phrase_hits), "substring"

        best: tuple[str, float, int, int] | None = None
        for idx in indices:
            matched = greedy_ordered_anchor_hits(label_anchors, books[idx])
            if matched < 2:
                continue
            ratio = matched / len(label_anchors)
            if ratio < 0.15:
                continue
            row = (books[idx].rel_path, ratio, matched, matched)
            if best is None or (row[2], row[1]) > (best[2], best[1]):
                best = row
        if best is not None:
            return [best], "best_effort"

    return [], "none"


def _sentences_from_token_df(df: pd.DataFrame) -> list[dict]:
    if "id" not in df.columns or "token" not in df.columns:
        raise ValueError("Dataset must have 'id' and 'token' columns.")

    if "raw_tags" not in df.columns and "ner_tags" in df.columns:
        df = df.copy()
        df["raw_tags"] = "O"

    df = df.dropna(subset=["token"])
    sentences: list[dict] = []

    for sent_id, group in df.groupby("id", sort=False):
        raw_tokens = [str(t) for t in group["token"]]
        tokens = merge_wordpieces(raw_tokens)
        content = [t for t in tokens if t.strip() and t not in STRUCTURAL_TOKENS]
        if not content:
            continue
        anchors = extract_label_anchors(content)
        try:
            sid_int = int(sent_id)
        except (TypeError, ValueError):
            sid_int = sent_id
        sentences.append(
            {
                "sentence_id": sid_int,
                "text": " ".join(content),
                "anchors": anchors,
            }
        )
    sentences.sort(key=lambda s: (isinstance(s["sentence_id"], str), s["sentence_id"]))
    return sentences


def load_labeled_dataset(path: Path, *, sheet_name: str) -> tuple[pd.DataFrame, str]:
    """Load token-level NER data from Excel (``ner_dataset`` sheet) or CSV."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
        return df, f"xlsx:{sheet_name}"
    df, encoding = read_ner_dataset_csv(path)
    return df, encoding


def sentences_from_labeled_dataset(path: Path, *, sheet_name: str) -> list[dict]:
    df, source = load_labeled_dataset(path, sheet_name=sheet_name)
    print(f"Loaded labeled data ({source}) — {df['id'].nunique()} sentence ids")
    return _sentences_from_token_df(df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_XLSX,
        help=f"Token-level labeled Excel or CSV (default: {DEFAULT_DATASET_XLSX})",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=DEFAULT_XLSX_SHEET,
        help=f"Excel sheet name (default: {DEFAULT_XLSX_SHEET!r})",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "books",
        help="Root directory of book subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sentence_book_sources.json",
        help="JSON output with full match details",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sentence_book_sources.csv",
        help="Flat table: id, sentence, file_name, file_count, file_context (first 5 files)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N sentences (0 = all)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if not args.books_dir.is_dir():
        print(f"Books directory not found: {args.books_dir}", file=sys.stderr)
        return 1

    print(f"Indexing books under {args.books_dir} ...")
    books, inverted = build_book_index(args.books_dir)
    books_by_path = {b.rel_path: b for b in books}
    print(f"Indexed {len(books)} text files.")

    sentences = sentences_from_labeled_dataset(dataset_path, sheet_name=args.sheet)
    if args.limit > 0:
        sentences = sentences[: args.limit]

    results = []
    unmatched = 0
    multi = 0
    no_anchors = sum(1 for s in sentences if not s["anchors"])
    if no_anchors:
        print(
            f"Warning: {no_anchors} sentences have no anchor tokens "
            f"(check CSV encoding / abbreviation-only lines)."
        )

    for i, sent in enumerate(sentences, start=1):
        matches, match_mode = find_books_for_sentence(sent["anchors"], books, inverted)
        files = [m[0] for m in matches]
        file_context = build_context_column(
            files,
            books_by_path,
            sent["anchors"],
            sent["text"],
        )
        if not files:
            unmatched += 1
        elif len(files) > 1:
            multi += 1

        results.append(
            {
                "sentence_id": sent["sentence_id"],
                "text": sent["text"],
                "anchor_count": len(sent["anchors"]),
                "match_mode": match_mode,
                "file_count": len(files),
                "matched_files": files,
                "file_context": file_context,
                "match_details": [
                    {
                        "file": path,
                        "anchor_match_ratio": ratio,
                        "matched_anchors": matched,
                        "relevant_anchors": relevant,
                        "context": (
                            format_match_context(
                                books_by_path[path], sent["anchors"], sent["text"]
                            )
                            if path in books_by_path
                            else ""
                        ),
                    }
                    for path, ratio, matched, relevant in matches[:5]
                ],
            }
        )
        if i % 50 == 0 or i == len(sentences):
            print(f"  mapped {i}/{len(sentences)} sentences ...")

    results.sort(key=lambda r: r["sentence_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(dataset_path),
        "sheet": args.sheet if dataset_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"} else None,
        "books_dir": str(args.books_dir),
        "sentences": len(results),
        "unmatched_sentences": unmatched,
        "sentences_with_multiple_files": multi,
        "indexed_book_files": len(books),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table_rows: list[dict[str, object]] = []
    for row in results:
        files = row["matched_files"]
        table_rows.append(
            {
                "id": row["sentence_id"],
                "sentence": str(row["text"]),
                "file_name": "; ".join(files) if files else "",
                "file_count": len(files),
                "file_context": row.get("file_context", ""),
            }
        )

    table_df = pd.DataFrame(
        table_rows,
        columns=["id", "sentence", "file_name", "file_count", "file_context"],
    )
    table_path = args.table_output
    if table_path.suffix.lower() in {".xlsx", ".xlsm"}:
        table_df.to_excel(table_path, index=False, sheet_name="sentence_sources")
    else:
        table_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    print(
        f"Wrote {args.output} and {table_path} — unmatched: {unmatched}, "
        f"multi-file: {multi}, total: {len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
