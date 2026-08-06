"""Hebrew NER corpus encoding detection and corruption guards."""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

HEBREW_CHAR_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")
CORRUPT_TOKEN_RE = re.compile(r"^\?+$")
STRUCTURAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "[MASK]"})

DEFAULT_CSV_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "cp1255",
    "windows-1255",
    "iso-8859-8",
    "iso-8859-8-i",
    "cp1252",
)


class HebrewCorpusEncodingError(RuntimeError):
    """Raised when text looks corrupted (e.g. all ``?``) or not Hebrew."""


def _skip_hebrew_validation() -> bool:
    return (os.environ.get("THESIS_SKIP_HEBREW_TEXT_VALIDATION") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def hebrew_char_count(text: str) -> int:
    return len(HEBREW_CHAR_RE.findall(str(text)))


def _is_content_token(token: str) -> bool:
    t = str(token).strip()
    if not t or t in STRUCTURAL_TOKENS:
        return False
    return True


def score_token_column_hebrew(df: pd.DataFrame, token_col: str = "token", sample_rows: int = 4000) -> int:
    if token_col not in df.columns:
        return 0
    series = df[token_col].dropna().astype(str)
    if series.empty:
        return 0
    if len(series) > sample_rows:
        series = series.sample(n=sample_rows, random_state=42)
    total = 0
    for tok in series:
        if not _is_content_token(tok):
            continue
        total += hebrew_char_count(tok)
    return total


def score_sentence_list_hebrew(sentences: list[dict[str, Any]], max_sentences: int = 500) -> int:
    total = 0
    for sent in sentences[:max_sentences]:
        text = str(sent.get("text", ""))
        for word in text.split():
            if not _is_content_token(word):
                continue
            total += hebrew_char_count(word)
    return total


def _encoding_candidates(raw: bytes) -> list[str]:
    override = (os.environ.get("THESIS_CSV_ENCODING") or "").strip()
    candidates: list[str] = []
    if override:
        candidates.append(override)
    try:
        import chardet

        detected = chardet.detect(raw)
        enc = (detected or {}).get("encoding")
        if enc:
            candidates.append(str(enc))
    except Exception:
        pass
    for enc in DEFAULT_CSV_ENCODINGS:
        if enc not in candidates:
            candidates.append(enc)
    return candidates


def read_ner_dataset_csv(
    path: str | Path,
    *,
    token_col: str = "token",
    min_hebrew_score: int = 50,
) -> tuple[pd.DataFrame, str]:
    """Read ``ner_dataset.csv`` using the encoding that yields the most Hebrew letters.

    Raises ``HebrewCorpusEncodingError`` if no decode yields meaningful Hebrew
    (typical when the file was saved as ``?`` placeholders or wrong export).
    """
    csv_path = Path(path)
    raw = csv_path.read_bytes()
    if not raw.strip():
        raise HebrewCorpusEncodingError(f"Dataset file is empty: {csv_path}")

    best_df: pd.DataFrame | None = None
    best_enc: str | None = None
    best_score = -1
    decode_errors: list[str] = []

    for enc in _encoding_candidates(raw):
        try:
            df = pd.read_csv(io.BytesIO(raw), delimiter=",", encoding=enc)
        except Exception as exc:
            decode_errors.append(f"{enc}: {exc}")
            continue
        score = score_token_column_hebrew(df, token_col=token_col)
        if score > best_score:
            best_score = score
            best_df = df
            best_enc = enc

    if best_df is None or best_enc is None:
        raise HebrewCorpusEncodingError(
            f"Could not decode {csv_path} with any candidate encoding. "
            f"Tried: {_encoding_candidates(raw)}. Errors: {decode_errors[:5]}"
        )

    if _skip_hebrew_validation():
        return best_df, best_enc

    if best_score < min_hebrew_score:
        preview = ""
        if token_col in best_df.columns:
            sample = best_df[token_col].dropna().astype(str).head(8).tolist()
            preview = f" Sample tokens: {sample!r}."
        raise HebrewCorpusEncodingError(
            "Hebrew NER CSV appears corrupted or wrong encoding "
            f"(encoding={best_enc!r}, hebrew_char_score={best_score}, need >={min_hebrew_score})."
            f"{preview} "
            "Re-copy `ner_dataset.csv` as UTF-8 (or Windows-1255) from the original source, "
            "or set THESIS_CSV_ENCODING if you know the correct codec. "
            "Do not train until tokens show Hebrew letters (Unicode Hebrew block), not question marks."
        )

    return best_df, best_enc


def validate_hebrew_sentence_list(
    sentences: list[dict[str, Any]],
    *,
    context: str = "corpus",
    min_hebrew_char_score: int = 50,
    max_corrupt_token_ratio: float = 0.35,
    min_hebrew_token_ratio: float = 0.12,
) -> None:
    """Fail fast before training if splits look like ``????`` gibberish."""
    if _skip_hebrew_validation():
        return
    if not sentences:
        raise HebrewCorpusEncodingError(f"{context}: no sentences to validate.")

    hebrew_score = score_sentence_list_hebrew(sentences)
    if hebrew_score < min_hebrew_char_score:
        sample_text = str(sentences[0].get("text", ""))[:120]
        raise HebrewCorpusEncodingError(
            f"{context}: almost no Hebrew detected (score={hebrew_score}). "
            f"First sentence preview: {sample_text!r}. "
            "Regenerate splits from a valid UTF-8 ner_dataset.csv."
        )

    corrupt = 0
    hebrew_tokens = 0
    content_tokens = 0
    for sent in sentences[: min(len(sentences), 800)]:
        for word in str(sent.get("text", "")).split():
            if not _is_content_token(word):
                continue
            content_tokens += 1
            if CORRUPT_TOKEN_RE.match(word):
                corrupt += 1
            elif hebrew_char_count(word) > 0:
                hebrew_tokens += 1

    if content_tokens == 0:
        raise HebrewCorpusEncodingError(f"{context}: no content tokens found in sentences.")

    corrupt_ratio = corrupt / content_tokens
    hebrew_ratio = hebrew_tokens / content_tokens
    if corrupt_ratio > max_corrupt_token_ratio or hebrew_ratio < min_hebrew_token_ratio:
        raise HebrewCorpusEncodingError(
            f"{context}: text looks corrupted "
            f"(corrupt_token_ratio={corrupt_ratio:.2f}, hebrew_token_ratio={hebrew_ratio:.2f}). "
            "Fix dataset encoding and rerun exp07 split generation before training."
        )


def validate_hebrew_dataframe(
    df: pd.DataFrame,
    *,
    context: str = "dataset",
    token_col: str = "token",
    min_hebrew_score: int = 50,
) -> None:
    if _skip_hebrew_validation():
        return
    score = score_token_column_hebrew(df, token_col=token_col)
    if score < min_hebrew_score:
        sample: list[str] = []
        if token_col in df.columns:
            sample = df[token_col].dropna().astype(str).head(8).tolist()
        raise HebrewCorpusEncodingError(
            f"{context}: Hebrew validation failed (score={score}). Sample tokens: {sample!r}."
        )
