"""
predict_ner.py — Run NER on an unlabeled CSV using a model from the Hugging Face Hub.

Input CSV has the same shape as the project's training data but WITHOUT the label
columns — i.e. just ``id`` and ``token``:

    id,token
    1,ר׳
    1,משה
    1,בן
    1,מימון
    2,ספר
    2,המצוות

Each ``id`` is one sentence; rows are tokens in order. Special markers such as
``[CLS]`` / ``[SEP]`` and blank rows are ignored.

The regular NER model (Exp01) is a standard ``AutoModelForTokenClassification``, so its
label set travels with it (``config.id2label``) — nothing needs to be hardcoded here.

Usage
-----
    python predict_ner.py --input ner_unlabeled.csv --output predictions.csv
    python predict_ner.py --input ner_unlabeled.csv --repo-id msperka/berel-ner-regular

For a private repo, set HF_TOKEN in the environment first.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Tokens that are structural markers in the dataset, not real words (compared lowercased).
_SPECIAL_TOKENS = {"[cls]", "[sep]", "[pad]", "", "nan", "none"}


# A tiny sample in the training-data shape (with [CLS]/[SEP] markers) for the demo run.
_DEMO_ROWS = [
    (1, "[CLS]"), (1, "ר׳"), (1, "משה"), (1, "בן"), (1, "מימון"), (1, "כתב"), (1, "[SEP]"),
    (2, "[CLS]"), (2, "ספר"), (2, "המצוות"), (2, "פרק"), (2, "ג"), (2, "[SEP]"),
    (3, "[CLS]"), (3, "רש״י"), (3, "על"), (3, "התורה"), (3, "[SEP]"),
]


def write_demo_csv(path: str) -> str:
    """Write a small demo CSV (id, token) in the training-data shape and return its path."""
    pd.DataFrame(_DEMO_ROWS, columns=["id", "token"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    return path


def read_sentences(csv_path: str, id_col: str, token_col: str):
    """Yield (sentence_id, [words]) from an unlabeled CSV, skipping marker tokens."""
    df = pd.read_csv(csv_path)
    for col in (id_col, token_col):
        if col not in df.columns:
            raise SystemExit(
                f"Column '{col}' not found in {csv_path}. Available: {list(df.columns)}"
            )
    df[token_col] = df[token_col].astype(str)
    for sid, group in df.groupby(id_col, sort=False):
        words = [
            tok for tok in group[token_col].tolist()
            if tok.strip() and tok.strip().lower() not in _SPECIAL_TOKENS
        ]
        if words:
            yield sid, words


@torch.no_grad()
def predict_sentence(words, tokenizer, model, id2label, device):
    """Return a list of (token, pred_label, prob) for one pre-tokenized sentence."""
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    logits = model(**encoding).logits[0]
    probs = logits.softmax(dim=-1)
    word_ids = encoding.word_ids()

    rows = []
    seen: set[int] = set()
    for position, word_idx in enumerate(word_ids):
        # Keep only the first sub-token of each original word (matches training).
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        prob, label_idx = probs[position].max(dim=-1)
        rows.append((words[word_idx], id2label[int(label_idx)], float(prob)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NER labels for an unlabeled CSV using a HF model.")
    parser.add_argument("--input", help="Path to the unlabeled CSV (columns: id, token). If omitted, a demo CSV is generated.")
    parser.add_argument("--demo", action="store_true", help="Run on a generated demo CSV instead of --input.")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions (default: predictions.csv).")
    parser.add_argument("--repo-id", default="msperka/berel-ner-regular",
                        help="Hugging Face model repo (default: msperka/berel-ner-regular).")
    parser.add_argument("--id-col", default="id", help="Sentence-id column name (default: id).")
    parser.add_argument("--token-col", default="token", help="Token column name (default: token).")
    args = parser.parse_args()

    if args.demo or not args.input:
        args.input = write_demo_csv("ner_unlabeled_demo.csv")
        print(f"[demo] No --input given; generated demo CSV -> {args.input}")

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip() or None

    print(f"[load] {args.repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, token=token)
    model = AutoModelForTokenClassification.from_pretrained(args.repo_id, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    id2label = model.config.id2label

    results = []
    n_sentences = 0
    for sid, words in read_sentences(args.input, args.id_col, args.token_col):
        n_sentences += 1
        for token_text, label, prob in predict_sentence(words, tokenizer, model, id2label, device):
            results.append({
                "id": sid,
                "token": token_text,
                "pred_label": label,
                "prob": round(prob, 4),
            })

    out = pd.DataFrame(results)
    # utf-8-sig so Hebrew displays correctly when opened in Excel.
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[done] {n_sentences} sentence(s), {len(out)} token(s) -> {args.output}")
    if not out.empty:
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
