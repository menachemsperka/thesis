from __future__ import annotations

import os
import sys
from pathlib import Path

import chardet
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from common import (
    configure_model_environment,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
    write_split_runs_excel,
)


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from auc_2t_training import (  # type: ignore
    convert_binary_to_bio,
    evaluate_model,
    predict_with_auc_2t,
    preprocess_data,
    train_auc_2t,
)
import th_functions as tf  # type: ignore


TRAIN_FRACTION = 0.7
RANDOM_STATE = 42
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
BATCH_SIZE = 16
LAMBDA_PARAM = 100.0
MARGIN = 1.0


def _resolve_split_seed() -> int:
    env_seed = (os.environ.get("THESIS_SPLIT_SEED") or "").strip()
    if env_seed:
        try:
            return int(env_seed)
        except ValueError:
            pass
    return RANDOM_STATE


def _read_csv_with_detected_encoding(path: Path) -> pd.DataFrame:
    with open(path, "rb") as handle:
        detected = chardet.detect(handle.read())

    candidates = [
        detected.get("encoding"),
        "utf-8",
        "utf-8-sig",
        "cp1255",
        "windows-1255",
        "iso-8859-8",
        "cp1252",
    ]

    tried = []
    for encoding in candidates:
        if not encoding or encoding in tried:
            continue
        tried.append(encoding)
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("csv", b"", 0, 1, f"Failed to decode CSV with encodings: {tried}")


def _split_sentence_level(data: pd.DataFrame, train_fraction: float, split_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    sentence_col = "Sentence #" if "Sentence #" in data.columns else "id"
    sentence_items = []
    for sentence_id, group in data.groupby(sentence_col, sort=False):
        labels = group["raw_tags"].dropna().astype(str).tolist()
        sentence_items.append({"sentence_id": sentence_id, "labels": labels})

    train_items, val_items = tf.split_list(
        sentence_items,
        split_ratio=train_fraction,
        seed=split_seed,
        ensure_label_coverage=True,
    )
    train_ids = [item["sentence_id"] for item in train_items]
    val_ids = [item["sentence_id"] for item in val_items]

    train_data = data[data[sentence_col].isin(train_ids)].copy()
    val_data = data[data[sentence_col].isin(val_ids)].copy()
    return train_data, val_data, len(train_ids), len(val_ids)


def _build_detailed_results(model, eval_data: pd.DataFrame, tokenizer, entity_types: list[str]) -> pd.DataFrame:
    sentences, true_entity_labels, true_begin_labels = preprocess_data(eval_data, tokenizer)
    pred_entity_labels, pred_begin_labels = predict_with_auc_2t(model, sentences, tokenizer)

    rows = []
    default_entity = entity_types[0] if entity_types else "ENT"

    for idx, sentence in enumerate(sentences):
        pred_bio = convert_binary_to_bio(pred_entity_labels[idx], pred_begin_labels[idx], entity_types)
        true_bio = []
        for e, b in zip(true_entity_labels[idx], true_begin_labels[idx]):
            if e == -1:
                true_bio.append("O")
            elif b == 1:
                true_bio.append(f"B-{default_entity}")
            else:
                true_bio.append(f"I-{default_entity}")

        min_len = min(len(true_bio), len(pred_bio))
        rows.append(
            {
                "sentence": sentence,
                "true_labels": " ".join(true_bio[:min_len]),
                "predicted_labels": " ".join(pred_bio[:min_len]),
            }
        )

    return pd.DataFrame(rows)


def _build_token_level_df(detailed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sentence_idx, row in detailed_df.reset_index(drop=True).iterrows():
        tokens = str(row.get("sentence", "")).split()
        true_labels = str(row.get("true_labels", "")).split()
        pred_labels = str(row.get("predicted_labels", "")).split()
        for token_idx, (token, true_label, pred_label) in enumerate(
            zip(tokens, true_labels, pred_labels),
            start=1,
        ):
            rows.append(
                {
                    "sentence_id": sentence_idx + 1,
                    "token_id": token_idx,
                    "token": token,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "is_correct": true_label == pred_label,
                }
            )
    return pd.DataFrame(rows)


def _build_extra_sheets(token_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if token_df.empty:
        return {}

    confusion_df = pd.crosstab(token_df["true_label"], token_df["predicted_label"]).reset_index()
    report_dict = classification_report(
        token_df["true_label"],
        token_df["predicted_label"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    mismatches_df = token_df[token_df["is_correct"] == False].copy()
    return {
        "token_level": token_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
        "token_errors": mismatches_df,
    }


def run() -> dict:
    dataset_path = resolve_dataset("ner_dataset.csv")
    model_name, is_local_model = configure_model_environment()
    split_seed = _resolve_split_seed()
    data = _read_csv_with_detected_encoding(dataset_path)

    # Support pre-computed splits from experiment 07
    presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
    presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
    if presplit_train and presplit_eval and Path(presplit_train).exists() and Path(presplit_eval).exists():
        from split_io import load_split, sentences_to_dataframe
        train_sentences = load_split(Path(presplit_train))
        eval_sentences = load_split(Path(presplit_eval))
        train_data = sentences_to_dataframe(train_sentences, start_id=1)
        val_data = sentences_to_dataframe(eval_sentences, start_id=len(train_sentences) + 1)
        train_sent_count = len(train_sentences)
        val_sent_count = len(eval_sentences)
        split_strategy = f"pre-split from experiment 07 ({Path(presplit_train).stem})"
    else:
        train_data, val_data, train_sent_count, val_sent_count = _split_sentence_level(data, TRAIN_FRACTION, split_seed)
        split_strategy = "statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)"

    with suppress_output_if_needed():
        model, _ = train_auc_2t(
            train_data,
            model_name=model_name,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lambda_param=LAMBDA_PARAM,
            margin=MARGIN,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local_model)
        metrics = evaluate_model(model, val_data, tokenizer, entity_types=["ENT"])
        detailed_df = _build_detailed_results(model, val_data, tokenizer, entity_types=["ENT"])

    f1 = metrics.get("f1")
    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "f1": float(f1) if f1 is not None else None,
                "precision": float(metrics.get("precision")) if metrics.get("precision") is not None else None,
                "recall": float(metrics.get("recall")) if metrics.get("recall") is not None else None,
                "train_sentences": train_sent_count,
                "validation_sentences": val_sent_count,
            }
        ]
    )
    token_df = _build_token_level_df(detailed_df)
    extra_sheets = _build_extra_sheets(token_df)
    metrics_file = write_result_excel(
        "exp03",
        "auc_2t_results",
        metrics_df,
        detailed_df,
        extra_sheets=extra_sheets,
    )

    result = {
        "experiment_id": "exp03",
        "name": "AUC-2T NER",
        "description": "AUC-2T trained once on full training split (70%) and validated on 30%.",
        "dataset": str(dataset_path),
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": {
            "model_name": model_name,
            "model_local_only": is_local_model,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": 1 - TRAIN_FRACTION,
            "split_seed": split_seed,
            "split_strategy": split_strategy,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lambda_param": LAMBDA_PARAM,
            "margin": MARGIN,
        },
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": 1 - TRAIN_FRACTION,
            "train_sentences": train_sent_count,
            "validation_sentences": val_sent_count,
            "split_seed": split_seed,
        },
        "metrics_file": str(metrics_file),
        "f1": float(f1) if f1 is not None else None,
        "precision": float(metrics.get("precision")) if metrics.get("precision") is not None else None,
        "recall": float(metrics.get("recall")) if metrics.get("recall") is not None else None,
        "status": "ok",
    }
    out_path = write_result_json("exp03", "auc_2t", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    existing_seed = os.environ.get("THESIS_SPLIT_SEED")
    if existing_seed is not None:
        payload = run()
        if payload["f1"] is not None:
            print(
                f"[exp03] F1={payload['f1']:.4f} "
                f"(train={payload['split']['train_sentences']} | val={payload['split']['validation_sentences']})"
                f" | {payload['description']}"
            )
        else:
            print(f"[exp03] F1=N/A | {payload['description']}")
    else:
        num_runs = int((os.environ.get("THESIS_DIRECT_SPLIT_RUNS") or "5").strip() or "5")
        base_seed = int((os.environ.get("THESIS_DIRECT_BASE_SEED") or "42").strip() or "42")
        split_rows = []
        for run_idx in range(1, num_runs + 1):
            split_seed = base_seed + (run_idx - 1)
            os.environ["THESIS_SPLIT_SEED"] = str(split_seed)
            payload = run()
            split = payload.get("split", {})
            split_rows.append(
                {
                    "run_index": run_idx,
                    "split_seed": split_seed,
                    "f1": payload.get("f1"),
                    "precision": payload.get("precision"),
                    "recall": payload.get("recall"),
                    "train_fraction": split.get("train_fraction"),
                    "validation_fraction": split.get("validation_fraction"),
                    "train_sentences": split.get("train_sentences"),
                    "validation_sentences": split.get("validation_sentences"),
                    "split_strategy": (payload.get("training_parameters") or {}).get("split_strategy"),
                    "metrics_file": payload.get("metrics_file"),
                    "result_file": payload.get("result_file"),
                    "status": payload.get("status"),
                }
            )
            print(f"[exp03] run {run_idx}/{num_runs} seed={split_seed} F1={payload.get('f1')}")

        if existing_seed is None:
            os.environ.pop("THESIS_SPLIT_SEED", None)

        runs_df = pd.DataFrame(split_rows)
        ok_f1 = [float(v) for v in runs_df["f1"].dropna().tolist()]
        summary_df = pd.DataFrame(
            [
                {
                    "runs": num_runs,
                    "base_seed": base_seed,
                    "f1_mean": (sum(ok_f1) / len(ok_f1)) if ok_f1 else None,
                    "f1_best": max(ok_f1) if ok_f1 else None,
                    "f1_worst": min(ok_f1) if ok_f1 else None,
                }
            ]
        )
        out = write_split_runs_excel("exp03", "split_runs", runs_df, summary_df=summary_df)
        print(f"[exp03] Saved split summary: {out}")
