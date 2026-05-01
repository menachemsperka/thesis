from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score

from common import (
    configure_model_environment,
    get_experiment_output_dir,
    is_debug_enabled,
    now_timestamp,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
    write_split_runs_excel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from NERtraining import PrepDataSetNERTraining  # type: ignore


def _safe_label_name(label_idx: int, label_list: list[str]) -> str | None:
    if label_idx == -100:
        return None
    if 0 <= int(label_idx) < len(label_list):
        return label_list[int(label_idx)]
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits)
    exp_vals = np.exp(stable)
    return exp_vals / np.sum(exp_vals)


def _build_regular_token_predictions(eval_ds, trainer, tokenizer, label_list: list[str]) -> pd.DataFrame:
    preds, _, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds, axis=2)

    rows = []
    for sentence_idx, item in enumerate(eval_ds, start=1):
        input_ids = item["input_ids"]
        true_ids = item["labels"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_id = 0
        for token, true_id, pred_id, token_logits in zip(tokens, true_ids, pred_ids[sentence_idx - 1], preds[sentence_idx - 1]):
            true_label = _safe_label_name(int(true_id), label_list)
            pred_label = _safe_label_name(int(pred_id), label_list)
            if int(true_id) == -100 or true_label is None or pred_label is None or str(token).startswith("["):
                continue

            token_id += 1
            probs = _softmax(np.asarray(token_logits, dtype=np.float64))
            rows.append(
                {
                    "sentence_id": sentence_idx,
                    "token_idx": token_id,
                    "token": str(token),
                    "true_label": true_label,
                    "regular_pred_label": pred_label,
                    "regular_prob": float(np.max(probs)),
                }
            )

    return pd.DataFrame(rows)


def _bio_type_to_label(bio_value, etype_value) -> str:
    bio = str(bio_value) if bio_value is not None else "O"
    if bio == "O":
        return "O"
    etype = None if etype_value is None or pd.isna(etype_value) else str(etype_value)
    if not etype or etype == "None":
        return "O"
    return f"{bio}-{etype}"


def _load_cascaded_predicted_tokens(excel_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(excel_path, sheet_name=None)
    detailed = sheets.get("detailed_results")
    if detailed is None or detailed.empty:
        raise ValueError(f"Missing or empty detailed_results sheet in: {excel_path}")

    if "eval_mode" in detailed.columns:
        detailed = detailed[detailed["eval_mode"].astype(str) == "predicted"].copy()

    required_columns = {"sentence_id", "token_idx", "token", "true_bio", "true_etype", "pred_bio", "pred_etype", "entity_prob", "bio_prob"}
    missing = required_columns - set(detailed.columns)
    if missing:
        raise ValueError(f"Cascaded output missing required columns: {sorted(missing)}")

    detailed["sentence_id"] = detailed["sentence_id"].astype(int)
    detailed["token_idx"] = detailed["token_idx"].astype(int)

    def _cascade_confidence(row) -> float:
        e_prob = float(row.get("entity_prob", 0.0) or 0.0)
        b_prob = float(row.get("bio_prob", 0.0) or 0.0)
        pred_bio = str(row.get("pred_bio", "O"))
        if pred_bio == "O":
            return 1.0 - e_prob
        return e_prob * b_prob

    cascaded = detailed[["sentence_id", "token_idx", "token", "true_bio", "true_etype", "pred_bio", "pred_etype", "entity_prob", "bio_prob"]].copy()
    cascaded["cascade_true_label"] = [
        _bio_type_to_label(b, t)
        for b, t in zip(cascaded["true_bio"], cascaded["true_etype"])
    ]
    cascaded["cascade_pred_label"] = [
        _bio_type_to_label(b, t)
        for b, t in zip(cascaded["pred_bio"], cascaded["pred_etype"])
    ]
    cascaded["cascade_prob"] = cascaded.apply(_cascade_confidence, axis=1)
    return cascaded


def _to_seqeval_lists(df: pd.DataFrame, col_true: str, col_pred: str) -> tuple[list[list[str]], list[list[str]]]:
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    ordered = df.sort_values(["sentence_id", "token_idx"]).groupby("sentence_id", sort=True)
    for _, group in ordered:
        y_true.append(group[col_true].astype(str).tolist())
        y_pred.append(group[col_pred].astype(str).tolist())

    return y_true, y_pred


def _run_cascaded_pipeline_subprocess() -> subprocess.CompletedProcess:
    debug = is_debug_enabled()
    run_kwargs = {
        "cwd": str(CORE_DIR),
        "check": False,
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
    }
    if not debug:
        run_kwargs.update({
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        })

    return subprocess.run([sys.executable, str(CORE_DIR / "auc_cascaded_pipeline.py")], **run_kwargs)


def run() -> dict:
    dataset_path = resolve_dataset("ner_dataset.csv")
    model_name, is_local_model = configure_model_environment()

    seed_raw = (os.environ.get("THESIS_SPLIT_SEED") or "42").strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 42

    with suppress_output_if_needed():
        worker = PrepDataSetNERTraining()
        data = worker.load_and_prepare_data(str(dataset_path))

        # Support pre-computed splits from experiment 07
        presplit_train = (os.environ.get("THESIS_PRESPLIT_TRAIN_JSON") or "").strip()
        presplit_eval = (os.environ.get("THESIS_PRESPLIT_EVAL_JSON") or "").strip()
        if presplit_train and presplit_eval and Path(presplit_train).exists() and Path(presplit_eval).exists():
            from split_io import load_split
            train_sentences = load_split(Path(presplit_train))
            eval_sentences = load_split(Path(presplit_eval))
            trainer, eval_results, label_list, ds_eval = worker.run_training_with_presplit(
                data, train_sentences, eval_sentences,
            )
        else:
            trainer, eval_results, label_list, ds_eval = worker.run_training_steps(data)

        processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        regular_tokens_df = _build_regular_token_predictions(ds_eval, trainer, processor, label_list)

    completed = _run_cascaded_pipeline_subprocess()
    if completed.returncode != 0:
        stderr_tail = ""
        stdout_tail = ""
        if isinstance(completed.stderr, str):
            stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-40:])
        if isinstance(completed.stdout, str):
            stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-40:])
        details = stderr_tail or stdout_tail or "No subprocess output captured."
        raise RuntimeError(
            "Cascaded pipeline failed. "
            f"Exit code: {completed.returncode}. "
            f"Last output lines:\n{details}"
        )

    source_metrics_path = CORE_DIR / "cascaded_pipeline_results.xlsx"
    exp_dir = get_experiment_output_dir("exp06")
    archived_source_metrics = exp_dir / f"fusion_source_cascaded_{now_timestamp()}.xlsx"
    if source_metrics_path.exists():
        shutil.move(str(source_metrics_path), str(archived_source_metrics))
    else:
        raise FileNotFoundError("cascaded_pipeline_results.xlsx was not generated by experiment 4 core pipeline")

    cascaded_tokens_df = _load_cascaded_predicted_tokens(archived_source_metrics)

    merged = regular_tokens_df.merge(
        cascaded_tokens_df,
        on=["sentence_id", "token_idx"],
        how="inner",
        suffixes=("_regular", "_cascade"),
    )
    if merged.empty:
        raise RuntimeError("Fusion failed: no aligned tokens between regular and cascaded outputs.")

    merged["true_label"] = merged["true_label"].astype(str)
    merged["cascade_true_label"] = merged["cascade_true_label"].astype(str)

    mismatched_truth = int((merged["true_label"] != merged["cascade_true_label"]).sum())

    merged["disagree"] = merged["regular_pred_label"].astype(str) != merged["cascade_pred_label"].astype(str)

    def _choose_fused(row) -> tuple[str, str, float]:
        regular_label = str(row["regular_pred_label"])
        cascade_label = str(row["cascade_pred_label"])
        regular_prob = float(row["regular_prob"])
        cascade_prob = float(row["cascade_prob"])

        if regular_label == cascade_label:
            return regular_label, "agree", max(regular_prob, cascade_prob)

        if regular_prob >= cascade_prob:
            return regular_label, "regular", regular_prob
        return cascade_label, "cascade", cascade_prob

    selected = merged.apply(_choose_fused, axis=1, result_type="expand")
    selected.columns = ["fused_pred_label", "selected_source", "selected_confidence"]
    merged = pd.concat([merged, selected], axis=1)

    y_true, y_pred = _to_seqeval_lists(merged, "true_label", "fused_pred_label")
    precision = float(precision_score(y_true, y_pred)) if y_true else None
    recall = float(recall_score(y_true, y_pred)) if y_true else None
    f1 = float(f1_score(y_true, y_pred)) if y_true else None

    disagreement_count = int(merged["disagree"].sum())
    fusion_from_regular = int((merged["selected_source"] == "regular").sum())
    fusion_from_cascade = int((merged["selected_source"] == "cascade").sum())
    agreement_count = int((merged["selected_source"] == "agree").sum())

    metrics_df = pd.DataFrame(
        [
            {
                "dataset_name": "ner_dataset.csv",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tokens_aligned": int(len(merged)),
                "disagreements": disagreement_count,
                "selected_regular": fusion_from_regular,
                "selected_cascade": fusion_from_cascade,
                "agreements": agreement_count,
                "truth_label_mismatch_between_models": mismatched_truth,
            }
        ]
    )

    detailed_df = merged[
        [
            "sentence_id",
            "token_idx",
            "token_regular",
            "true_label",
            "regular_pred_label",
            "regular_prob",
            "cascade_pred_label",
            "cascade_prob",
            "disagree",
            "selected_source",
            "selected_confidence",
            "fused_pred_label",
        ]
    ].rename(columns={"token_regular": "token"})

    metrics_file = write_result_excel(
        "exp06",
        "fusion_regular_cascaded_results",
        metrics_df,
        detailed_df,
        extra_sheets={
            "regular_tokens": regular_tokens_df,
            "cascaded_tokens": cascaded_tokens_df,
        },
    )

    result = {
        "experiment_id": "exp06",
        "name": "Fusion of Regular NER and AUC Cascaded Pipeline",
        "description": "Combines experiment 1 and experiment 4 predictions; when labels contradict, selects the prediction with the higher probability.",
        "dataset": str(dataset_path),
        "model": model_name,
        "model_local": is_local_model,
        "training_parameters": {
            "model_name": model_name,
            "model_local_only": is_local_model,
            "train_fraction": 0.7,
            "validation_fraction": 0.3,
            "split_seed": split_seed,
            "split_strategy": "statistical stratified sentence split preserving non-O label distribution (with best-effort train coverage)",
            "fusion_rule": "if regular_pred_label != cascade_pred_label, choose label with higher confidence/probability",
            "regular_confidence": "max softmax probability from regular token classifier",
            "cascade_confidence": "(1-entity_prob) for O else entity_prob*bio_prob",
        },
        "source_metrics_file_cascaded": str(archived_source_metrics),
        "metrics_file": str(metrics_file),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "status": "ok",
    }

    out_path = write_result_json("exp06", "fusion_regular_cascaded", result)
    result["result_file"] = str(out_path)
    return result


if __name__ == "__main__":
    existing_seed = os.environ.get("THESIS_SPLIT_SEED")
    if existing_seed is not None:
        payload = run()
        print(
            f"[exp06] F1={payload['f1']:.4f} | {payload['description']}"
            if payload["f1"] is not None
            else f"[exp06] F1=N/A | {payload['description']}"
        )
    else:
        num_runs = int((os.environ.get("THESIS_DIRECT_SPLIT_RUNS") or "5").strip() or "5")
        base_seed = int((os.environ.get("THESIS_DIRECT_BASE_SEED") or "42").strip() or "42")
        split_rows = []
        for run_idx in range(1, num_runs + 1):
            split_seed = base_seed + (run_idx - 1)
            os.environ["THESIS_SPLIT_SEED"] = str(split_seed)
            payload = run()
            split = payload.get("training_parameters", {})
            split_rows.append(
                {
                    "run_index": run_idx,
                    "split_seed": split_seed,
                    "f1": payload.get("f1"),
                    "precision": payload.get("precision"),
                    "recall": payload.get("recall"),
                    "train_fraction": split.get("train_fraction"),
                    "validation_fraction": split.get("validation_fraction"),
                    "split_strategy": split.get("split_strategy"),
                    "metrics_file": payload.get("metrics_file"),
                    "result_file": payload.get("result_file"),
                    "status": payload.get("status"),
                }
            )
            print(f"[exp06] run {run_idx}/{num_runs} seed={split_seed} F1={payload.get('f1')}")

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
        out = write_split_runs_excel("exp06", "split_runs", runs_df, summary_df=summary_df)
        print(f"[exp06] Saved split summary: {out}")
