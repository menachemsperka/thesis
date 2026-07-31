"""
BERT + linear emissions + CRF for Hebrew NER (Experiment 10_regular)
======================================================================

Purpose
-------
This module implements the **regular** (single-pass) NER architecture for Experiment 10.
It mirrors ``core/th_functions.py`` + ``experiments/experiment_01_regular_ner.py``, but replaces
the final **softmax** classification head with:

1. a **linear emission layer** on top of BERT hidden states, and
2. a **LinearChainCRF** (see ``core/crf_layer.py``) for sequence loss and Viterbi decoding.

Why a custom ``CRFTrainer``?
----------------------------
Hugging Face ``Trainer`` assumes ``predictions`` are per-class logits and applies ``argmax``.
For a CRF, the correct inference rule is **Viterbi** on emissions. ``CRFTrainer.prediction_step``
therefore decodes with ``model.crf.viterbi_decode`` before metrics are computed.

Class imbalance (Souza et al. 2019)
------------------------------------
When the label list contains ``O``, the bias of the emission layer's ``O`` row is initialized to **6.0**
so the model does not predict ``O`` for every token in the first epochs.

Public API (what experiment scripts call)
-----------------------------------------
* ``setup_bert_crf_token_classification`` — build model, tokenizer, datasets (WordPiece-aligned labels).
* ``train_and_evaluate_bert_crf`` — run ``CRFTrainer`` training + ``evaluate()``.
* ``prepare_eval_results_crf`` — sentence/table export when predictions are already Viterbi tag ids.

Teaching path: read ``experiments/experiment_10_README.md`` then step through ``experiment_10_regular_ner_crf.run()``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput

import pandas as pd

from crf_layer import LinearChainCRF


class BertCRFForTokenClassification(nn.Module):
    """
    Encoder (BERT) + dropout + linear emissions + CRF.

    ``forward`` returns ``TokenClassifierOutput``:
    * with ``labels`` → ``loss`` = CRF negative log-likelihood
    * without ``labels`` → ``logits`` = emissions only (decode with ``self.crf.viterbi_decode``)
    """

    def __init__(self, base_model_name: str, num_labels: int, local_files_only: bool = False, o_label_id: int | None = None):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name, local_files_only=local_files_only)
        self.bert = AutoModel.from_pretrained(base_model_name, config=config, local_files_only=local_files_only)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = LinearChainCRF(num_labels)
        self.num_labels = num_labels
        if o_label_id is not None:
            with torch.no_grad():
                self.classifier.bias[o_label_id] = 6.0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        if labels is not None:
            loss = self.crf.neg_log_likelihood(emissions, labels, attention_mask)
            return TokenClassifierOutput(loss=loss, logits=emissions)
        return TokenClassifierOutput(logits=emissions)


class CRFTrainer(Trainer):
    """Hugging Face Trainer that Viterbi-decodes CRF emissions in ``prediction_step``."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            emissions = outputs.logits
            labels = inputs.get("labels")
            mask = inputs.get("attention_mask")
            decoded = model.crf.viterbi_decode(emissions, mask)
            batch_size, seq_len, num_labels = emissions.shape
            pred_ids = emissions.new_full((batch_size, seq_len), fill_value=0, dtype=torch.long)
            for i, path in enumerate(decoded):
                for t, tag in enumerate(path):
                    if t < seq_len:
                        pred_ids[i, t] = tag
            loss = None
            if labels is not None and not prediction_loss_only:
                loss = model.crf.neg_log_likelihood(emissions, labels, mask)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, pred_ids, labels)


def setup_bert_crf_token_classification(
    data,
    train_data,
    eval_data,
    test_data,
    model_name: str,
    local_files_only: bool = False,
):
    label_list = data.raw_tags.dropna().astype(str).unique().tolist()
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    o_label_id = label_to_id.get("O")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = BertCRFForTokenClassification(
        model_name,
        num_labels=len(label_list),
        local_files_only=local_files_only,
        o_label_id=o_label_id,
    )

    class TokenClassificationTorchDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.features = []
            for record in records:
                words = str(record["text"]).split()
                labels = [str(label) for label in record["labels"]]
                encoding = tokenizer(words, is_split_into_words=True, truncation=True)
                try:
                    word_ids = encoding.word_ids()
                except Exception:
                    word_ids = [None] + list(range(min(len(words), max(0, len(encoding["input_ids"]) - 2)))) + [None]
                    if len(word_ids) < len(encoding["input_ids"]):
                        word_ids.extend([None] * (len(encoding["input_ids"]) - len(word_ids)))
                    elif len(word_ids) > len(encoding["input_ids"]):
                        word_ids = word_ids[: len(encoding["input_ids"])]

                aligned_labels = []
                prev_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        aligned_labels.append(-100)
                    elif word_idx != prev_word_idx:
                        label_name = labels[word_idx] if word_idx < len(labels) else "O"
                        aligned_labels.append(label_to_id.get(label_name, label_to_id.get("O", 0)))
                    else:
                        aligned_labels.append(-100)
                    prev_word_idx = word_idx

                encoding["labels"] = aligned_labels
                self.features.append(encoding)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]

    from transformers import DataCollatorForTokenClassification

    data_collator = DataCollatorForTokenClassification(tokenizer)
    ds_train = TokenClassificationTorchDataset(train_data)
    ds_eval = TokenClassificationTorchDataset(eval_data)
    ds_test = TokenClassificationTorchDataset(test_data)
    return model, tokenizer, data_collator, ds_train, ds_eval, ds_test, label_list


def train_and_evaluate_bert_crf(model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"):
    import evaluate
    import th_functions as tf

    metric = None
    try:
        metric = evaluate.load(metric_name)
    except Exception:
        metric = None

    def compute_metrics_crf(p):
        predictions, labels = p
        if getattr(predictions, "ndim", 0) == 3:
            predictions = predictions.argmax(axis=2)
        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        if metric is not None:
            return metric.compute(predictions=true_predictions, references=true_labels)
        try:
            from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

            return {
                "overall_precision": precision_score(true_labels, true_predictions),
                "overall_recall": recall_score(true_labels, true_predictions),
                "overall_f1": f1_score(true_labels, true_predictions),
                "overall_accuracy": accuracy_score(true_labels, true_predictions),
            }
        except Exception:
            return {
                "overall_precision": 0.0,
                "overall_recall": 0.0,
                "overall_f1": 0.0,
                "overall_accuracy": 0.0,
            }

    epochs_raw = (os.environ.get("THESIS_NUM_EPOCHS") or "").strip()
    try:
        num_train_epochs = float(epochs_raw) if epochs_raw else 3.0
    except ValueError:
        num_train_epochs = 3.0

    is_colab = os.environ.get("THESIS_RUN_ENV") == "colab"
    if is_colab:
        exp_id_env = os.environ.get("THESIS_CURRENT_EXP_ID", "unknown_exp")
        model_id_env = os.environ.get("THESIS_MODEL_NAME", "unknown_model")
        cond_key_env = os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default")
        seed_env = os.environ.get("THESIS_SPLIT_SEED", "42")
        model_short_env = model_id_env.replace("/", "_").replace("\\", "_").split("_")[-1]
        unique_run_name = f"{exp_id_env}_{model_short_env}_{cond_key_env}_seed{seed_env}"
        default_out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "trainer_checkpoints", unique_run_name)
        sig = inspect.signature(TrainingArguments.__init__)
        colab_kwargs: dict[str, Any] = {
            "output_dir": default_out_dir,
            "num_train_epochs": num_train_epochs,
            "save_strategy": "steps",
            "save_steps": 100,
            "eval_steps": 100,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "overall_f1",
            "fp16": True,
        }
        if "save_only_model" in sig.parameters:
            colab_kwargs["save_only_model"] = True
        if "eval_strategy" in sig.parameters:
            colab_kwargs["eval_strategy"] = "steps"
        else:
            colab_kwargs["evaluation_strategy"] = "steps"
        training_args = TrainingArguments(**colab_kwargs)
    else:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tmp_trainer")
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=num_train_epochs,
            save_strategy="no",
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": ds_train,
        "eval_dataset": ds_eval,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_crf,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = CRFTrainer(**trainer_kwargs)
    if num_train_epochs > 0.0:
        trainer.train()
    evaluation_results = trainer.evaluate()
    return trainer, evaluation_results


def prepare_eval_results_crf(eval_ds, trainer, tokenizer, label_list):
    """Like NERtraining.prepare_eval_results but accepts Viterbi label ids (2D predictions)."""
    import numpy as np
    from NERtraining import prepare_eval_results

    preds, _, _ = trainer.predict(eval_ds)
    if getattr(preds, "ndim", 0) == 3:
        return prepare_eval_results(eval_ds, trainer, tokenizer, label_list)

    preds_labels = preds

    def _safe_label_name(label_idx):
        if label_idx == -100:
            return None
        if isinstance(label_idx, (np.integer, int)) and 0 <= int(label_idx) < len(label_list):
            return label_list[int(label_idx)]
        return None

    sentences, true_labels, pred_labels = [], [], []
    for i, item in enumerate(eval_ds):
        tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        true = [_safe_label_name(l) for l in item["labels"]]
        pred = [_safe_label_name(p) for p in preds_labels[i]]
        filtered = [
            (tok, t, p)
            for tok, t, p, l in zip(tokens, true, pred, item["labels"])
            if l != -100 and t is not None and p is not None and not str(tok).startswith("[")
        ]
        if filtered:
            tokens_f, true_f, pred_f = zip(*filtered)
            sentences.append(" ".join(tokens_f))
            true_labels.append(" ".join(true_f))
            pred_labels.append(" ".join(pred_f))

    df_sentences = pd.DataFrame(
        {"Sentence": sentences, "True Labels": true_labels, "Predicted Labels": pred_labels}
    )

    all_true, all_pred = [], []
    for i, item in enumerate(eval_ds):
        tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        for t, p, l, tok in zip(item["labels"], preds_labels[i], item["labels"], tokens):
            true_label = _safe_label_name(t)
            pred_label = _safe_label_name(p)
            if l != -100 and true_label is not None and pred_label is not None and not str(tok).startswith("["):
                all_true.append(true_label)
                all_pred.append(pred_label)

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report as sk_classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    cm = confusion_matrix(all_true, all_pred, labels=label_list)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
    report_dict = sk_classification_report(
        all_true, all_pred, labels=label_list, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    cm_array = cm.values if hasattr(cm, "values") else cm
    metrics_list = []
    for i, lbl in enumerate(label_list):
        tp = cm_array[i, i]
        fp = cm_array[:, i].sum() - tp
        fn = cm_array[i, :].sum() - tp
        tn = cm_array.sum() - (tp + fp + fn)
        metrics_list.append([lbl, tp, fp, fn, tn])
    extra_df = pd.DataFrame(metrics_list, columns=["Label", "TP", "FP", "FN", "TN"]).set_index("Label")
    report_df = report_df.join(extra_df, how="left")

    all_true_no_o = [t for t in all_true if t != "O"]
    all_pred_no_o = [p for t, p in zip(all_true, all_pred) if t != "O"]

    global_metrics = {
        "accuracy_with_o": accuracy_score(all_true, all_pred),
        "balanced_accuracy_with_o": balanced_accuracy_score(all_true, all_pred),
        "cohen_kappa_with_o": cohen_kappa_score(all_true, all_pred),
        "matthews_corrcoef_with_o": matthews_corrcoef(all_true, all_pred),
        "f1_micro_with_o": f1_score(all_true, all_pred, average="micro"),
        "f1_macro_with_o": f1_score(all_true, all_pred, average="macro"),
        "f1_weighted_with_o": f1_score(all_true, all_pred, average="weighted"),
        "accuracy_no_o": accuracy_score(all_true_no_o, all_pred_no_o) if all_true_no_o else 0.0,
        "balanced_accuracy_no_o": balanced_accuracy_score(all_true_no_o, all_pred_no_o) if all_true_no_o else 0.0,
        "cohen_kappa_no_o": cohen_kappa_score(all_true_no_o, all_pred_no_o) if all_true_no_o else 0.0,
        "matthews_corrcoef_no_o": matthews_corrcoef(all_true_no_o, all_pred_no_o) if all_true_no_o else 0.0,
        "f1_micro_no_o": f1_score(all_true_no_o, all_pred_no_o, average="micro") if all_true_no_o else 0.0,
        "f1_macro_no_o": f1_score(all_true_no_o, all_pred_no_o, average="macro") if all_true_no_o else 0.0,
        "f1_weighted_no_o": f1_score(all_true_no_o, all_pred_no_o, average="weighted") if all_true_no_o else 0.0,
    }
    return df_sentences, cm_df, report_df, [cm, label_list], global_metrics
