"""Load encoder backbones and token-classification heads for Hebrew + multilingual models."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput


def encode_words_for_ner(tokenizer, words: list[str], *, max_length: int = 512, return_tensors=None):
    """Tokenize pre-split words with model-specific options (RoBERTa needs add_prefix_space)."""
    kwargs: dict = {
        "is_split_into_words": True,
        "truncation": True,
        "max_length": max_length,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    tok_name = type(tokenizer).__name__.lower()
    if "roberta" in tok_name or "xlm" in tok_name:
        kwargs["add_prefix_space"] = True
    return tokenizer(words, **kwargs)


def model_is_seq2seq_encoder(model_name: str, *, local_files_only: bool = False) -> bool:
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    return config.model_type in {"mt5", "t5"}


def load_encoder_backbone(model_name: str, *, local_files_only: bool = False):
    """Return a module whose forward(input_ids, attention_mask) exposes ``last_hidden_state``."""
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    if config.model_type == "mt5":
        from transformers import MT5EncoderModel

        return MT5EncoderModel.from_pretrained(
            model_name, config=config, local_files_only=local_files_only
        )
    if config.model_type == "t5":
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(
            model_name, config=config, local_files_only=local_files_only
        )
    return AutoModel.from_pretrained(model_name, config=config, local_files_only=local_files_only)


class _EncoderTokenClassifier(nn.Module):
    """Token classifier on top of an encoder (used for mT5 / T5 encoder stacks)."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)


def load_token_classification_model(
    model_name: str,
    *,
    label_list: list[str],
    local_files_only: bool = False,
) -> tuple[nn.Module, AutoTokenizer, AutoConfig]:
    """Load a token-classification model and tokenizer for Exp01-style training."""
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label_to_id.items()}
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label_to_id,
        num_labels=num_labels,
        local_files_only=local_files_only,
    )
    for attr in ("finetuning_task",):
        if hasattr(config, attr):
            delattr(config, attr)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    if config.model_type in {"mt5", "t5"}:
        encoder = load_encoder_backbone(model_name, local_files_only=local_files_only)
        hidden_size = int(getattr(encoder.config, "d_model", getattr(config, "d_model", 768)))
        model = _EncoderTokenClassifier(
            encoder,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label_to_id,
            hidden_size=hidden_size,
        )
        return model, tokenizer, config

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer, config
