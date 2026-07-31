"""
Cascaded NER pipeline with a full-sequence CRF head (Experiment 10_cascade)
============================================================================

Teaching overview
-----------------
Experiment **04** (`core/auc_cascaded_pipeline.py`) trains **three heads** on a shared BERT encoder:

1. Is this token an entity? (binary)
2. Is it B or I? (binary, masked to entity tokens)
3. What entity type? (multi-class, masked to entity tokens)

Predictions are **composed** into BIO + type, then span F1 is computed. That composition can break
BIO constraints even when each head is accurate in isolation.

Experiment **10_cascade** **keeps** those three heads (so students can still read step-wise metrics),
and adds a **fourth head**: linear emissions over **full tags** (`B-PER`, `I-PER`, `O`, …) plus a
**CRF loss** and **Viterbi decode** for the **pipeline span** prediction used in ``detailed_results``.

Post-processing
---------------
When ``THESIS_STEP3_BI_TYPE_RECONCILE=1``, the same **B/I entity-type consistency** idea as
Experiment 05_ready is applied to decoded BIO + type rows (if ``B-X`` is followed by ``I-Y``, pick one type).

Entry points
------------
* ``run_cascaded_crf_pipeline()`` — main training/eval/export loop (invoked by subprocess from
  ``experiments/experiment_10_cascaded_pipeline_crf.py``).
* Reuses data loading, span F1, and threshold tuning helpers from ``auc_cascaded_pipeline`` as ``cap``.

See ``experiments/experiment_10_README.md`` for the architecture diagram.
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from crf_layer import LinearChainCRF

import auc_cascaded_pipeline as cap


class CascadedNERDatasetCRF(cap.CascadedNERDataset):
    def __init__(self, data, tokenizer, etype_to_id, tag_to_id, max_length=cap.MAX_LENGTH):
        super().__init__(data, tokenizer, etype_to_id, max_length=max_length)
        self.tag_to_id = tag_to_id

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample = self.data[idx]
        tokens = sample["tokens"]
        bio_tags = sample["bio_tags"]
        entity_types = sample["entity_types"]
        w_tags = []
        for bio, etype in zip(bio_tags, entity_types):
            if bio == "O":
                w_tags.append("O")
            else:
                w_tags.append(f"{bio}-{etype}")
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        a_tags = []
        for wi in word_ids:
            if wi is None:
                a_tags.append(-100)
            else:
                a_tags.append(self.tag_to_id.get(w_tags[wi], self.tag_to_id.get("O", 0)))
        item["tag_labels"] = torch.tensor(a_tags, dtype=torch.long)
        return item


class CascadedNERModelCRF(cap.CascadedNERModel):
    def __init__(self, base_model, num_entity_types, num_full_tags, dropout=0.1, o_tag_id: int | None = None):
        super().__init__(base_model, num_entity_types, dropout=dropout)
        h = base_model.config.hidden_size
        self.tag_emissions = nn.Linear(h, num_full_tags)
        self.tag_crf = LinearChainCRF(num_full_tags)
        if o_tag_id is not None:
            with torch.no_grad():
                self.tag_emissions.bias[o_tag_id] = 6.0

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ent = self.entity_head(h).squeeze(-1)
        bio = self.bio_head(h).squeeze(-1)
        typ = self.type_head(h)
        tag_emit = self.tag_emissions(h)
        return ent, bio, typ, tag_emit


def _collate_crf(pad_token_id, batch):
    base = cap.make_collate_fn(pad_token_id)(batch)
    mx = base["input_ids"].size(1)
    tags = []
    for x in batch:
        t = x["tag_labels"]
        p = mx - t.size(0)
        tags.append(F.pad(t, (0, p), value=-100))
    base["tag_labels"] = torch.stack(tags)
    return base


def train_epoch_crf(model, loader, optimizer, scheduler, device, cfg_loss, grad_accum, max_grad_norm, lambda_crf=0.5):
    model.train()
    entity_loss_fn = cap.get_binary_loss_fn(cfg_loss["entity_loss"], cfg_loss)
    bio_loss_fn = cap.get_binary_loss_fn(cfg_loss["bio_loss"], cfg_loss)
    lam_bio = cfg_loss["lambda_bio"]
    lam_type = cfg_loss["lambda_type"]
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        e_lab = batch["entity_labels"].to(device)
        b_lab = batch["bio_labels"].to(device)
        t_lab = batch["type_labels"].to(device)
        tag_lab = batch["tag_labels"].to(device)

        ent_logits, bio_logits, typ_logits, tag_emit = model(ids, mask)

        valid = e_lab != -100
        if valid.sum() == 0:
            continue
        loss = entity_loss_fn(ent_logits[valid], e_lab[valid].float())

        bio_valid = b_lab != -100
        if bio_valid.sum() > 0:
            loss = loss + lam_bio * bio_loss_fn(bio_logits[bio_valid], b_lab[bio_valid].float())

        typ_valid = t_lab != -100
        if typ_valid.sum() > 0:
            loss = loss + lam_type * F.cross_entropy(typ_logits[typ_valid], t_lab[typ_valid])

        loss = loss + lambda_crf * model.tag_crf.neg_log_likelihood(tag_emit, tag_lab, mask)

        (loss / grad_accum).backward()
        total_loss += loss.item()

        if step % grad_accum == 0 or step == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(1, len(loader))


def _aggregate_tag_emissions_to_words(tag_emit, token_indices, orig_tokens, orig_bio_tags, orig_entity_types):
    words = OrderedDict()
    for i in range(tag_emit.size(0)):
        wi = token_indices[i].item()
        if wi < 0:
            continue
        if wi not in words:
            words[wi] = {"emits": []}
        words[wi]["emits"].append(tag_emit[i].cpu().numpy())

    results = []
    for wi in sorted(words.keys()):
        emit = np.mean(np.stack(words[wi]["emits"], axis=0), axis=0)
        tok = orig_tokens[wi] if wi < len(orig_tokens) else ""
        true_bio = orig_bio_tags[wi] if wi < len(orig_bio_tags) else "O"
        true_etype = orig_entity_types[wi] if wi < len(orig_entity_types) else None
        results.append(
            {
                "word_idx": wi,
                "token": tok,
                "tag_emit": emit,
                "true_bio": true_bio,
                "true_etype": true_etype,
            }
        )
    return results


def _apply_bi_consistency_rows(rows: list[dict]) -> None:
    if (os.environ.get("THESIS_STEP3_BI_TYPE_RECONCILE") or "").strip() != "1":
        return
    by_sent: dict[int, list[dict]] = {}
    for row in rows:
        by_sent.setdefault(int(row["sentence_id"]), []).append(row)
    for sent_rows in by_sent.values():
        sent_rows.sort(key=lambda r: int(r["token_idx"]))
        for i in range(len(sent_rows) - 1):
            curr, nxt = sent_rows[i], sent_rows[i + 1]
            if curr.get("pred_bio") == "B" and nxt.get("pred_bio") == "I":
                if str(curr.get("pred_etype")) != str(nxt.get("pred_etype")):
                    c_prob = float(curr.get("bio_prob") or 0.0)
                    n_prob = float(nxt.get("bio_prob") or 0.0)
                    if c_prob >= n_prob:
                        nxt["pred_etype"] = curr["pred_etype"]
                    else:
                        curr["pred_etype"] = nxt["pred_etype"]


def evaluate_crf(
    model,
    loader,
    device,
    entity_types,
    id_to_etype,
    id_to_tag,
    t_entity=0.5,
    t_bio=0.5,
    collect_details=True,
    use_oracle=True,
):
    model.eval()
    s1_true, s1_pred = [], []
    s2_true, s2_pred = [], []
    s3_true, s3_pred = [], []
    all_true_spans, all_pred_spans = set(), set()
    span_offset = 0
    detail_rows = []

    with torch.no_grad():
        sentence_counter = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            e_lab = batch["entity_labels"].to(device)
            b_lab = batch["bio_labels"].to(device)
            t_lab = batch["type_labels"].to(device)
            t_idx = batch["token_indices"].to(device)
            tok_lists = batch["tokens"]
            bio_lists = batch["bio_tags"]
            etype_lists = batch["entity_types"]

            ent_logits, bio_logits, typ_logits, tag_emit = model(ids, mask)
            batch_size = ids.size(0)
            for si in range(batch_size):
                sentence_id = sentence_counter + 1
                wl = cap.aggregate_to_words(
                    ent_logits[si],
                    bio_logits[si],
                    typ_logits[si],
                    e_lab[si],
                    b_lab[si],
                    t_lab[si],
                    t_idx[si],
                    tok_lists[si],
                    bio_lists[si],
                    etype_lists[si],
                    t_entity=t_entity,
                    t_bio=t_bio,
                )
                word_tags = _aggregate_tag_emissions_to_words(
                    tag_emit[si], t_idx[si], tok_lists[si], bio_lists[si], etype_lists[si]
                )
                tag_by_wi = {w["word_idx"]: w for w in word_tags}
                if word_tags:
                    emit_stack = torch.tensor(
                        np.stack([w["tag_emit"] for w in word_tags], axis=0),
                        device=device,
                        dtype=tag_emit.dtype,
                    ).unsqueeze(0)
                    mask_word = torch.ones(1, emit_stack.size(1), device=device, dtype=torch.long)
                    decoded = model.tag_crf.viterbi_decode(emit_stack, mask_word)[0]
                else:
                    decoded = []

                if not wl:
                    sentence_counter += 1
                    span_offset += 1
                    continue

                for w in wl:
                    s1_true.append(w["e_true"])
                    s1_pred.append(w["e_pred"])

                for w in wl:
                    if use_oracle:
                        if w["b_true"] != -100:
                            s2_true.append(w["b_true"])
                            s2_pred.append(w["b_pred"])
                    else:
                        if w["e_pred"] == 1:
                            s2_true.append(w["b_true"] if w["b_true"] != -100 else 0)
                            s2_pred.append(w["b_pred"])

                for w in wl:
                    if use_oracle:
                        if w["t_true"] != -100:
                            s3_true.append(w["t_true"])
                            s3_pred.append(w["t_pred"])
                    else:
                        if w["e_pred"] == 1 and w["t_true"] != -100:
                            s3_true.append(w["t_true"])
                            s3_pred.append(w["t_pred"])

                pred_bio, pred_etype, true_bio, true_etype = [], [], [], []
                for j, w in enumerate(wl):
                    true_bio.append(w["true_bio"])
                    true_etype.append(w["true_etype"])
                    tag_id = decoded[j] if j < len(decoded) else 0
                    full_tag = id_to_tag.get(tag_id, "O")
                    if full_tag == "O":
                        pred_bio.append("O")
                        pred_etype.append(None)
                    elif full_tag.startswith("B-") or full_tag.startswith("I-"):
                        pred_bio.append(full_tag.split("-", 1)[0])
                        pred_etype.append(full_tag.split("-", 1)[1])
                    else:
                        pred_bio.append("O")
                        pred_etype.append(None)

                pred_bio = cap.enforce_bio_constraints(pred_bio)
                ts = cap.extract_spans(true_bio, true_etype)
                ps = cap.extract_spans(pred_bio, pred_etype)
                for s, e, et in ts:
                    all_true_spans.add((s + span_offset, e + span_offset, et))
                for s, e, et in ps:
                    all_pred_spans.add((s + span_offset, e + span_offset, et))
                span_offset += len(wl) + 1

                if collect_details and len(detail_rows) < cap.MAX_DETAIL_ROWS:
                    for w, pb, pt in zip(wl, pred_bio, pred_etype):
                        detail_rows.append(
                            {
                                "sentence_id": sentence_id,
                                "token_idx": int(w["word_idx"]) + 1,
                                "token": w["token"],
                                "true_bio": w["true_bio"],
                                "true_etype": w["true_etype"],
                                "pred_bio": pb,
                                "pred_etype": pt,
                                "entity_prob": round(w["e_prob"], 4),
                                "bio_prob": round(w["b_prob"], 4),
                                "type_prob": round(w.get("t_prob", 0.0), 4),
                            }
                        )
                        if len(detail_rows) >= cap.MAX_DETAIL_ROWS:
                            break
                sentence_counter += 1

    _apply_bi_consistency_rows(detail_rows)

    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    results = {}
    p, r, f1, _ = precision_recall_fscore_support(s1_true, s1_pred, average="binary", zero_division=0)
    results["step1_entity"] = {"precision": p, "recall": r, "f1": f1, "support": len(s1_true)}

    if s2_true:
        p, r, f1, _ = precision_recall_fscore_support(s2_true, s2_pred, average="binary", zero_division=0)
        results["step2_bio"] = {"precision": p, "recall": r, "f1": f1, "support": len(s2_true)}
    else:
        results["step2_bio"] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}

    if s3_true:
        acc = accuracy_score(s3_true, s3_pred)
        type_names = [entity_types[i] for i in sorted(set(s3_true))]
        report = classification_report(
            s3_true,
            s3_pred,
            labels=sorted(set(s3_true)),
            target_names=type_names,
            output_dict=True,
            zero_division=0,
        )
        results["step3_type"] = {"accuracy": acc, "per_type": report, "support": len(s3_true)}
    else:
        results["step3_type"] = {"accuracy": 0, "per_type": {}, "support": 0}

    sp, sr, sf = cap.span_f1(all_pred_spans, all_true_spans)
    results["pipeline_span"] = {"precision": sp, "recall": sr, "f1": sf}
    return results, detail_rows


def optimise_thresholds_crf(model, loader, device, entity_types, id_to_etype):
    """Same grid search as the base pipeline, unpacking the CRF model forward tuple."""
    best_t_ent, best_t_bio, best_score = 0.5, 0.5, -1.0
    model.eval()
    all_words = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            e_lab = batch["entity_labels"].to(device)
            b_lab = batch["bio_labels"].to(device)
            t_lab = batch["type_labels"].to(device)
            t_idx = batch["token_indices"].to(device)
            ent_logits, bio_logits, typ_logits, _ = model(ids, mask)
            for si in range(ids.size(0)):
                wl = cap.aggregate_to_words(
                    ent_logits[si],
                    bio_logits[si],
                    typ_logits[si],
                    e_lab[si],
                    b_lab[si],
                    t_lab[si],
                    t_idx[si],
                    batch["tokens"][si],
                    batch["bio_tags"][si],
                    batch["entity_types"][si],
                )
                all_words.extend(wl)
    if not all_words:
        return 0.5, 0.5
    from sklearn.metrics import precision_recall_fscore_support

    for te in cap.THRESHOLD_SWEEP:
        for tb in cap.THRESHOLD_SWEEP:
            e_true = [w["e_true"] for w in all_words]
            e_pred = [1 if w["e_prob"] >= te else 0 for w in all_words]
            b_true = [w["b_true"] for w in all_words if w["b_true"] != -100]
            b_pred = [1 if w["b_prob"] >= tb else 0 for w in all_words if w["b_true"] != -100]
            _, _, f1_e, _ = precision_recall_fscore_support(e_true, e_pred, average="binary", zero_division=0)
            if b_true:
                _, _, f1_b, _ = precision_recall_fscore_support(b_true, b_pred, average="binary", zero_division=0)
            else:
                f1_b = 0.0
            combined = f1_e + f1_b
            if combined > best_score:
                best_score = combined
                best_t_ent = te
                best_t_bio = tb
    return float(best_t_ent), float(best_t_bio)


def run_cascaded_crf_pipeline() -> str:
    print("=" * 70)
    print("Cascaded Multi-Step NER Pipeline (CRF joint tag head)")
    print("=" * 70)

    _presplit_train = os.environ.get("THESIS_PRESPLIT_TRAIN_JSON", "").strip()
    _presplit_eval = os.environ.get("THESIS_PRESPLIT_EVAL_JSON", "").strip()

    if _presplit_train and _presplit_eval and os.path.exists(_presplit_train) and os.path.exists(_presplit_eval):
        _raw_train = json.loads(open(_presplit_train, encoding="utf-8").read())
        _raw_eval = json.loads(open(_presplit_eval, encoding="utf-8").read())

        def _convert_presplit(raw_sentences):
            sents = []
            etype_set = set()
            tag_set = set()
            for sent in raw_sentences:
                tokens = str(sent.get("text", "")).split()
                labels = list(sent.get("labels", []))
                bios, etypes = [], []
                for label in labels:
                    label = str(label).strip()
                    tag_set.add(label)
                    if label.startswith("B-"):
                        bios.append("B")
                        et = label[2:]
                        etypes.append(et)
                        etype_set.add(et)
                    elif label.startswith("I-"):
                        bios.append("I")
                        et = label[2:]
                        etypes.append(et)
                        etype_set.add(et)
                    else:
                        bios.append("O")
                        etypes.append(None)
                if tokens:
                    sents.append({"tokens": tokens, "bio_tags": bios, "entity_types": etypes})
            return cap.SentenceDataset.from_list(sents), sorted(etype_set), sorted(tag_set)

        train_data, _train_etypes, train_tags = _convert_presplit(_raw_train)
        val_data, _val_etypes, val_tags = _convert_presplit(_raw_eval)
        test_data = cap.SentenceDataset()
        entity_types = sorted(set(_train_etypes) | set(_val_etypes))
        all_tags = sorted(set(train_tags) | set(val_tags) | {"O"})
    elif cap.DATA_SOURCE == "csv":
        csv_ds, entity_types = cap.load_csv_dataset(cap.DEFAULT_CSV_PATH)
        if cap.USE_FULL_DATASET:
            train_data, val_data, test_data = cap.split_dataset(
                csv_ds, val_frac=1 - cap.FULL_DATASET_TRAIN_FRACTION, test_frac=0.0, seed=cap.CSV_SHUFFLE_SEED
            )
        else:
            train_data, val_data, test_data = cap.split_dataset(
                csv_ds, val_frac=cap.CSV_SPLIT_VAL, test_frac=cap.CSV_SPLIT_TEST, seed=cap.CSV_SHUFFLE_SEED
            )
        all_tags = {"O"}
        for i in range(len(train_data)):
            sent = train_data[i]
            for bio, et in zip(sent["bio_tags"], sent["entity_types"]):
                if bio == "O":
                    all_tags.add("O")
                else:
                    all_tags.add(f"{bio}-{et}")
        all_tags = sorted(all_tags)
    else:
        raise ValueError(f"Unsupported DATA_SOURCE for CRF cascaded pipeline: {cap.DATA_SOURCE}")

    etype_to_id = {et: i for i, et in enumerate(entity_types)}
    id_to_etype = {i: et for et, i in etype_to_id.items()}
    tag_to_id = {t: i for i, t in enumerate(all_tags)}
    id_to_tag = {i: t for t, i in tag_to_id.items()}
    o_tag_id = tag_to_id.get("O")

    tokenizer = AutoTokenizer.from_pretrained(cap.BASE_MODEL_NAME, local_files_only=cap.MODEL_LOCAL_ONLY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = _collate_crf(tokenizer.pad_token_id)

    val_ds = CascadedNERDatasetCRF(val_data, tokenizer, etype_to_id, tag_to_id)
    val_loader = DataLoader(val_ds, batch_size=cap.TRAINING_CONFIG["eval_batch_size"], collate_fn=collate)
    train_ds = CascadedNERDatasetCRF(train_data, tokenizer, etype_to_id, tag_to_id)
    train_loader = DataLoader(
        train_ds, batch_size=cap.TRAINING_CONFIG["train_batch_size"], shuffle=True, collate_fn=collate
    )

    base_model = AutoModel.from_pretrained(cap.BASE_MODEL_NAME, local_files_only=cap.MODEL_LOCAL_ONLY)
    model = CascadedNERModelCRF(base_model, len(entity_types), len(all_tags), o_tag_id=o_tag_id).to(device)

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cap.TRAINING_CONFIG["encoder_lr"]},
            {"params": head_params, "lr": cap.TRAINING_CONFIG["head_lr"]},
        ],
        weight_decay=cap.TRAINING_CONFIG["weight_decay"],
    )

    total_steps = (len(train_loader) // cap.TRAINING_CONFIG["grad_accum_steps"]) * cap.TRAINING_CONFIG["epochs"]
    warmup_steps = int(total_steps * cap.TRAINING_CONFIG["warmup_fraction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    num_epochs = cap.TRAINING_CONFIG["epochs"]
    grad_accum = max(1, cap.TRAINING_CONFIG["grad_accum_steps"])
    max_grad_norm = cap.TRAINING_CONFIG["max_grad_norm"]
    patience = cap.TRAINING_CONFIG["early_stopping_patience"]
    min_delta = cap.TRAINING_CONFIG["early_stopping_min_delta"]
    best_monitored = float("-inf")
    patience_ctr = 0
    metrics_history = []
    details_by_mode = {}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch_crf(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            cap.LOSS_CONFIG,
            grad_accum,
            max_grad_norm,
        )
        monitored = None
        for use_oracle in cap.EVAL_CASCADE_MODES:
            mode_tag = cap.eval_mode_tag(use_oracle)
            res, _ = evaluate_crf(
                model,
                val_loader,
                device,
                entity_types,
                id_to_etype,
                id_to_tag,
                t_entity=0.5,
                t_bio=0.5,
                collect_details=False,
                use_oracle=use_oracle,
            )
            s1, s2, s3, sp = res["step1_entity"], res["step2_bio"], res["step3_type"], res["pipeline_span"]
            print(
                f"Epoch {epoch}/{num_epochs}  TrainLoss={train_loss:.4f}  [mode={mode_tag}]  "
                f"Pipeline F1={sp['f1']:.3f}"
            )
            metrics_history.append(
                {
                    "epoch": epoch,
                    "eval_mode": mode_tag,
                    "train_loss": train_loss,
                    "step1_entity_p": s1["precision"],
                    "step1_entity_r": s1["recall"],
                    "step1_entity_f1": s1["f1"],
                    "step2_bio_p": s2["precision"],
                    "step2_bio_r": s2["recall"],
                    "step2_bio_f1": s2["f1"],
                    "step3_type_acc": s3["accuracy"],
                    "pipeline_span_p": sp["precision"],
                    "pipeline_span_r": sp["recall"],
                    "pipeline_span_f1": sp["f1"],
                }
            )
            if mode_tag == "predicted":
                monitored = s1["f1"] + s2["f1"]

        if monitored is not None and monitored > best_monitored + min_delta:
            best_monitored = monitored
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    best_te, best_tb = optimise_thresholds_crf(model, val_loader, device, entity_types, id_to_etype)
    for use_oracle in cap.EVAL_CASCADE_MODES:
        mode_tag = cap.eval_mode_tag(use_oracle)
        res_opt, details_opt = evaluate_crf(
            model,
            val_loader,
            device,
            entity_types,
            id_to_etype,
            id_to_tag,
            t_entity=best_te,
            t_bio=best_tb,
            collect_details=True,
            use_oracle=use_oracle,
        )
        s1, s2, s3, sp = res_opt["step1_entity"], res_opt["step2_bio"], res_opt["step3_type"], res_opt["pipeline_span"]
        metrics_history.append(
            {
                "epoch": "final_optimised",
                "eval_mode": mode_tag,
                "train_loss": None,
                "step1_entity_p": s1["precision"],
                "step1_entity_r": s1["recall"],
                "step1_entity_f1": s1["f1"],
                "step2_bio_p": s2["precision"],
                "step2_bio_r": s2["recall"],
                "step2_bio_f1": s2["f1"],
                "step3_type_acc": s3["accuracy"],
                "pipeline_span_p": sp["precision"],
                "pipeline_span_r": sp["recall"],
                "pipeline_span_f1": sp["f1"],
                "threshold_entity": best_te,
                "threshold_bio": best_tb,
            }
        )
        details_by_mode[mode_tag] = details_opt

    df_metrics = pd.DataFrame(metrics_history)
    details_frames = []
    for mode_tag in [cap.eval_mode_tag(m) for m in cap.EVAL_CASCADE_MODES]:
        rows = details_by_mode.get(mode_tag, [])
        if rows:
            df_mode = pd.DataFrame(rows[: cap.MAX_DETAIL_ROWS])
            df_mode["eval_mode"] = mode_tag
            details_frames.append(df_mode)
    df_details = pd.concat(details_frames, ignore_index=True) if details_frames else pd.DataFrame()
    excel_path = os.path.join(os.path.dirname(__file__), "cascaded_pipeline_crf_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_metrics.to_excel(writer, sheet_name="metrics", index=False)
        if not df_details.empty:
            df_details.to_excel(writer, sheet_name="detailed_results", index=False)
    print(f"\nResults exported to {excel_path}")

    from core.model_cleanup import cleanup_training_artifacts_if_enabled

    cleanup_training_artifacts_if_enabled()

    return excel_path


if __name__ == "__main__":
    run_cascaded_crf_pipeline()
