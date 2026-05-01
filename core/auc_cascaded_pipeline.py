"""
auc_cascaded_pipeline.py — Cascaded Multi-Step NER Pipeline
============================================================

Decomposes NER into 3 independent sub-tasks.  Each step is evaluated using
ground-truth ("oracle") inputs from previous steps so that per-step accuracy
is measured in isolation.

  Step 1  Entity Detection   — Is this token part of an entity?  (binary)
  Step 2  BIO Position       — Is this entity-token a B or I?    (binary)
  Step 3  Entity Type        — What type of entity is this?      (multi-class)

A *full-pipeline* evaluation (no oracle) is also run to report end-to-end
span-level F1.

Improvements over the single-pass AUC-2T baseline
--------------------------------------------------
* *Masked (conditional) training*: B/I and type heads train ONLY on entity
  tokens, so they never see confusing O tokens.
* *Focal loss*: handles the severe class imbalance between B/I/O tokens.
* *Per-task threshold optimisation*: sweeps thresholds on the validation set.
* *Separate learning rates*: lower LR for the pre-trained encoder, higher LR
  for the randomly-initialised classification heads.
* *Span-level F1*: the standard NER metric (exact boundary + type match).
* *BIO constraint enforcement*: post-processing to fix invalid transitions.

Suggested further ideas (not yet implemented)
----------------------------------------------
 1. CRF layer — learn valid BIO transition scores jointly with the model.
 2. Span prediction — predict (start, end, type) instead of per-token tags.
 3. Gazetteer features — inject known entity lists as binary features.
 4. Self-training — iteratively pseudo-label unlabelled data and retrain.
 5. Character-level CNN/LSTM — useful for morphologically-rich Hebrew.
 6. Cross-lingual transfer — pre-train on English NER, adapt to Hebrew.
 7. Ensemble — average predictions from multiple seeds.
 8. Data augmentation — entity swapping, sentence cropping, token dropout.
 9. Label smoothing on the type cross-entropy loss.
10. Active learning — select the most informative samples for annotation.
11. Curriculum learning — order training examples easy → hard.
12. Auxiliary tasks — POS tagging or dependency parsing sharing the encoder.
"""

# ============================================================================
# Imports
# ============================================================================
from datasets import load_dataset
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
import pandas as pd
from collections import defaultdict, OrderedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Proxy
# ============================================================================
if any(
    os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
):
    pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CSV_PATH = os.environ.get(
    "THESIS_NER_CSV",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ner_dataset.csv"),
)
CSV_SPLIT_VAL = 0.1
CSV_SPLIT_TEST = 0.1
CSV_SHUFFLE_SEED = 42
DATA_SOURCE = "csv"               # "csv" or "conll"
USE_FULL_DATASET = True
FULL_DATASET_TRAIN_FRACTION = 0.7
BASE_MODEL_NAME = "dicta-il/dictabert"
_INTERNAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dictabert")
if os.path.exists(os.path.join(_INTERNAL_MODEL, "config.json")):
    BASE_MODEL_NAME = _INTERNAL_MODEL
BASE_MODEL_NAME = os.environ.get("THESIS_MODEL_NAME", BASE_MODEL_NAME)
MODEL_LOCAL_ONLY = os.environ.get("THESIS_MODEL_LOCAL_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_LENGTH = 128
MAX_DETAIL_ROWS = 100_000

CSV_ENCODING_PREFERENCE = [
    "utf-8", "utf-8-sig", "cp1255", "windows-1255", "iso-8859-8", "cp1252",
]

# CoNLL-2003 tag ↔ (BIO, entity-type) lookup
CONLL_TAG_MAP = {
    0: ("O", None),
    1: ("B", "PER"),  2: ("I", "PER"),
    3: ("B", "ORG"),  4: ("I", "ORG"),
    5: ("B", "LOC"),  6: ("I", "LOC"),
    7: ("B", "MISC"), 8: ("I", "MISC"),
}

TRAINING_CONFIG = {
    "epochs": 10,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "encoder_lr": 2e-5,        # lower LR for pre-trained encoder
    "head_lr": 1e-3,           # higher LR for new classification heads
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,    # fraction of total steps for LR warm-up
    "grad_accum_steps": 2,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-3,
}

LOSS_CONFIG = {
    "entity_loss": "focal",    # "focal" | "bce" | "auc"
    "bio_loss": "focal",       # "focal" | "bce" | "auc"
    "type_loss": "ce",         # cross-entropy
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "lambda_bio": 10.0,
    "lambda_type": 5.0,
}

THRESHOLD_SWEEP = np.arange(0.10, 0.91, 0.05)

# ---------------------------------------------------------------------------
# Evaluation modes used after each epoch.
#
# False  -> "predicted" mode
#           Step 2 and Step 3 are evaluated only on tokens that Step 1
#           predicted as entity tokens. This simulates the real pipeline,
#           where errors from Step 1 propagate into later steps.
#
# True   -> "oracle" mode
#           Step 2 and Step 3 are evaluated on the ground-truth entity tokens.
#           This isolates the quality of each later step by removing Step 1
#           errors from the measurement.
#
# Order matters here only for printing/export. We run predicted first and then
# oracle so the output naturally shows:
#   1) realistic end-to-end behavior
#   2) idealized per-step capability
# ---------------------------------------------------------------------------
EVAL_CASCADE_MODES = [False, True]

# Optional performance tuning (opt-in; defaults preserve original behavior)
if _env_flag("THESIS_EXP04_FAST", default=False):
    TRAINING_CONFIG["epochs"] = min(TRAINING_CONFIG["epochs"], 4)
    TRAINING_CONFIG["early_stopping_patience"] = min(TRAINING_CONFIG["early_stopping_patience"], 2)
    THRESHOLD_SWEEP = np.arange(0.20, 0.81, 0.10)
    EVAL_CASCADE_MODES = [False]

TRAINING_CONFIG["epochs"] = _env_int("THESIS_EXP04_EPOCHS", TRAINING_CONFIG["epochs"], minimum=1)
TRAINING_CONFIG["train_batch_size"] = _env_int("THESIS_EXP04_TRAIN_BATCH", TRAINING_CONFIG["train_batch_size"], minimum=1)
TRAINING_CONFIG["eval_batch_size"] = _env_int("THESIS_EXP04_EVAL_BATCH", TRAINING_CONFIG["eval_batch_size"], minimum=1)
TRAINING_CONFIG["grad_accum_steps"] = _env_int("THESIS_EXP04_GRAD_ACCUM", TRAINING_CONFIG["grad_accum_steps"], minimum=1)

_thr_min = _env_float("THESIS_EXP04_THRESHOLD_MIN", float(THRESHOLD_SWEEP[0]), minimum=0.0, maximum=1.0)
_thr_max = _env_float("THESIS_EXP04_THRESHOLD_MAX", float(THRESHOLD_SWEEP[-1]), minimum=0.0, maximum=1.0)
_thr_step = _env_float("THESIS_EXP04_THRESHOLD_STEP", 0.05, minimum=0.01, maximum=1.0)
if _thr_max < _thr_min:
    _thr_min, _thr_max = _thr_max, _thr_min
THRESHOLD_SWEEP = np.arange(_thr_min, _thr_max + (_thr_step * 0.5), _thr_step)

if _env_flag("THESIS_EXP04_SKIP_ORACLE_EVAL", default=False):
    EVAL_CASCADE_MODES = [False]


def eval_mode_tag(use_oracle):
    """Convert the boolean evaluation flag into a readable label."""
    return "oracle" if use_oracle else "predicted"

# ============================================================================
# Lightweight sentence container (same idea as original)
# ============================================================================
class SentenceDataset:
    """List wrapper with select / shuffle / split utilities."""

    def __init__(self, data=None):
        self._data = list(data) if data else []

    @classmethod
    def from_list(cls, items):
        return cls(items)

    def to_list(self):
        return list(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def select(self, indices):
        idx = list(indices)
        return SentenceDataset([self._data[i] for i in idx if 0 <= i < len(self._data)])

    def shuffle(self, seed):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self._data))
        rng.shuffle(idx)
        return SentenceDataset([self._data[i] for i in idx])

    def train_test_split(self, test_size, seed):
        shuffled = self.shuffle(seed)
        n = max(1, int(round(len(shuffled) * test_size)))
        return {"train": SentenceDataset(shuffled._data[n:]),
                "test":  SentenceDataset(shuffled._data[:n])}


# ============================================================================
# Data loading — unified format: {tokens, bio_tags, entity_types}
# ============================================================================
def _read_csv_with_fallback(csv_path, encoding_override=None):
    candidates = []
    if encoding_override:
        if isinstance(encoding_override, (list, tuple)):
            candidates.extend(encoding_override)
        else:
            candidates.append(encoding_override)
    candidates.extend(CSV_ENCODING_PREFERENCE)
    seen, errors = set(), {}
    for enc in candidates:
        if not enc or enc in seen:
            continue
        seen.add(enc)
        try:
            return pd.read_csv(csv_path, encoding=enc), enc
        except UnicodeDecodeError as e:
            errors[enc] = str(e)
    raise UnicodeError(f"Cannot decode {csv_path} with {list(seen)}: {errors}")


def merge_wordpieces(tokens, bio_tags, entity_types, preview_state=None):
    """Merge ##-prefix wordpieces back into whole words (keep first-piece label)."""
    m_tok, m_bio, m_etype = [], [], []
    for tok, bio, etype in zip(tokens, bio_tags, entity_types):
        if tok.startswith("##") and m_tok:
            m_tok[-1] += tok[2:]
        else:
            m_tok.append(tok)
            m_bio.append(bio)
            m_etype.append(etype)
    if preview_state is not None and not preview_state.get("shown") and m_tok != tokens:
        preview_state["shown"] = True
    return m_tok, m_bio, m_etype


def load_csv_dataset(csv_path, encoding_override=None):
    """Return (SentenceDataset, entity_type_list)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df, enc = _read_csv_with_fallback(csv_path, encoding_override)
    print(f"CSV loaded ({enc})")
    for col in ("id", "token", "raw_tags"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df["token"]    = df["token"].fillna("")
    df["raw_tags"] = df["raw_tags"].fillna("O")

    sentences, etype_set = [], set()
    special = {"[CLS]", "[SEP]"}
    preview = {"shown": False}

    for _, grp in df.groupby("id", sort=False):
        toks, bios, etypes = [], [], []
        for tok, raw in zip(grp["token"], grp["raw_tags"]):
            t = str(tok).strip()
            if not t or t in special:
                continue
            raw = str(raw).strip()
            if raw.startswith("B-"):
                bios.append("B"); et = raw[2:]; etypes.append(et); etype_set.add(et)
            elif raw.startswith("I-"):
                bios.append("I"); et = raw[2:]; etypes.append(et); etype_set.add(et)
            else:
                bios.append("O"); etypes.append(None)
            toks.append(t)
        if toks:
            toks, bios, etypes = merge_wordpieces(toks, bios, etypes, preview)
            sentences.append({"tokens": toks, "bio_tags": bios, "entity_types": etypes})

    if not sentences:
        raise ValueError("No sentences parsed from CSV")
    etype_list = sorted(etype_set)
    print(f"Loaded {len(sentences)} CSV sentences, entity types: {etype_list}")
    return SentenceDataset.from_list(sentences), etype_list


def load_conll_dataset():
    """Return (train, val, test) SentenceDatasets and entity_type_list."""
    ds = load_dataset("conll2003", cache_dir="./data_cache")
    etype_list = sorted({v[1] for v in CONLL_TAG_MAP.values() if v[1]})

    def _convert(split):
        sents = []
        for row in split:
            toks = row["tokens"]
            bios, etypes = [], []
            for tag_id in row["ner_tags"]:
                bio, etype = CONLL_TAG_MAP.get(tag_id, ("O", None))
                bios.append(bio)
                etypes.append(etype)
            sents.append({"tokens": toks, "bio_tags": bios, "entity_types": etypes})
        return SentenceDataset.from_list(sents)

    return _convert(ds["train"]), _convert(ds["validation"]), _convert(ds["test"]), etype_list


def split_dataset(full, val_frac, test_frac, seed):
    shuffled = full.shuffle(seed)
    if test_frac > 0:
        sp = shuffled.train_test_split(test_frac, seed)
        rest, test = sp["train"], sp["test"]
    else:
        rest, test = shuffled, SentenceDataset()
    if val_frac > 0:
        rel = val_frac / (1 - test_frac) if (1 - test_frac) > 0 else 0
        sp2 = rest.train_test_split(rel, seed)
        train, val = sp2["train"], sp2["test"]
    else:
        train, val = rest, SentenceDataset()
    return train, val, test


# ============================================================================
# PyTorch Dataset — produces per-token labels for all 3 tasks
# ============================================================================
class CascadedNERDataset(TorchDataset):
    """
    Labels produced per token:
        entity_label : 1 = entity, 0 = O,  -100 = special/padding
        bio_label    : 1 = B,      0 = I,  -100 = O / special
        type_label   : entity-type index,   -100 = O / special
    """

    def __init__(self, data, tokenizer, etype_to_id, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.etype_to_id = etype_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens       = sample["tokens"]
        bio_tags     = sample["bio_tags"]
        entity_types = sample["entity_types"]

        # Word-level numeric labels
        w_entity, w_bio, w_type = [], [], []
        for bio, etype in zip(bio_tags, entity_types):
            if bio == "O":
                w_entity.append(0);  w_bio.append(-100); w_type.append(-100)
            elif bio == "B":
                w_entity.append(1);  w_bio.append(1);    w_type.append(self.etype_to_id.get(etype, 0))
            else:  # I
                w_entity.append(1);  w_bio.append(0);    w_type.append(self.etype_to_id.get(etype, 0))

        enc = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        word_ids = enc.word_ids()

        a_entity, a_bio, a_type, a_widx = [], [], [], []
        for wi in word_ids:
            if wi is None:
                a_entity.append(-100); a_bio.append(-100); a_type.append(-100); a_widx.append(-1)
            else:
                a_entity.append(w_entity[wi]); a_bio.append(w_bio[wi])
                a_type.append(w_type[wi]);     a_widx.append(wi)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask":  enc["attention_mask"].squeeze(0),
            "entity_labels":  torch.tensor(a_entity, dtype=torch.long),
            "bio_labels":     torch.tensor(a_bio,    dtype=torch.long),
            "type_labels":    torch.tensor(a_type,   dtype=torch.long),
            "token_indices":  torch.tensor(a_widx,   dtype=torch.long),
            "tokens":         tokens,
            "bio_tags":       bio_tags,
            "entity_types":   entity_types,
        }


# ============================================================================
# Model — shared encoder, 3 independent heads
# ============================================================================
class CascadedNERModel(nn.Module):
    def __init__(self, base_model, num_entity_types, dropout=0.1):
        super().__init__()
        h = base_model.config.hidden_size
        self.encoder = base_model
        self.entity_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))
        self.bio_head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))
        self.type_head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, num_entity_types))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return (
            self.entity_head(h).squeeze(-1),   # (B, L)
            self.bio_head(h).squeeze(-1),      # (B, L)
            self.type_head(h),                 # (B, L, C)
        )


# ============================================================================
# Loss functions
# ============================================================================
class FocalLoss(nn.Module):
    """Binary focal loss — reduces contribution of easy-to-classify tokens."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets_f + (1 - p) * (1 - targets_f)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class AUCMarginLoss(nn.Module):
    """Pairwise squared-hinge AUC surrogate (from original script)."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, targets):
        # targets: 0 or 1
        pos = logits[targets == 1]
        neg = logits[targets == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return (logits * 0).sum()
        pairwise = self.margin - pos.unsqueeze(1) + neg.unsqueeze(0)
        return pairwise.pow(2).mean()


def get_binary_loss_fn(name, cfg):
    if name == "focal":
        return FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "auc":
        return AUCMarginLoss(margin=1.0)
    raise ValueError(f"Unknown loss: {name}")


# ============================================================================
# Collate
# ============================================================================
def make_collate_fn(pad_token_id):
    def collate(batch):
        mx = max(x["input_ids"].size(0) for x in batch)
        ids, att, ent, bio, typ, widx = [], [], [], [], [], []
        tok_list, bio_list, etype_list = [], [], []
        for x in batch:
            L = x["input_ids"].size(0)
            p = mx - L
            ids.append(F.pad(x["input_ids"],      (0, p), value=pad_token_id))
            att.append(F.pad(x["attention_mask"],  (0, p), value=0))
            ent.append(F.pad(x["entity_labels"],   (0, p), value=-100))
            bio.append(F.pad(x["bio_labels"],      (0, p), value=-100))
            typ.append(F.pad(x["type_labels"],     (0, p), value=-100))
            widx.append(F.pad(x["token_indices"],  (0, p), value=-1))
            tok_list.append(x["tokens"])
            bio_list.append(x["bio_tags"])
            etype_list.append(x["entity_types"])
        return {
            "input_ids":      torch.stack(ids),
            "attention_mask":  torch.stack(att),
            "entity_labels":  torch.stack(ent),
            "bio_labels":     torch.stack(bio),
            "type_labels":    torch.stack(typ),
            "token_indices":  torch.stack(widx),
            "tokens":         tok_list,
            "bio_tags":       bio_list,
            "entity_types":   etype_list,
        }
    return collate


# ============================================================================
# Training one epoch
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, device, cfg_loss, grad_accum, max_grad_norm):
    """Train the model for one full pass over the training loader.

    Beginner-friendly view of what happens here:

    1. Put the model in training mode.
        This enables behaviors such as dropout.

    2. Loop over mini-batches from the training data.
        Each batch contains tokenized sentences and labels for the 3 tasks:
        - entity detection
        - B/I prediction
        - entity type prediction

    3. Run a forward pass.
        The encoder produces contextual token representations, then the three
        heads produce logits for the three sub-tasks.

    4. Compute losses.
        - Step 1 loss uses all valid tokens.
        - Step 2 and Step 3 losses are computed only on ground-truth entity
          tokens (masked training). This is important because we do not want
          O-tokens to dominate the learning signal for B/I and type.

    5. Backpropagate the combined loss.
        We optionally divide by `grad_accum` so we can simulate a larger batch
        size by accumulating gradients across several smaller batches.

    6. Every `grad_accum` steps, update the weights.
        We clip gradients for stability, take an optimizer step, update the
        learning-rate scheduler, and reset gradients.

    7. Return the average training loss for the epoch.
    """
    model.train()
    entity_loss_fn = get_binary_loss_fn(cfg_loss["entity_loss"], cfg_loss)
    bio_loss_fn    = get_binary_loss_fn(cfg_loss["bio_loss"], cfg_loss)
    lam_bio  = cfg_loss["lambda_bio"]
    lam_type = cfg_loss["lambda_type"]
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        # Move the current mini-batch to the selected device (CPU/GPU).
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        e_lab = batch["entity_labels"].to(device)
        b_lab = batch["bio_labels"].to(device)
        t_lab = batch["type_labels"].to(device)

        # Forward pass: one shared encoder + three task-specific heads.
        ent_logits, bio_logits, typ_logits = model(ids, mask)

        # --- Step 1 loss: entity detection on all valid tokens ---
        # Special/padding tokens use label -100 and are ignored.
        valid = e_lab != -100
        if valid.sum() == 0:
            continue
        loss = entity_loss_fn(ent_logits[valid], e_lab[valid].float())

        # --- Step 2 loss: B vs I on ground-truth entity tokens only ---
        # Only real entity tokens participate here. O-tokens are masked out.
        bio_valid = b_lab != -100
        if bio_valid.sum() > 0:
            loss = loss + lam_bio * bio_loss_fn(bio_logits[bio_valid], b_lab[bio_valid].float())

        # --- Step 3 loss: entity type on ground-truth entity tokens only ---
        # Again, only entity tokens should influence this classifier.
        typ_valid = t_lab != -100
        if typ_valid.sum() > 0:
            loss = loss + lam_type * F.cross_entropy(typ_logits[typ_valid], t_lab[typ_valid])

        # Backpropagation. Divide by grad_accum so accumulated gradients match
        # the scale of one larger batch.
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if step % grad_accum == 0 or step == len(loader):
            # Gradient clipping protects training from unstable large updates.
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(1, len(loader))


# ============================================================================
# Span extraction helpers
# ============================================================================
def extract_spans(bio_tags, entity_types):
    """Extract (start, end_exclusive, entity_type) spans from parallel lists."""
    spans = set()
    i = 0
    while i < len(bio_tags):
        if bio_tags[i] == "B" and entity_types[i] is not None:
            et = entity_types[i]
            start = i
            i += 1
            while i < len(bio_tags) and bio_tags[i] == "I" and entity_types[i] == et:
                i += 1
            spans.add((start, i, et))
        else:
            i += 1
    return spans


def span_f1(pred_spans, true_spans):
    tp = len(pred_spans & true_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def enforce_bio_constraints(bio_tags):
    """First entity token after O must be B; consecutive entities must not
    start with I unless preceded by a same-type B/I (simple heuristic)."""
    out = list(bio_tags)
    for i, tag in enumerate(out):
        if tag == "I" and (i == 0 or out[i - 1] == "O"):
            out[i] = "B"
    return out


# ============================================================================
# Word-level aggregation (merge sub-tokens → words)
# ============================================================================
def aggregate_to_words(
    entity_logits,  # (L,) raw logits
    bio_logits,     # (L,)
    type_logits,    # (L, C)
    entity_labels,  # (L,) ground truth
    bio_labels,     # (L,)
    type_labels,    # (L,)
    token_indices,  # (L,)
    orig_tokens,    # list[str]
    orig_bio_tags,  # list[str]
    orig_entity_types,  # list[str|None]
    t_entity=0.5,
    t_bio=0.5,
):
    """Aggregate sub-token predictions back to word level.

    Returns a list of dicts (one per word) with predictions and ground truth.
    """
    words = OrderedDict()  # word_idx → stats
    for i in range(entity_logits.size(0)):
        wi = token_indices[i].item()
        el = entity_labels[i].item()
        if wi < 0 or el == -100:
            continue
        if wi not in words:
            words[wi] = {
                "e_logits": [], "b_logits": [], "t_logits": [],
                "e_lab": el,
                "b_lab": bio_labels[i].item(),
                "t_lab": type_labels[i].item(),
            }
        words[wi]["e_logits"].append(entity_logits[i].item())
        words[wi]["b_logits"].append(bio_logits[i].item())
        words[wi]["t_logits"].append(type_logits[i].cpu().numpy())

    results = []
    for wi, w in words.items():
        # Entity: max prob across sub-tokens
        e_prob = max(torch.sigmoid(torch.tensor(w["e_logits"])).tolist())
        e_pred = 1 if e_prob >= t_entity else 0

        # B/I: first sub-token (positional)
        b_prob = torch.sigmoid(torch.tensor(w["b_logits"][0])).item()
        b_pred = 1 if b_prob >= t_bio else 0

        # Type: first sub-token argmax
        t_logits_first = np.asarray(w["t_logits"][0], dtype=np.float64)
        t_stable = t_logits_first - np.max(t_logits_first)
        t_probs = np.exp(t_stable) / np.sum(np.exp(t_stable))
        t_pred = int(np.argmax(t_probs))
        t_prob = float(t_probs[t_pred])

        # Ground truth
        tok = orig_tokens[wi] if wi < len(orig_tokens) else ""
        true_bio = orig_bio_tags[wi] if wi < len(orig_bio_tags) else "O"
        true_etype = orig_entity_types[wi] if wi < len(orig_entity_types) else None

        results.append({
            "word_idx": wi,
            "token": tok,
            # predictions
            "e_prob": e_prob, "e_pred": e_pred,
            "b_prob": b_prob, "b_pred": b_pred,
            "t_pred": t_pred, "t_prob": t_prob,
            # ground truth
            "e_true": w["e_lab"],
            "b_true": w["b_lab"],   # 1=B, 0=I, -100=O
            "t_true": w["t_lab"],   # type idx or -100
            "true_bio": true_bio,
            "true_etype": true_etype,
        })
    return results


# ============================================================================
# Evaluate — oracle cascade + full pipeline
# ============================================================================
def evaluate(
    model, loader, device, entity_types, id_to_etype,
    t_entity=0.5, t_bio=0.5, collect_details=True,
    use_oracle=True,
):
    """Evaluate the trained model on the validation loader.

    This function supports two teaching-friendly evaluation styles:

    `use_oracle=False`  (predicted mode)
        This is the realistic pipeline view.
        Step 1 first decides which tokens are entities.
        Then Step 2 and Step 3 are evaluated only on those predicted-entity
        tokens. Mistakes from Step 1 therefore affect later steps.

    `use_oracle=True`   (oracle mode)
        This is the diagnostic view.
        Step 2 and Step 3 are evaluated on the ground-truth entity tokens.
        That means we are asking:
        "If Step 1 had been perfect, how good are the later steps?"

    Why both are useful:
    - Predicted mode tells us how the full system behaves in practice.
    - Oracle mode tells us whether later components are intrinsically good,
      even if Step 1 is still weak.
    """
    model.eval()

    # Collectors — oracle per-step
    s1_true, s1_pred = [], []               # entity detection
    s2_true, s2_pred = [], []               # B vs I (oracle entity mask)
    s3_true, s3_pred = [], []               # entity type  (oracle entity mask)
    # Full pipeline spans
    all_true_spans, all_pred_spans = set(), set()
    span_offset = 0
    detail_rows = []

    with torch.no_grad():
        sentence_counter = 0
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            e_lab = batch["entity_labels"].to(device)
            b_lab = batch["bio_labels"].to(device)
            t_lab = batch["type_labels"].to(device)
            t_idx = batch["token_indices"].to(device)
            tok_lists   = batch["tokens"]
            bio_lists   = batch["bio_tags"]
            etype_lists = batch["entity_types"]

            ent_logits, bio_logits, typ_logits = model(ids, mask)

            batch_size = ids.size(0)
            for si in range(batch_size):
                sentence_id = sentence_counter + 1
                # Convert sub-token outputs back to original word-level tokens.
                # This is easier to interpret and matches the dataset labels.
                wl = aggregate_to_words(
                    ent_logits[si], bio_logits[si], typ_logits[si],
                    e_lab[si], b_lab[si], t_lab[si], t_idx[si],
                    tok_lists[si], bio_lists[si], etype_lists[si],
                    t_entity=t_entity, t_bio=t_bio,
                )
                if not wl:
                    sentence_counter += 1
                    span_offset += 1
                    continue

                # ---- Step 1: entity detection (all tokens) ----
                for w in wl:
                    s1_true.append(w["e_true"])
                    s1_pred.append(w["e_pred"])

                # ---- Step 2: B vs I ----
                # Oracle mode:
                #   use the true entity mask, so Step 2 is judged in isolation.
                # Predicted mode:
                #   use Step 1 predictions, so this reflects real pipeline usage.
                for w in wl:
                    if use_oracle:
                        if w["b_true"] != -100:        # GT says entity
                            s2_true.append(w["b_true"])
                            s2_pred.append(w["b_pred"])
                    else:
                        if w["e_pred"] == 1:           # Step 1 predicted entity
                            # True label: use GT if available, else treat as 0 (non-B)
                            s2_true.append(w["b_true"] if w["b_true"] != -100 else 0)
                            s2_pred.append(w["b_pred"])

                # ---- Step 3: entity type ----
                # In oracle mode we test type prediction on all true entities.
                # In predicted mode we only test tokens predicted as entities,
                # and only when they are truly entities so the target label is
                # well-defined.
                for w in wl:
                    if use_oracle:
                        if w["t_true"] != -100:
                            s3_true.append(w["t_true"])
                            s3_pred.append(w["t_pred"])
                    else:
                        if w["e_pred"] == 1 and w["t_true"] != -100:
                            # Evaluate type quality only where a predicted-entity is truly an entity.
                            # This keeps type evaluation well-defined in non-oracle mode.
                            s3_true.append(w["t_true"])
                            s3_pred.append(w["t_pred"])

                # ---- Full pipeline: build predicted BIO+type tags ----
                # This is always the real end-to-end prediction, independent of
                # oracle/predicted evaluation mode above.
                pred_bio, pred_etype = [], []
                true_bio, true_etype = [], []
                for w in wl:
                    # Ground truth
                    true_bio.append(w["true_bio"])
                    true_etype.append(w["true_etype"])
                    # Pipeline prediction
                    if w["e_pred"] == 1:       # model says entity
                        pb = "B" if w["b_pred"] == 1 else "I"
                        pt = id_to_etype.get(w["t_pred"], entity_types[0])
                    else:
                        pb = "O"
                        pt = None
                    pred_bio.append(pb)
                    pred_etype.append(pt)

                pred_bio = enforce_bio_constraints(pred_bio)

                # Span extraction (offset spans so they're globally unique)
                ts = extract_spans(true_bio, true_etype)
                ps = extract_spans(pred_bio, pred_etype)
                for (s, e, et) in ts:
                    all_true_spans.add((s + span_offset, e + span_offset, et))
                for (s, e, et) in ps:
                    all_pred_spans.add((s + span_offset, e + span_offset, et))
                span_offset += len(wl) + 1

                # Detail rows
                if collect_details and len(detail_rows) < MAX_DETAIL_ROWS:
                    for w, pb, pt in zip(wl, pred_bio, pred_etype):
                        detail_rows.append({
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
                        })
                        if len(detail_rows) >= MAX_DETAIL_ROWS:
                            break
                sentence_counter += 1

    # ---- Compute metrics ----
    results = {}

    # Step 1
    p, r, f1, _ = precision_recall_fscore_support(s1_true, s1_pred, average="binary", zero_division=0)
    results["step1_entity"] = {"precision": p, "recall": r, "f1": f1, "support": len(s1_true)}

    # Step 2
    if s2_true:
        p, r, f1, _ = precision_recall_fscore_support(s2_true, s2_pred, average="binary", zero_division=0)
        results["step2_bio"] = {"precision": p, "recall": r, "f1": f1, "support": len(s2_true)}
    else:
        results["step2_bio"] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}

    # Step 3
    if s3_true:
        acc = accuracy_score(s3_true, s3_pred)
        # Per-type report
        type_names = [entity_types[i] for i in sorted(set(s3_true))]
        report = classification_report(
            s3_true, s3_pred,
            labels=sorted(set(s3_true)),
            target_names=type_names,
            output_dict=True, zero_division=0,
        )
        results["step3_type"] = {"accuracy": acc, "per_type": report, "support": len(s3_true)}
    else:
        results["step3_type"] = {"accuracy": 0, "per_type": {}, "support": 0}

    # Full pipeline span-level F1
    sp, sr, sf = span_f1(all_pred_spans, all_true_spans)
    results["pipeline_span"] = {"precision": sp, "recall": sr, "f1": sf}

    return results, detail_rows


# ============================================================================
# Threshold optimisation
# ============================================================================
def optimise_thresholds(model, loader, device, entity_types, id_to_etype):
    """Grid-search over entity and bio thresholds on validation data."""
    best_t_ent, best_t_bio, best_score = 0.5, 0.5, -1.0

    # Precompute all word-level data once
    model.eval()
    all_words = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            e_lab = batch["entity_labels"].to(device)
            b_lab = batch["bio_labels"].to(device)
            t_lab = batch["type_labels"].to(device)
            t_idx = batch["token_indices"].to(device)

            ent_logits, bio_logits, typ_logits = model(ids, mask)

            for si in range(ids.size(0)):
                wl = aggregate_to_words(
                    ent_logits[si], bio_logits[si], typ_logits[si],
                    e_lab[si], b_lab[si], t_lab[si], t_idx[si],
                    batch["tokens"][si], batch["bio_tags"][si], batch["entity_types"][si],
                )
                all_words.extend(wl)

    if not all_words:
        return 0.5, 0.5

    print(f"Threshold sweep over {len(THRESHOLD_SWEEP)} × {len(THRESHOLD_SWEEP)} grid …")
    for te in THRESHOLD_SWEEP:
        for tb in THRESHOLD_SWEEP:
            # Quick token-level entity F1 + bio F1
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

    print(f"  Best thresholds: entity={best_t_ent:.2f}  bio={best_t_bio:.2f}  (combined F1={best_score:.4f})")
    return float(best_t_ent), float(best_t_bio)


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # ======================================================================
    # High-level script flow
    #
    # 1. Load and split the dataset.
    # 2. Build tokenizers, datasets, dataloaders, model, optimizer, scheduler.
    # 3. Train the model epoch by epoch.
    # 4. After each epoch, evaluate in BOTH modes:
    #       - predicted mode  (realistic pipeline)
    #       - oracle mode     (isolated later-step quality)
    # 5. Apply early stopping based on predicted-mode performance.
    # 6. After training, search for better thresholds for Step 1 and Step 2.
    # 7. Run final evaluation again in both modes.
    # 8. Export all results into one Excel file.
    #
    # This structure is useful in teaching because it separates:
    #   training,
    #   diagnostic evaluation,
    #   realistic evaluation,
    #   and reporting/export.
    # ======================================================================
    print("=" * 70)
    print("Cascaded Multi-Step NER Pipeline")
    print("=" * 70)

    pass

    # ---- Load data (supports pre-split JSON from experiment 07) ----
    _presplit_train = os.environ.get("THESIS_PRESPLIT_TRAIN_JSON", "").strip()
    _presplit_eval = os.environ.get("THESIS_PRESPLIT_EVAL_JSON", "").strip()

    if _presplit_train and _presplit_eval and os.path.exists(_presplit_train) and os.path.exists(_presplit_eval):
        # Load pre-computed splits produced by experiment 07
        import json as _json
        _raw_train = _json.loads(open(_presplit_train, encoding="utf-8").read())
        _raw_eval = _json.loads(open(_presplit_eval, encoding="utf-8").read())

        def _convert_presplit(raw_sentences):
            sents = []
            etype_set = set()
            for sent in raw_sentences:
                tokens = str(sent.get("text", "")).split()
                labels = list(sent.get("labels", []))
                bios, etypes = [], []
                for label in labels:
                    label = str(label).strip()
                    if label.startswith("B-"):
                        bios.append("B"); et = label[2:]; etypes.append(et); etype_set.add(et)
                    elif label.startswith("I-"):
                        bios.append("I"); et = label[2:]; etypes.append(et); etype_set.add(et)
                    else:
                        bios.append("O"); etypes.append(None)
                if tokens:
                    sents.append({"tokens": tokens, "bio_tags": bios, "entity_types": etypes})
            return SentenceDataset.from_list(sents), sorted(etype_set)

        train_data, _train_etypes = _convert_presplit(_raw_train)
        val_data, _val_etypes = _convert_presplit(_raw_eval)
        test_data = SentenceDataset()
        entity_types = sorted(set(_train_etypes) | set(_val_etypes))
        print(f"Pre-split loaded: train={len(train_data)} val={len(val_data)} entity_types={entity_types}")
    elif DATA_SOURCE == "conll":
        train_data, val_data, test_data, entity_types = load_conll_dataset()
        print(f"CoNLL-2003: train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    elif DATA_SOURCE == "csv":
        csv_ds, entity_types = load_csv_dataset(DEFAULT_CSV_PATH)
        if USE_FULL_DATASET:
            train_data, val_data, test_data = split_dataset(
                csv_ds, val_frac=1 - FULL_DATASET_TRAIN_FRACTION, test_frac=0.0, seed=CSV_SHUFFLE_SEED,
            )
        else:
            train_data, val_data, test_data = split_dataset(
                csv_ds, val_frac=CSV_SPLIT_VAL, test_frac=CSV_SPLIT_TEST, seed=CSV_SHUFFLE_SEED,
            )
        print(f"CSV split: train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    else:
        raise ValueError(f"Unsupported DATA_SOURCE: {DATA_SOURCE}")

    etype_to_id = {et: i for i, et in enumerate(entity_types)}
    id_to_etype = {i: et for et, i in etype_to_id.items()}
    num_etypes = len(entity_types)
    print(f"Entity types ({num_etypes}): {entity_types}")

    # ---- Tokenizer & device ----
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, local_files_only=MODEL_LOCAL_ONLY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    collate = make_collate_fn(tokenizer.pad_token_id)

    val_ds = CascadedNERDataset(val_data, tokenizer, etype_to_id)
    val_loader = DataLoader(val_ds, batch_size=TRAINING_CONFIG["eval_batch_size"], collate_fn=collate)

    train_ds = CascadedNERDataset(train_data, tokenizer, etype_to_id)
    train_loader = DataLoader(
        train_ds, batch_size=TRAINING_CONFIG["train_batch_size"], shuffle=True, collate_fn=collate,
    )

    # ---- Model ----
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, local_files_only=MODEL_LOCAL_ONLY)
    model = CascadedNERModel(base_model, num_etypes).to(device)

    # ---- Optimizer with separate LR groups ----
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.entity_head.parameters())
        + list(model.bio_head.parameters())
        + list(model.type_head.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": TRAINING_CONFIG["encoder_lr"]},
        {"params": head_params,    "lr": TRAINING_CONFIG["head_lr"]},
    ], weight_decay=TRAINING_CONFIG["weight_decay"])

    total_steps = (len(train_loader) // TRAINING_CONFIG["grad_accum_steps"]) * TRAINING_CONFIG["epochs"]
    warmup_steps = int(total_steps * TRAINING_CONFIG["warmup_fraction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    num_epochs = TRAINING_CONFIG["epochs"]
    grad_accum = max(1, TRAINING_CONFIG["grad_accum_steps"])
    max_grad_norm = TRAINING_CONFIG["max_grad_norm"]
    patience = TRAINING_CONFIG["early_stopping_patience"]
    min_delta = TRAINING_CONFIG["early_stopping_min_delta"]
    best_monitored = float("-inf")
    patience_ctr = 0
    metrics_history = []
    details_by_mode = {}

    # ---- Training loop ----
    print(f"\nTraining for up to {num_epochs} epochs …")
    print(f"Loss config: {LOSS_CONFIG}")
    print(f"Training config: {TRAINING_CONFIG}")
    print()

    for epoch in range(1, num_epochs + 1):
        # ---------------------------
        # Part A: Train for one epoch
        # ---------------------------
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            LOSS_CONFIG, grad_accum, max_grad_norm,
        )

        monitored = None
        for use_oracle in EVAL_CASCADE_MODES:
            # --------------------------------------------------------------
            # Part B: Evaluate the SAME trained model in two different ways.
            #
            # The model weights are not changed here.
            # We are only changing how we *measure* Step 2 and Step 3.
            # --------------------------------------------------------------
            mode_tag = eval_mode_tag(use_oracle)
            res, _ = evaluate(
                model, val_loader, device, entity_types, id_to_etype,
                t_entity=0.5, t_bio=0.5, collect_details=False,
                use_oracle=use_oracle,
            )

            s1 = res["step1_entity"]
            s2 = res["step2_bio"]
            s3 = res["step3_type"]
            sp = res["pipeline_span"]

            print(
                f"Epoch {epoch}/{num_epochs}  TrainLoss={train_loss:.4f}  [mode={mode_tag}]\n"
                f"  Step 1 Entity Detection:      P={s1['precision']:.3f}  R={s1['recall']:.3f}  F1={s1['f1']:.3f}\n"
                f"  Step 2 B-vs-I ({mode_tag:>9s}):  P={s2['precision']:.3f}  R={s2['recall']:.3f}  F1={s2['f1']:.3f}\n"
                f"  Step 3 Type   ({mode_tag:>9s}):  Acc={s3['accuracy']:.3f}\n"
                f"  Pipeline Span-level:          P={sp['precision']:.3f}  R={sp['recall']:.3f}  F1={sp['f1']:.3f}"
            )

            if s3.get("per_type"):
                for name, vals in s3["per_type"].items():
                    if isinstance(vals, dict) and "f1-score" in vals:
                        print(f"    {name:>8s}: P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1-score']:.3f}  Support={vals.get('support', 0)}")

            metrics_history.append({
                "epoch": epoch,
                "eval_mode": mode_tag,
                "train_loss": train_loss,
                "step1_entity_p": s1["precision"], "step1_entity_r": s1["recall"], "step1_entity_f1": s1["f1"],
                "step2_bio_p": s2["precision"],    "step2_bio_r": s2["recall"],    "step2_bio_f1": s2["f1"],
                "step3_type_acc": s3["accuracy"],
                "pipeline_span_p": sp["precision"], "pipeline_span_r": sp["recall"], "pipeline_span_f1": sp["f1"],
            })

            # We use predicted mode for early stopping because it reflects the
            # behavior of the real pipeline that would be used in practice.
            if mode_tag == "predicted":
                monitored = s1["f1"] + s2["f1"]

        if monitored is None and metrics_history:
            last = metrics_history[-1]
            monitored = last["step1_entity_f1"] + last["step2_bio_f1"]

        # Early stopping on combined step1 + step2 F1 (predicted mode)
        # If the realistic pipeline stops improving, we stop training.
        if monitored > best_monitored + min_delta:
            best_monitored = monitored
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} evals)")
                break

    # ---- Threshold optimisation ----
    # We tune the decision thresholds AFTER training.
    # This does not retrain the model; it only changes how probabilities are
    # converted to binary decisions for Step 1 and Step 2.
    print("\nOptimising thresholds on validation set …")
    best_te, best_tb = optimise_thresholds(model, val_loader, device, entity_types, id_to_etype)

    # ---- Final evaluation with optimised thresholds ----
    # We now report final results in both modes using the tuned thresholds.
    print(f"\nFinal evaluation with optimised thresholds (entity={best_te:.2f}, bio={best_tb:.2f}):")
    for use_oracle in EVAL_CASCADE_MODES:
        mode_tag = eval_mode_tag(use_oracle)
        res_opt, details_opt = evaluate(
            model, val_loader, device, entity_types, id_to_etype,
            t_entity=best_te, t_bio=best_tb, collect_details=True,
            use_oracle=use_oracle,
        )

        s1 = res_opt["step1_entity"]
        s2 = res_opt["step2_bio"]
        s3 = res_opt["step3_type"]
        sp = res_opt["pipeline_span"]

        print(
            f"  Step 1 Entity Detection:      P={s1['precision']:.3f}  R={s1['recall']:.3f}  F1={s1['f1']:.3f}  [mode={mode_tag}]\n"
            f"  Step 2 B-vs-I ({mode_tag:>9s}):  P={s2['precision']:.3f}  R={s2['recall']:.3f}  F1={s2['f1']:.3f}\n"
            f"  Step 3 Type   ({mode_tag:>9s}):  Acc={s3['accuracy']:.3f}\n"
            f"  Pipeline Span-level:          P={sp['precision']:.3f}  R={sp['recall']:.3f}  F1={sp['f1']:.3f}"
        )

        if s3.get("per_type"):
            for name, vals in s3["per_type"].items():
                if isinstance(vals, dict) and "f1-score" in vals:
                    print(f"    {name:>8s}: P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1-score']:.3f}  Support={vals.get('support', 0)}")

        metrics_history.append({
            "epoch": "final_optimised",
            "eval_mode": mode_tag,
            "train_loss": None,
            "step1_entity_p": s1["precision"], "step1_entity_r": s1["recall"], "step1_entity_f1": s1["f1"],
            "step2_bio_p": s2["precision"],    "step2_bio_r": s2["recall"],    "step2_bio_f1": s2["f1"],
            "step3_type_acc": s3["accuracy"],
            "pipeline_span_p": sp["precision"], "pipeline_span_r": sp["recall"], "pipeline_span_f1": sp["f1"],
            "threshold_entity": best_te, "threshold_bio": best_tb,
        })
        details_by_mode[mode_tag] = details_opt

    # ---- Export results ----
    # `metrics` contains epoch-by-epoch summaries.
    # `detailed_results` contains token-level predictions for inspection.
    # Both include `eval_mode` so students can compare predicted vs oracle.
    df_metrics = pd.DataFrame(metrics_history)
    details_frames = []
    for mode_tag in [eval_mode_tag(m) for m in EVAL_CASCADE_MODES]:
        rows = details_by_mode.get(mode_tag, [])
        if rows:
            df_mode = pd.DataFrame(rows[:MAX_DETAIL_ROWS])
            df_mode["eval_mode"] = mode_tag
            details_frames.append(df_mode)
    df_details = pd.concat(details_frames, ignore_index=True) if details_frames else pd.DataFrame()
    excel_path = os.path.join(os.path.dirname(__file__), "cascaded_pipeline_results.xlsx")
    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)
            if not df_details.empty:
                df_details.to_excel(writer, sheet_name="detailed_results", index=False)
        print(f"\nResults exported to {excel_path}")
    except Exception as e:
        print(f"Excel export failed: {e}")

    print("\nDone.")
