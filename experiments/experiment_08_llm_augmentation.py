"""experiment_08_llm_augmentation.py — LLM Mask-Filling Augmentation for Rare Labels
=====================================================================================

This experiment evaluates whether augmenting the training set with synthetically
generated sentences for underrepresented entity types improves NER performance.

Motivation
----------
In NER datasets with many entity types, some labels appear in very few
sentences.  A model trained on such imbalanced data may fail to learn rare
entity types.  This experiment addresses the imbalance by generating new
training sentences using the same DictaBERT model's **masked language model**
(fill-mask) capability — no additional model is required.

Approach
--------
1. Split the dataset using the **label-aware** strategy from experiment 01
   (70 % train / 30 % test, ``tf.split_list`` with ``ensure_label_coverage``).
2. **Do not touch the test set.**
3. Identify rare labels — entity types whose training-set sentence count is
   below the max.
4. For each rare label, select training sentences that contain that entity.
   Use **inverse-frequency scoring** (inspired by experiment 07) so that
   sentences rich in rare labels are prioritised for augmentation.
5. For each selected sentence, apply **three complementary generation strategies**:

   a. **Single-mask fill** — mask the first entity token, predict replacements
      from the known entity vocabulary.
   b. **Multi-position mask** — mask *each* entity occurrence of the target
      label independently, generating variants from every position.
   c. **Context-preserving duplication** — duplicate sentences containing
      extremely rare labels (Q1 threshold) so the model sees them more often,
      even if fill-mask cannot produce enough diversity.

6. Apply a generation **multiplier** (default ×3) so more variants are
   produced per label deficit than the bare delta.
7. Combine original + generated sentences into the augmented training set.
8. Train NER on the **baseline** (original only) and **augmented** (original +
   generated) training sets; evaluate both on the **same unmodified test set**.
9. Repeat across multiple random seeds and aggregate metrics as mean ± std.
10. **Save** the baseline and augmented train/eval splits as JSON for later
    reuse by experiments 03–06.

Outputs
-------
* ``outputs/exp08/*.xlsx`` — Excel workbook with score tables, label-count
  comparison, generation log, and documentation sheet.
* ``outputs/exp08/*.json`` — machine-readable results.
* ``outputs/exp08/*.csv``  — per-seed / metric-stats / thesis-summary CSVs.
* ``outputs/exp08/splits/`` — saved train/eval JSON files for reuse.

Usage
-----
::

    python experiments/experiment_08_llm_augmentation.py

Environment variables
---------------------
``THESIS_SPLIT_SEED``          Base random seed (default 42).
``THESIS_EXP08_NUM_SEEDS``     Number of seeds to evaluate (default 5, min 2).
``THESIS_EXP08_MULTIPLIER``    Generation multiplier (default 3).
``THESIS_AUGMENTATION_MODEL_NAME``
                               Optional model/path used only for fill-mask augmentation.
                               If unset, resolved from the NER model's original base source.
``THESIS_NER_CSV``             Override dataset path.
``THESIS_DEBUG``               Set to ``1`` for verbose output.
"""

from __future__ import annotations

import json as _json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    configure_model_environment,
    get_experiment_output_dir,
    now_timestamp,
    resolve_dataset,
    suppress_output_if_needed,
    write_result_excel,
    write_result_json,
)
from split_io import build_thesis_documentation_df, save_split


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import th_functions as tf  # type: ignore
from NERtraining import PrepDataSetNERTraining  # type: ignore

try:
    from transformers import logging as transformers_logging
    from transformers import pipeline as hf_pipeline
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:  # pragma: no cover
    transformers_logging = None
    hf_pipeline = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

# ── Constants ──────────────────────────────────────────────────────────────

BASELINE_VARIANT = "baseline_no_augmentation"
AUGMENTED_VARIANT = "augmented_llm_mask_filling"
ALL_VARIANTS = [BASELINE_VARIANT, AUGMENTED_VARIANT]

VARIANT_DESCRIPTIONS = {
    BASELINE_VARIANT: "Baseline: label-aware split (exp01 strategy), no augmentation",
    AUGMENTED_VARIANT: (
        "Training data augmented with LLM mask-filling (multi-position + "
        "context-preserving duplication) for rare entity labels"
    ),
}

THESIS_LABELS = {
    BASELINE_VARIANT: "Baseline (no augmentation)",
    AUGMENTED_VARIANT: "LLM mask-fill augmented",
}


# ── Configuration ─────────────────────────────────────────────────────────

def _resolve_seed(default: int = 42) -> int:
    raw = (os.environ.get("THESIS_SPLIT_SEED") or str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_num_seeds(default: int = 5) -> int:
    raw = (os.environ.get("THESIS_EXP08_NUM_SEEDS") or str(default)).strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return default


def _resolve_multiplier(default: int = 3) -> int:
    """Generation multiplier: how many times the raw deficit to generate."""
    raw = (os.environ.get("THESIS_EXP08_MULTIPLIER") or str(default)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _configure_quiet_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if transformers_logging is not None:
        transformers_logging.set_verbosity_error()


def _looks_like_mlm_config(config_path: Path) -> bool:
    """Heuristic check for MLM-capable checkpoints."""
    try:
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    archs = cfg.get("architectures") or []
    for arch in archs:
        a = str(arch)
        if "MaskedLM" in a or a.endswith("LMHeadModel"):
            return True
    return False


def _resolve_local_snapshot_model(hf_id: str) -> str | None:
    """Resolve a local huggingface snapshot directory for a model id.

    Expected cache form: models/hf_models/models--owner--name/snapshots/<sha>/
    """
    if "/" not in hf_id:
        return None

    owner, name = hf_id.split("/", 1)
    base = Path(__file__).resolve().parents[1] / "models" / "hf_models" / f"models--{owner}--{name}" / "snapshots"
    if not base.exists():
        return None

    try:
        candidates = [p for p in base.iterdir() if p.is_dir() and (p / "config.json").exists()]
    except Exception:
        return None

    if not candidates:
        return None

    mlm_first = [p for p in candidates if _looks_like_mlm_config(p / "config.json")]
    if mlm_first:
        return str(mlm_first[0])
    return None


def _resolve_local_flat_model(hf_id: str) -> str | None:
    """Resolve a local flat cache directory if it is MLM-capable.

    Expected cache form: models/hf_models/<owner>__<name>/
    """
    if "/" not in hf_id:
        return None

    owner, name = hf_id.split("/", 1)
    base = Path(__file__).resolve().parents[1] / "models" / "hf_models" / f"{owner}__{name}"
    cfg_path = base / "config.json"
    if cfg_path.exists() and _looks_like_mlm_config(cfg_path):
        return str(base)
    return None


def _known_base_model_from_name(name: str) -> str | None:
    lower = name.lower()
    if "berel_3.0" in lower or "berel" in lower:
        return "dicta-il/BEREL_3.0"
    if "dictabert" in lower:
        return "dicta-il/dictabert"
    if "alephbertgimmel" in lower:
        return "dicta-il/alephbertgimmel-base"
    return None


def _local_family_fallback_ids(hf_id: str) -> list[str]:
    """Model-id aliases that are acceptable local MLM substitutes.

    DictaBERT-family NER checkpoints can use AlephBERT-Gimmel MLM for fill-mask.
    """
    aliases = [hf_id]
    if hf_id in {"dicta-il/dictabert", "dicta-il/dictabert-ner"}:
        aliases.append("dicta-il/alephbertgimmel-base")
    return aliases


def _looks_like_hf_model_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "/" not in text:
        return False
    if "\\" in text:
        return False
    if text.startswith("./") or text.startswith("../"):
        return False
    try:
        if Path(text).exists():
            return False
    except Exception:
        pass
    return True


def _resolve_augmentation_model_name(ner_model_name: str) -> str:
    """Resolve which model to use for fill-mask augmentation.

    Priority:
    1) ``THESIS_AUGMENTATION_MODEL_NAME`` if set.
    2) Resolve base model family (DictaBERT/BEREL/...) from NER model.
    3) Use local MLM cache first (snapshot, then flat cache).
    4) If local MLM cache not available, fall back to Hub ID (download).
    5) Final fallback to ``ner_model_name``.
    """
    override = (os.environ.get("THESIS_AUGMENTATION_MODEL_NAME") or "").strip()
    if override:
        return override

    try:
        model_path = Path(ner_model_name)
    except Exception:
        model_path = Path("")

    candidates: list[str] = []

    # Candidate from model id passed directly.
    if _looks_like_hf_model_id(ner_model_name):
        candidates.append(ner_model_name)

    # Candidates from local model path metadata/name.
    if model_path.exists():
        cfg_path = model_path / "config.json"
        if cfg_path.exists():
            try:
                cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                original = str(cfg.get("_name_or_path") or "").strip()
                if original and "/" in original and original != ner_model_name:
                    candidates.append(original)
            except Exception:
                pass

        by_name = _known_base_model_from_name(model_path.name)
        if by_name:
            candidates.append(by_name)

    # Candidate from raw string model name.
    by_raw_name = _known_base_model_from_name(ner_model_name)
    if by_raw_name:
        candidates.append(by_raw_name)

    # Keep unique order.
    unique_candidates: list[str] = []
    for c in candidates:
        if c and c not in unique_candidates:
            unique_candidates.append(c)

    # Prefer local MLM caches first.
    for hf_id in unique_candidates:
        for candidate_id in _local_family_fallback_ids(hf_id):
            snap = _resolve_local_snapshot_model(candidate_id)
            if snap:
                return snap
            flat = _resolve_local_flat_model(candidate_id)
            if flat:
                return flat

    # No local MLM copy: allow transformers to download from Hub.
    if unique_candidates:
        return unique_candidates[0]

    return ner_model_name


def _build_fill_mask_pipeline(fill_mask_model: str):
    """Build fill-mask pipeline with explicit local/offline preference.

    Prevents unnecessary hub calls when a local model path is available.
    """
    if hf_pipeline is None:
        raise RuntimeError("transformers pipeline is unavailable")

    model_path = Path(fill_mask_model)
    local_only_env = (os.environ.get("THESIS_AUGMENTATION_LOCAL_ONLY") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    local_only = local_only_env or model_path.exists()

    if local_only and AutoTokenizer is not None and AutoModelForMaskedLM is not None:
        tokenizer = AutoTokenizer.from_pretrained(fill_mask_model, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(fill_mask_model, local_files_only=True)
        return hf_pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=200)

    return hf_pipeline("fill-mask", model=fill_mask_model, top_k=200)


# ── Augmentation Core ─────────────────────────────────────────────────────

def _get_entity_tokens_from_df(data_df: pd.DataFrame, label_suffix: str) -> list[str]:
    """Get distinct tokens for B-{label} entities from the full corpus."""
    result_df = tf.filter_df(data_df, "raw_tags", f"B-{label_suffix}", f"I-{label_suffix}")
    if result_df.empty:
        # Fallback: also try multi-token entities (first token of B-{suffix})
        b_rows = data_df[data_df["raw_tags"] == f"B-{label_suffix}"]
        if b_rows.empty:
            return []
        return b_rows["token"].dropna().unique().tolist()
    return result_df["token"].dropna().unique().tolist()


def _find_all_entity_positions(labels: list[str], label_suffix: str) -> list[int]:
    """Return indices of all tokens matching B-{suffix} or I-{suffix}."""
    target_tags = {f"B-{label_suffix}", f"I-{label_suffix}"}
    return [i for i, lb in enumerate(labels) if lb in target_tags]


def _filter_valid_targets(fill_mask_fn, target_tokens: list[str]) -> list[str]:
    """Keep only target tokens that exist in the model vocabulary.

    This prevents noisy pipeline warnings such as:
    "The specified target token ... does not exist in the model vocabulary".
    """
    tokenizer = getattr(fill_mask_fn, "tokenizer", None)
    if tokenizer is None:
        return list(dict.fromkeys(target_tokens))

    unk_id = getattr(tokenizer, "unk_token_id", None)
    valid: list[str] = []
    seen: set[str] = set()

    for tok in target_tokens:
        t = str(tok).strip()
        if not t or t in seen:
            continue
        try:
            tok_id = tokenizer.convert_tokens_to_ids(t)
        except Exception:
            continue

        if tok_id is None:
            continue
        if unk_id is not None and tok_id == unk_id:
            continue

        seen.add(t)
        valid.append(t)

    return valid


def _generate_variants_single(
    sentence: dict,
    label_suffix: str,
    mask_idx: int,
    num_variants: int,
    fill_mask_fn,
    known_entity_tokens: list[str],
) -> list[dict]:
    """Generate variants by masking a single position and running fill-mask."""
    tokens = sentence["text"].split()
    labels = list(sentence["labels"])

    if mask_idx >= len(tokens) or not known_entity_tokens:
        return []

    original_token = tokens[mask_idx]
    masked_tokens = list(tokens)
    masked_tokens[mask_idx] = "[MASK]"
    masked_text = " ".join(masked_tokens)

    valid_targets = _filter_valid_targets(fill_mask_fn, known_entity_tokens)
    if not valid_targets:
        return []

    try:
        effective_k = min(num_variants + 5, len(valid_targets))
        if effective_k <= 0:
            return []
        predictions = fill_mask_fn(masked_text, targets=valid_targets, top_k=effective_k)
    except Exception:
        return []

    if isinstance(predictions, dict):
        predictions = [predictions]

    variants: list[dict] = []
    for pred in predictions:
        if len(variants) >= num_variants:
            break
        pred_token = (pred.get("token_str") or "").strip()
        if not pred_token or pred_token == original_token:
            continue
        new_tokens = list(tokens)
        new_tokens[mask_idx] = pred_token
        variants.append({"text": " ".join(new_tokens), "labels": list(labels)})

    return variants


def _generate_variants_multi_position(
    sentence: dict,
    label_suffix: str,
    num_variants_per_position: int,
    fill_mask_fn,
    known_entity_tokens: list[str],
) -> list[dict]:
    """Generate variants by masking each entity position independently."""
    labels = list(sentence["labels"])
    positions = _find_all_entity_positions(labels, label_suffix)
    tokens = sentence["text"].split()

    if not positions or not known_entity_tokens:
        return []

    all_variants: list[dict] = []
    seen_texts: set[str] = set()

    for pos in positions:
        if pos >= len(tokens):
            continue
        variants = _generate_variants_single(
            sentence, label_suffix, pos, num_variants_per_position,
            fill_mask_fn, known_entity_tokens,
        )
        for v in variants:
            if v["text"] not in seen_texts:
                seen_texts.add(v["text"])
                all_variants.append(v)

    return all_variants


def _duplicate_sentence(sentence: dict, count: int) -> list[dict]:
    """Create exact copies of a sentence (context-preserving duplication)."""
    return [{"text": sentence["text"], "labels": list(sentence["labels"])} for _ in range(count)]


def _score_sentence_by_rarity(
    sentence: dict,
    global_label_counts: dict[str, int],
    max_label_count: int,
) -> float:
    """Score a sentence by inverse-frequency of its labels (exp07-inspired).

    Sentences containing rarer labels get higher scores, making them
    higher-priority targets for augmentation.  Uses log-dampened weighting
    to avoid extreme skew: ``log(1 + max_count / label_count)``.
    """
    labels_in_sent = {str(lb) for lb in sentence.get("labels", []) if str(lb) != "O"}
    if not labels_in_sent:
        return 0.0
    return sum(
        math.log(1.0 + max_label_count / global_label_counts.get(lb, max_label_count))
        for lb in labels_in_sent
    )


def _augment_training_data(
    train_sentences: list[dict],
    full_data_df: pd.DataFrame,
    model_name: str,
    multiplier: int = 3,
    rng_seed: int = 42,
    augmentation_model_name: str | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Generate new sentence variants for rare labels via DictaBERT mask-filling.

    Improvements over v1
    --------------------
    * **Multiplier** — generates ``multiplier × delta`` variants per label
      (default ×3) instead of 1:1 with the deficit.
    * **Multi-position masking** — masks each entity token position in a
      sentence independently, not just the first.
    * **Inverse-freq sentence scoring** — sentences rich in rare labels get
      more augmentation (inspired by experiment 07).
    * **Context-preserving duplication** — for extremely rare labels (≤ Q1
      threshold), duplicate sentences directly to ensure the model gets
      enough exposure even when fill-mask diversity is limited.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    rng = random.Random(rng_seed)

    fill_mask_model = augmentation_model_name or _resolve_augmentation_model_name(model_name)
    print("    Creating fill-mask pipeline …")
    fill_mask = _build_fill_mask_pipeline(fill_mask_model)

    label_stats = tf.generate_label_df(train_sentences)
    generated: list[dict] = []
    log_rows: list[dict] = []
    current_train = list(train_sentences)

    # Compute Q1 threshold for "extremely rare" labels → also duplicate these
    if not label_stats.empty:
        q1_threshold = float(label_stats["Distinct Sentence Count"].quantile(0.25))
    else:
        q1_threshold = 0

    # Global label token counts for inverse-freq scoring
    global_label_counts = _non_o_label_counts(train_sentences)
    max_label_count = max(global_label_counts.values()) if global_label_counts else 1

    for _, row in label_stats.iterrows():
        label_suffix = row["Label"]
        delta = int(row["Delta to Max"])
        sent_count = int(row["Distinct Sentence Count"])
        if delta <= 0:
            continue

        candidates = tf.filter_items_by_label_suffix(train_sentences, label_suffix=label_suffix)
        if not candidates:
            continue

        # ── Score sentences by rarity (exp07-inspired) ────────────────
        scored = [
            (sent, _score_sentence_by_rarity(sent, global_label_counts, max_label_count))
            for sent in candidates
        ]
        scored.sort(key=lambda x: -x[1])  # highest rarity score first

        # How many total variants to aim for
        target_total = delta * multiplier

        # Get known entity tokens for fill-mask targets
        known_tokens = _get_entity_tokens_from_df(full_data_df, label_suffix)

        is_extremely_rare = sent_count <= q1_threshold
        label_generated = 0
        label_duplicated = 0

        # ── Distribute generation budget across sentences ─────────────
        # Inverse-freq weighted: higher-scored sentences get more budget
        total_score = sum(s for _, s in scored) or 1.0
        for sent, score in scored:
            portion = score / total_score
            sent_budget = max(1, int(round(target_total * portion)))

            mask_variants: list[dict] = []

            # Strategy A: Multi-position fill-mask
            if known_tokens:
                entity_positions = _find_all_entity_positions(
                    list(sent.get("labels", [])), label_suffix
                )
                num_positions = max(1, len(entity_positions))
                variants_per_pos = max(1, sent_budget // num_positions)
                mask_variants = _generate_variants_multi_position(
                    sentence=sent,
                    label_suffix=label_suffix,
                    num_variants_per_position=variants_per_pos,
                    fill_mask_fn=fill_mask,
                    known_entity_tokens=known_tokens,
                )

            generated.extend(mask_variants)
            label_generated += len(mask_variants)

            # Strategy B: Context-preserving duplication for extremely rare
            shortfall = sent_budget - len(mask_variants)
            dup_count = 0
            if is_extremely_rare and shortfall > 0:
                dup_count = min(shortfall, sent_budget)
                dups = _duplicate_sentence(sent, dup_count)
                generated.extend(dups)
                label_duplicated += len(dups)

            log_rows.append({
                "label": label_suffix,
                "source_text_preview": sent["text"][:80],
                "rarity_score": round(score, 4),
                "budget": sent_budget,
                "mask_generated": len(mask_variants),
                "duplicated": dup_count,
                "known_token_count": len(known_tokens),
                "is_extremely_rare": is_extremely_rare,
            })

        total_for_label = label_generated + label_duplicated
        print(
            f"      {label_suffix}: {label_generated} mask-filled + "
            f"{label_duplicated} duplicated = {total_for_label} "
            f"(target {target_total}, delta {delta}, x{multiplier})"
        )

        # Recalculate stats so later labels account for what was generated
        current_train = list(train_sentences) + generated
        label_stats = tf.generate_label_df(current_train)
        global_label_counts = _non_o_label_counts(current_train)
        max_label_count = max(global_label_counts.values()) if global_label_counts else 1

    # Shuffle generated to avoid ordering bias during training
    rng.shuffle(generated)

    log_df = (
        pd.DataFrame(log_rows)
        if log_rows
        else pd.DataFrame(columns=[
            "label", "source_text_preview", "rarity_score", "budget",
            "mask_generated", "duplicated", "known_token_count", "is_extremely_rare",
        ])
    )
    return generated, log_df


# ── Label Analysis ────────────────────────────────────────────────────────

def _non_o_label_counts(sentences: list[dict]) -> dict[str, int]:
    """Count non-O label tokens across all sentences."""
    counts: dict[str, int] = {}
    for sent in sentences:
        for label in sent.get("labels", []):
            key = str(label)
            if key != "O":
                counts[key] = counts.get(key, 0) + 1
    return counts


def _sentence_presence_counts(sentences: list[dict]) -> dict[str, int]:
    """Count how many sentences contain each non-O label (presence, not tokens)."""
    counts: dict[str, int] = {}
    for sent in sentences:
        seen = {str(lb) for lb in sent.get("labels", []) if str(lb) != "O"}
        for label in seen:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _build_label_count_table(
    original_train: list[dict],
    augmented_train: list[dict],
) -> pd.DataFrame:
    """Compare label distributions in training data before and after augmentation."""
    orig_tok = _non_o_label_counts(original_train)
    aug_tok = _non_o_label_counts(augmented_train)
    orig_sent = _sentence_presence_counts(original_train)
    aug_sent = _sentence_presence_counts(augmented_train)

    all_labels = sorted(set(orig_tok) | set(aug_tok))
    if not all_labels:
        return pd.DataFrame()

    total_orig = sum(orig_tok.values()) or 1
    total_aug = sum(aug_tok.values()) or 1

    rows = []
    for label in all_labels:
        o_t = orig_tok.get(label, 0)
        a_t = aug_tok.get(label, 0)
        o_s = orig_sent.get(label, 0)
        a_s = aug_sent.get(label, 0)
        rows.append({
            "Label": label,
            "Original Token Count": o_t,
            "Original Token %": round(100.0 * o_t / total_orig, 2),
            "Augmented Token Count": a_t,
            "Augmented Token %": round(100.0 * a_t / total_aug, 2),
            "Delta Token Count": a_t - o_t,
            "Delta Token %": round(100.0 * a_t / total_aug - 100.0 * o_t / total_orig, 2),
            "Original Sentence Count": o_s,
            "Augmented Sentence Count": a_s,
            "Delta Sentence Count": a_s - o_s,
        })

    return pd.DataFrame(rows).sort_values(
        by=["Original Token Count", "Label"], ascending=[True, True], ignore_index=True
    )


# ── Training ──────────────────────────────────────────────────────────────

def _train_and_eval(
    data_df: pd.DataFrame,
    train_sentences: list[dict],
    eval_sentences: list[dict],
    model_name: str,
    is_local_model: bool,
) -> dict:
    """Train NER model on *train_sentences* and evaluate on *eval_sentences*."""
    model, tokenizer, data_collator, ds_train, ds_eval, _, label_list = (
        tf.setup_token_classification(
            data=data_df,
            train_data=train_sentences,
            test_data=eval_sentences,
            eval_data=eval_sentences,
            model_name=model_name,
            local_files_only=is_local_model,
        )
    )
    _, eval_results = tf.train_and_evaluate_model(
        model, ds_train, ds_eval, data_collator, tokenizer, label_list, metric_name="seqeval"
    )
    return {
        "f1": eval_results.get("eval_overall_f1"),
        "precision": eval_results.get("eval_overall_precision"),
        "recall": eval_results.get("eval_overall_recall"),
        "accuracy": eval_results.get("eval_overall_accuracy"),
        "loss": eval_results.get("eval_loss"),
    }


# ── Reporting helpers ─────────────────────────────────────────────────────

def _fmt(value: float | None) -> str:
    return f"{float(value):.4f}" if value is not None else "N/A"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        return None if pd.isna(v) else v
    except Exception:
        return None


def _build_metric_stats(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std per variant across seeds, plus delta row."""
    metrics = ["f1", "precision", "recall", "accuracy", "loss"]
    rows: list[dict] = []

    for variant in ALL_VARIANTS:
        subset = per_seed_df[per_seed_df["variant"] == variant]
        row: dict[str, Any] = {
            "variant": variant,
            "description": VARIANT_DESCRIPTIONS.get(variant, variant),
            "seeds": int(subset["seed"].nunique()),
        }
        for m in metrics:
            vals = pd.to_numeric(subset[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if not vals.empty else None
            row[f"{m}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1
                else 0.0 if len(vals) == 1
                else None
            )
        rows.append(row)

    # Delta row
    if len(rows) >= 2:
        base, aug = rows[0], rows[1]
        delta: dict[str, Any] = {
            "variant": "delta_augmented_minus_baseline",
            "description": "Delta: Augmented - Baseline",
            "seeds": min(base["seeds"], aug["seeds"]),
        }
        for m in metrics:
            b = _safe_float(base.get(f"{m}_mean"))
            a = _safe_float(aug.get(f"{m}_mean"))
            delta[f"{m}_mean"] = (a - b) if a is not None and b is not None else None
            delta[f"{m}_std"] = None
        rows.append(delta)

    return pd.DataFrame(rows)


def _build_thesis_summary(metric_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build a presentation-ready summary (Condition | F1 mean+-std | ...)."""
    non_delta = metric_stats_df[~metric_stats_df["variant"].str.startswith("delta_")].copy()
    non_delta["Condition"] = non_delta["variant"].map(THESIS_LABELS).fillna(non_delta["variant"])

    non_delta["F1 (mean+-std)"] = non_delta.apply(
        lambda r: f"{_fmt(r.get('f1_mean'))}+-{_fmt(r.get('f1_std'))}", axis=1
    )
    non_delta["Precision (mean+-std)"] = non_delta.apply(
        lambda r: f"{_fmt(r.get('precision_mean'))}+-{_fmt(r.get('precision_std'))}", axis=1
    )
    non_delta["Recall (mean+-std)"] = non_delta.apply(
        lambda r: f"{_fmt(r.get('recall_mean'))}+-{_fmt(r.get('recall_std'))}", axis=1
    )
    non_delta["Accuracy (mean+-std)"] = non_delta.apply(
        lambda r: f"{_fmt(r.get('accuracy_mean'))}+-{_fmt(r.get('accuracy_std'))}", axis=1
    )

    thesis_df = non_delta[
        ["Condition", "seeds", "F1 (mean+-std)", "Precision (mean+-std)",
         "Recall (mean+-std)", "Accuracy (mean+-std)"]
    ].copy()

    # Append delta row
    baseline = non_delta[non_delta["variant"] == BASELINE_VARIANT]
    augmented = non_delta[non_delta["variant"] == AUGMENTED_VARIANT]
    if not baseline.empty and not augmented.empty:
        bs = baseline.iloc[0]
        ag = augmented.iloc[0]
        delta_row = {
            "Condition": "Delta (Augmented - Baseline)",
            "seeds": min(int(bs["seeds"]), int(ag["seeds"])),
            "F1 (mean+-std)": _fmt(
                _safe_float(ag.get("f1_mean")) - _safe_float(bs.get("f1_mean"))
            ),
            "Precision (mean+-std)": _fmt(
                _safe_float(ag.get("precision_mean")) - _safe_float(bs.get("precision_mean"))
            ),
            "Recall (mean+-std)": _fmt(
                _safe_float(ag.get("recall_mean")) - _safe_float(bs.get("recall_mean"))
            ),
            "Accuracy (mean+-std)": _fmt(
                _safe_float(ag.get("accuracy_mean")) - _safe_float(bs.get("accuracy_mean"))
            ),
        }
        thesis_df = pd.concat([thesis_df, pd.DataFrame([delta_row])], ignore_index=True)

    return thesis_df


def _write_csv_outputs(
    exp_dir: Path,
    per_seed_df: pd.DataFrame,
    metric_stats_df: pd.DataFrame,
    thesis_df: pd.DataFrame,
    label_count_df: pd.DataFrame,
) -> dict:
    ts = now_timestamp()

    def _safe(df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            pass

    paths: dict[str, str] = {}
    for name, df in [
        ("per_seed", per_seed_df),
        ("metric_stats", metric_stats_df),
        ("thesis_summary", thesis_df),
        ("label_count", label_count_df),
    ]:
        timestamped = exp_dir / f"llm_augmentation_{name}_{ts}.csv"
        latest = exp_dir / f"llm_augmentation_{name}_latest.csv"
        df.to_csv(timestamped, index=False)
        _safe(df, latest)
        paths[f"{name}_csv"] = str(timestamped)
        paths[f"{name}_csv_latest"] = str(latest)

    return paths


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


# ── Split saving ──────────────────────────────────────────────────────────

def _get_exp08_splits_dir() -> Path:
    splits_dir = Path(__file__).resolve().parents[1] / "outputs" / "exp08" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    return splits_dir


def _save_exp08_splits(
    baseline_train: list[dict],
    baseline_eval: list[dict],
    augmented_train: list[dict],
    seed: int,
    multiplier: int,
    augmentation_model: str,
    augmented_f1_mean: float | None,
    baseline_f1_mean: float | None,
) -> Path:
    """Save baseline and augmented train/eval splits as JSON for reuse."""
    splits_dir = _get_exp08_splits_dir()

    save_split(baseline_train, splits_dir / "baseline_train.json")
    save_split(baseline_eval, splits_dir / "baseline_eval.json")
    save_split(augmented_train, splits_dir / "augmented_train.json")
    # eval is the same for augmented — save under a clear name too
    save_split(baseline_eval, splits_dir / "augmented_eval.json")

    meta = {
        "experiment": "exp08",
        "description": "LLM mask-filling augmentation splits for reuse by exp03-06",
        "seed": seed,
        "multiplier": multiplier,
        "augmentation_model": augmentation_model,
        "baseline_train_sentences": len(baseline_train),
        "augmented_train_sentences": len(augmented_train),
        "generated_sentences": len(augmented_train) - len(baseline_train),
        "eval_sentences": len(baseline_eval),
        "baseline_f1_mean": baseline_f1_mean,
        "augmented_f1_mean": augmented_f1_mean,
        "note": "eval set is identical for baseline and augmented (never modified)",
        "files": {
            "baseline_train": "baseline_train.json",
            "baseline_eval": "baseline_eval.json",
            "augmented_train": "augmented_train.json",
            "augmented_eval": "augmented_eval.json",
        },
    }
    meta_path = splits_dir / "split_meta.json"
    meta_path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Saved splits to {splits_dir}")
    return splits_dir


# ── Main entry point ──────────────────────────────────────────────────────

def run() -> dict:
    # ── Resolve paths & config ────────────────────────────────────────
    dataset_override = (os.environ.get("THESIS_NER_CSV") or "").strip()
    dataset_path = Path(dataset_override) if dataset_override else resolve_dataset("ner_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    split_seed = _resolve_seed()
    num_seeds = _resolve_num_seeds()
    multiplier = _resolve_multiplier()
    split_ratio = 0.7
    _configure_quiet_runtime()
    model_name, is_local_model = configure_model_environment()
    augmentation_model_name = _resolve_augmentation_model_name(model_name)
    exp_dir = get_experiment_output_dir("exp08")

    # ── Load data ─────────────────────────────────────────────────────
    worker = PrepDataSetNERTraining()
    with suppress_output_if_needed():
        data_df = worker.load_and_prepare_data(str(dataset_path))
        sentences = tf.train_data_fit(data_df)

    print(f"  Dataset: {dataset_path}")
    print(
        f"  Sentences: {len(sentences)}, Seeds: {num_seeds}, Multiplier: x{multiplier}, "
        f"NER model: {model_name}, Augmentation model: {augmentation_model_name}"
    )

    per_seed_rows: list[dict[str, Any]] = []
    first_seed_label_count_df: pd.DataFrame | None = None
    first_seed_gen_log_df: pd.DataFrame | None = None
    first_seed_augmentation_summary: dict | None = None
    first_seed_train: list[dict] | None = None
    first_seed_eval: list[dict] | None = None
    first_seed_augmented_train: list[dict] | None = None

    for seed_offset in range(num_seeds):
        current_seed = split_seed + seed_offset
        print(f"\n  -- Seed {current_seed} ({seed_offset + 1}/{num_seeds}) --")

        # Label-aware split (same strategy as experiment 01)
        train_sentences, eval_sentences = tf.split_list(
            sentences, split_ratio=split_ratio, seed=current_seed, ensure_label_coverage=True
        )
        print(f"    Split: {len(train_sentences)} train / {len(eval_sentences)} eval")

        # ── Baseline: train on original data ──────────────────────────
        print("    Training baseline ...")
        with suppress_output_if_needed():
            baseline_metrics = _train_and_eval(
                data_df, train_sentences, eval_sentences, model_name, is_local_model
            )
        print(f"    Baseline F1 = {_fmt(baseline_metrics['f1'])}")

        per_seed_rows.append({
            "variant": BASELINE_VARIANT,
            "seed": current_seed,
            "train_sentences": len(train_sentences),
            "eval_sentences": len(eval_sentences),
            "generated_sentences": 0,
            **baseline_metrics,
        })

        # ── Augment training data ─────────────────────────────────────
        print("    Augmenting training data with LLM mask-filling ...")
        with suppress_output_if_needed():
            generated_sents, gen_log_df = _augment_training_data(
                train_sentences, data_df, model_name,
                multiplier=multiplier, rng_seed=current_seed,
                augmentation_model_name=augmentation_model_name,
            )
        augmented_train = train_sentences + generated_sents
        print(f"    Generated {len(generated_sents)} new sentences -> {len(augmented_train)} total")

        # ── Augmented: train on original + generated ──────────────────
        print("    Training augmented model ...")
        with suppress_output_if_needed():
            augmented_metrics = _train_and_eval(
                data_df, augmented_train, eval_sentences, model_name, is_local_model
            )
        print(f"    Augmented F1 = {_fmt(augmented_metrics['f1'])}")

        per_seed_rows.append({
            "variant": AUGMENTED_VARIANT,
            "seed": current_seed,
            "train_sentences": len(augmented_train),
            "eval_sentences": len(eval_sentences),
            "generated_sentences": len(generated_sents),
            **augmented_metrics,
        })

        # Save first seed's detailed info for Excel + split saving
        if seed_offset == 0:
            first_seed_label_count_df = _build_label_count_table(train_sentences, augmented_train)
            first_seed_gen_log_df = gen_log_df
            first_seed_augmentation_summary = {
                "seed": current_seed,
                "multiplier": multiplier,
                "augmentation_model": augmentation_model_name,
                "original_train_sentences": len(train_sentences),
                "generated_sentences": len(generated_sents),
                "augmented_train_sentences": len(augmented_train),
                "eval_sentences": len(eval_sentences),
            }
            first_seed_train = list(train_sentences)
            first_seed_eval = list(eval_sentences)
            first_seed_augmented_train = list(augmented_train)

    # ── Aggregate across seeds ────────────────────────────────────────
    per_seed_df = pd.DataFrame(per_seed_rows)
    metric_stats_df = _build_metric_stats(per_seed_df)
    thesis_summary_df = _build_thesis_summary(metric_stats_df)

    label_count_df = first_seed_label_count_df if first_seed_label_count_df is not None else pd.DataFrame()
    gen_log_df = first_seed_gen_log_df if first_seed_gen_log_df is not None else pd.DataFrame()

    # ── Save splits for reuse by exp03-06 ─────────────────────────────
    baseline_f1_mean = _safe_float(
        metric_stats_df[metric_stats_df["variant"] == BASELINE_VARIANT].iloc[0].get("f1_mean")
    ) if not metric_stats_df.empty else None
    augmented_f1_mean = _safe_float(
        metric_stats_df[metric_stats_df["variant"] == AUGMENTED_VARIANT].iloc[0].get("f1_mean")
    ) if not metric_stats_df.empty else None

    if first_seed_train and first_seed_eval and first_seed_augmented_train:
        splits_dir = _save_exp08_splits(
            baseline_train=first_seed_train,
            baseline_eval=first_seed_eval,
            augmented_train=first_seed_augmented_train,
            seed=split_seed,
            multiplier=multiplier,
            augmentation_model=augmentation_model_name,
            augmented_f1_mean=augmented_f1_mean,
            baseline_f1_mean=baseline_f1_mean,
        )
    else:
        splits_dir = _get_exp08_splits_dir()

    # ── CSV outputs ───────────────────────────────────────────────────
    csv_paths = _write_csv_outputs(exp_dir, per_seed_df, metric_stats_df, thesis_summary_df, label_count_df)

    # ── Excel workbook ────────────────────────────────────────────────
    per_seed_display = per_seed_df.copy()
    per_seed_display["Condition"] = per_seed_display["variant"].map(THESIS_LABELS).fillna(per_seed_display["variant"])
    per_seed_display = per_seed_display[
        ["Condition", "variant", "seed", "train_sentences", "eval_sentences",
         "generated_sentences", "f1", "precision", "recall", "accuracy", "loss"]
    ]

    # Numeric summary with CI
    non_delta = metric_stats_df[~metric_stats_df["variant"].str.startswith("delta_")].copy()
    non_delta["Condition"] = non_delta["variant"].map(THESIS_LABELS).fillna(non_delta["variant"])
    non_delta["f1_ci95"] = non_delta.apply(
        lambda r: (1.96 * float(r["f1_std"]) / (float(r["seeds"]) ** 0.5))
        if _safe_float(r.get("f1_std")) is not None and float(r.get("seeds", 0)) > 0
        else None,
        axis=1,
    )

    metrics_file = write_result_excel(
        "exp08",
        "llm_augmentation",
        thesis_summary_df,
        per_seed_display,
        extra_sheets={
            "metric_stats": metric_stats_df,
            "score_summary_numeric": non_delta,
            "label_count_comparison": label_count_df,
            "generation_log": gen_log_df,
            "documentation": build_thesis_documentation_df(
                "exp08",
                "LLM Mask-Filling Augmentation for Rare Labels",
                extra_rows=[
                    {"Section": "Experiment", "Key": "Baseline", "Value": VARIANT_DESCRIPTIONS[BASELINE_VARIANT]},
                    {"Section": "Experiment", "Key": "Augmented", "Value": VARIANT_DESCRIPTIONS[AUGMENTED_VARIANT]},
                    {"Section": "Experiment", "Key": "Split Strategy",
                     "Value": "Label-aware greedy (tf.split_list, ensure_label_coverage=True)"},
                    {"Section": "Experiment", "Key": "Train / Test Ratio", "Value": "70 % / 30 %"},
                    {"Section": "Experiment", "Key": "NER Model", "Value": model_name},
                    {"Section": "Experiment", "Key": "Augmentation Model", "Value": augmentation_model_name},
                    {"Section": "Experiment", "Key": "Augmentation Multiplier", "Value": f"x{multiplier}"},
                    {"Section": "Experiment", "Key": "Augmentation Method",
                     "Value": (
                         "DictaBERT fill-mask: (1) multi-position masking - each entity position "
                         "masked independently, (2) inverse-freq sentence scoring - rare-label-rich "
                         "sentences get more budget, (3) context-preserving duplication for extremely "
                         "rare labels (Q1 threshold)"
                     )},
                    {"Section": "Data", "Key": "Test Data Modified", "Value": "No - test set is never touched"},
                    {"Section": "Saved Splits", "Key": "Location", "Value": str(splits_dir)},
                    {"Section": "Saved Splits", "Key": "Files",
                     "Value": "baseline_train.json, baseline_eval.json, augmented_train.json, augmented_eval.json, split_meta.json"},
                ],
            ),
        },
    )

    # ── JSON result ───────────────────────────────────────────────────
    variant_stats: dict[str, Any] = {}
    for rec in _records(metric_stats_df):
        variant_stats[rec["variant"]] = rec

    result: dict[str, Any] = {
        "experiment_id": "exp08",
        "name": "LLM Mask-Filling Augmentation for Rare Labels",
        "description": (
            "Compares baseline NER (label-aware split, no augmentation) against "
            "augmented training where rare-label sentences are expanded via "
            "DictaBERT masked-language-model fill-mask predictions with "
            "multi-position masking, inverse-freq sentence scoring, and "
            "context-preserving duplication."
        ),
        "dataset": str(dataset_path),
        "model": model_name,
        "augmentation_model": augmentation_model_name,
        "model_local": is_local_model,
        "split_parameters": {
            "base_split_seed": split_seed,
            "num_seeds": num_seeds,
            "seed_list": [split_seed + i for i in range(num_seeds)],
            "train_fraction": split_ratio,
            "eval_fraction": 1 - split_ratio,
            "split_strategy": "Label-aware greedy (tf.split_list, ensure_label_coverage=True)",
            "multiplier": multiplier,
        },
        "augmentation": {
            "method": (
                "DictaBERT fill-mask: multi-position masking + inverse-freq scoring + "
                "context-preserving duplication for extremely rare labels"
            ),
            "model": augmentation_model_name,
            "multiplier": multiplier,
            "first_seed_summary": first_seed_augmentation_summary,
        },
        "saved_splits": {
            "directory": str(splits_dir),
            "baseline_train": str(splits_dir / "baseline_train.json"),
            "baseline_eval": str(splits_dir / "baseline_eval.json"),
            "augmented_train": str(splits_dir / "augmented_train.json"),
            "augmented_eval": str(splits_dir / "augmented_eval.json"),
            "meta": str(splits_dir / "split_meta.json"),
        },
        "metric_stats": _records(metric_stats_df),
        "thesis_summary": _records(thesis_summary_df),
        "label_count_table": _records(label_count_df),
        "variant_stats": variant_stats,
        "csv_outputs": csv_paths,
        "metrics_file": str(metrics_file),
        "status": "ok",
    }

    out_path = write_result_json("exp08", "llm_augmentation", result)
    result["result_file"] = str(out_path)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    payload = run()
    split_params = payload.get("split_parameters", {})
    vstats = payload.get("variant_stats", {})
    aug_info = payload.get("augmentation", {}).get("first_seed_summary", {})

    print(f"\n[exp08] seeds={split_params.get('num_seeds')}  base_seed={split_params.get('base_split_seed')}  multiplier=x{split_params.get('multiplier')}")
    if aug_info:
        print(
            f"  First seed: {aug_info.get('original_train_sentences', '?')} original train + "
            f"{aug_info.get('generated_sentences', '?')} generated = "
            f"{aug_info.get('augmented_train_sentences', '?')} augmented train  |  "
            f"{aug_info.get('eval_sentences', '?')} eval (untouched)"
        )
    print()

    header = f"{'Condition':<35s} {'F1 (mean+-std)':>18s} {'Precision':>18s} {'Recall':>18s} {'Accuracy':>18s}"
    print(header)
    print("-" * len(header))
    for vk in ALL_VARIANTS:
        stats = vstats.get(vk, {})
        label = THESIS_LABELS.get(vk, vk)[:35]
        print(
            f"{label:<35s} "
            f"{_fmt(stats.get('f1_mean'))}+-{_fmt(stats.get('f1_std')):>7s} "
            f"{_fmt(stats.get('precision_mean'))}+-{_fmt(stats.get('precision_std')):>7s} "
            f"{_fmt(stats.get('recall_mean'))}+-{_fmt(stats.get('recall_std')):>7s} "
            f"{_fmt(stats.get('accuracy_mean'))}+-{_fmt(stats.get('accuracy_std')):>7s}"
        )

    delta = vstats.get("delta_augmented_minus_baseline", {})
    print(
        f"\n  Delta (Augmented - Baseline):  "
        f"F1={_fmt(delta.get('f1_mean'))}  "
        f"Prec={_fmt(delta.get('precision_mean'))}  "
        f"Rec={_fmt(delta.get('recall_mean'))}  "
        f"Acc={_fmt(delta.get('accuracy_mean'))}"
    )

    saved = payload.get("saved_splits", {})
    if saved:
        print(f"\n  Saved splits: {saved.get('directory')}")
