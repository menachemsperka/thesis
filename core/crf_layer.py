"""
Linear-chain Conditional Random Field (CRF) for sequence tagging
================================================================

Teaching overview
-----------------
In **Experiments 01–06**, the final NER layer is typically a **softmax** over tags
at each token. Each token is scored independently (given contextual embeddings),
which can produce **invalid BIO sequences** (for example ``I-PER`` immediately after ``O``).

A **linear-chain CRF** adds a learned score for every **adjacent tag pair**
``(y_{t-1}, y_t)``. Training maximizes the **log-probability of the gold tag sequence**
using the **forward algorithm**; inference uses the **Viterbi algorithm** to find the
single best tag path for the whole sentence.

This module is the mathematical core used by:

* ``core/bert_crf_training.py`` — regular BERT-CRF (Experiment ``10_regular``)
* ``core/cascaded_crf_runtime.py`` — joint full-tag head on the cascaded encoder (``10_cascade``)

References
----------
* Lafferty et al. (2001) — conditional random fields.
* Lample et al. (2016) — NER with neural emissions + CRF.
* Souza et al. (2019) — BERT-CRF practice (including class-imbalance tricks in the trainer).

Tensor shapes (student checklist)
---------------------------------
* ``emissions``: ``(batch, seq_len, num_tags)`` — one score vector per token from a linear head.
* ``labels``: ``(batch, seq_len)`` — gold tag indices; ``-100`` marks padding / WordPiece pieces to ignore.
* ``transitions``: ``(num_tags + 2, num_tags + 2)`` — includes START and STOP pseudo-states.

See ``experiments/experiment_10_README.md`` for diagrams and lab exercises.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """
    First-order (linear-chain) CRF with explicit START and STOP states.

    Parameters
    ----------
    num_tags : int
        Number of distinct tag indices ``0 .. num_tags-1`` (for example BIO labels).

    Attributes
    ----------
    transitions : nn.Parameter
        Matrix of size ``(num_tags + 2, num_tags + 2)``. Entry ``A[i, j]`` is the
        log-transition score from tag ``i`` to tag ``j``. Rows/columns ``num_tags`` and
        ``num_tags+1`` are START and STOP respectively.

    Teaching note
    -------------
    Emissions come from upstream networks (BERT + linear layer). The CRF only models
    **dependencies between neighboring tags**; it does not replace contextual encoding.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = int(num_tags)
        # transitions[from, to] for tag indices 0..num_tags-1 plus start/stop slots
        self.transitions = nn.Parameter(torch.zeros(self.num_tags + 2, self.num_tags + 2))
        self.start_idx = self.num_tags
        self.end_idx = self.num_tags + 1

    def _valid_mask(self, labels: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """True where CRF loss/decoding should attend (real tokens, not ``-100`` padding labels)."""
        mask = labels.ne(-100)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        return mask

    def _forward_scores(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Log-score ``log P(y | x)`` **numerator** for the gold tag sequence (one scalar per batch item).

        Walks time steps, accumulating ``transition + emission`` for each gold tag.
        Adds the final transition into STOP when the sequence is non-empty.
        """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device
        score = torch.zeros(batch_size, device=device, dtype=emissions.dtype)

        for b in range(batch_size):
            seq_score = emissions.new_zeros(())
            prev_tag = self.start_idx
            for t in range(seq_len):
                if not mask[b, t]:
                    continue
                tag = int(labels[b, t].item())
                seq_score = seq_score + self.transitions[prev_tag, tag] + emissions[b, t, tag]
                prev_tag = tag
            if mask[b].any():
                seq_score = seq_score + self.transitions[prev_tag, self.end_idx]
            score[b] = seq_score
        return score

    def _partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Log of the partition function ``log Z(x)`` via the forward algorithm.

        Only timesteps where ``mask`` is True participate in the chain (same as
        ``_forward_scores``), so [CLS]/[SEP] and padding with ``labels == -100`` are skipped.
        """
        batch_size, seq_len, num_tags = emissions.shape
        log_z = emissions.new_zeros(batch_size)

        for b in range(batch_size):
            valid_ts = mask[b].nonzero(as_tuple=False).flatten()
            if valid_ts.numel() == 0:
                continue
            t0 = int(valid_ts[0].item())
            log_alpha = emissions[b, t0] + self.transitions[self.start_idx, :num_tags]
            for step in range(1, valid_ts.numel()):
                t = int(valid_ts[step].item())
                emit = emissions[b, t]
                trans = self.transitions[:num_tags, :num_tags]
                scores = log_alpha.unsqueeze(1) + trans + emit.unsqueeze(0)
                log_alpha = torch.logsumexp(scores, dim=0)
            log_alpha = log_alpha + self.transitions[:num_tags, self.end_idx]
            log_z[b] = torch.logsumexp(log_alpha, dim=0)
        return log_z

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Mean negative log-likelihood over batch elements that contain at least one valid token.

        This is the loss term added to the transformer training objective in Experiment 10.
        """
        # Forward–backward / logsumexp in fp32 for numerical stability (especially with fp16 training).
        emissions = emissions.float()
        mask = self._valid_mask(labels, attention_mask)
        gold = self._forward_scores(emissions, labels, mask)
        log_z = self._partition_function(emissions, mask)
        nll = log_z - gold
        valid = mask.any(dim=1)
        if valid.any():
            return nll[valid].mean()
        return nll.mean()

    def _viterbi_on_valid_steps(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi on masked timesteps only; returns one tag id per sequence position."""
        batch_size, seq_len, num_tags = emissions.shape
        results: list[list[int]] = []
        for b in range(batch_size):
            full_path = [-100] * seq_len
            valid_ts = mask[b].nonzero(as_tuple=False).flatten()
            if valid_ts.numel() == 0:
                results.append(full_path)
                continue
            t0 = int(valid_ts[0].item())
            viterbi = emissions.new_full((valid_ts.numel(), num_tags), float("-inf"))
            backpointers: list[torch.Tensor] = []
            viterbi[0] = emissions[b, t0] + self.transitions[self.start_idx, :num_tags]
            for step in range(1, valid_ts.numel()):
                t = int(valid_ts[step].item())
                broadcast = viterbi[step - 1].unsqueeze(1) + self.transitions[:num_tags, :num_tags]
                best_scores, best_paths = broadcast.max(dim=0)
                viterbi[step] = best_scores + emissions[b, t]
                backpointers.append(best_paths)
            terminal = viterbi[valid_ts.numel() - 1] + self.transitions[:num_tags, self.end_idx]
            best_last = int(terminal.argmax().item())
            tags: list[int] = [best_last]
            for bp in reversed(backpointers):
                best_last = int(bp[best_last].item())
                tags.append(best_last)
            tags.reverse()
            for t_idx, tag in zip(valid_ts.tolist(), tags):
                full_path[int(t_idx)] = int(tag)
            results.append(full_path)
        return results

    @torch.no_grad()
    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Viterbi decoding: best tag path per sequence in the batch.

        When ``labels`` is provided, decoding uses the same mask as training (``labels != -100``).
        Each returned path has length ``seq_len``; positions outside the mask are ``-100``.
        """
        emissions = emissions.float()
        if labels is not None:
            mask = self._valid_mask(labels, attention_mask)
        elif attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones(
                emissions.shape[0],
                emissions.shape[1],
                dtype=torch.bool,
                device=emissions.device,
            )
        return self._viterbi_on_valid_steps(emissions, mask)
