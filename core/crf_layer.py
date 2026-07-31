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

        ``Z(x)`` sums ``exp(score(y))`` over **all** tag sequences ``y`` compatible with the mask.
        Training uses ``NLL = log Z - score(y*)`` so gradients push probability mass onto gold paths.
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        log_alpha = emissions.new_full((batch_size, num_tags), float("-inf"))
        log_alpha[:, :] = emissions[:, 0, :] + self.transitions[self.start_idx, :num_tags]
        log_alpha = torch.where(mask[:, 0:1], log_alpha, emissions.new_zeros(()))

        for t in range(1, seq_len):
            emit = emissions[:, t, :].unsqueeze(1)  # (B, 1, T)
            trans = self.transitions[:num_tags, :num_tags].unsqueeze(0)  # (1, T, T)
            prev = log_alpha.unsqueeze(2)  # (B, T, 1)
            scores = prev + trans + emit
            log_alpha = torch.logsumexp(scores, dim=1)
            step_mask = mask[:, t].unsqueeze(1)
            log_alpha = torch.where(step_mask, log_alpha, log_alpha.new_zeros(()))

        log_alpha = log_alpha + self.transitions[:num_tags, self.end_idx].unsqueeze(0)
        return torch.logsumexp(log_alpha, dim=1)

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
        mask = self._valid_mask(labels, attention_mask)
        gold = self._forward_scores(emissions, labels, mask)
        log_z = self._partition_function(emissions, mask)
        nll = log_z - gold
        valid = mask.any(dim=1)
        if valid.any():
            return nll[valid].mean()
        return nll.mean()

    @torch.no_grad()
    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Viterbi decoding: best tag path per sequence in the batch.

        Returns a list of length ``batch``; each inner list has one tag index per **valid**
        time step (up to ``attention_mask`` length). Used at inference instead of per-token argmax.

        Student exercise: compare Viterbi output to argmax on ``emissions`` alone on the same sentence.
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        results: list[list[int]] = []

        for b in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[b].sum().item())
            else:
                valid_len = seq_len
            valid_len = max(0, min(valid_len, seq_len))

            viterbi = emissions.new_full((valid_len, num_tags), float("-inf"))
            backpointers: list[torch.Tensor] = []
            viterbi[0] = emissions[b, 0] + self.transitions[self.start_idx, :num_tags]

            for t in range(1, valid_len):
                broadcast = viterbi[t - 1].unsqueeze(1) + self.transitions[:num_tags, :num_tags]
                best_scores, best_paths = broadcast.max(dim=0)
                viterbi[t] = best_scores + emissions[b, t]
                backpointers.append(best_paths)

            terminal = viterbi[valid_len - 1] + self.transitions[:num_tags, self.end_idx]
            best_last = int(terminal.argmax().item())
            best_path = [best_last]
            for bp in reversed(backpointers):
                best_last = int(bp[best_last].item())
                best_path.append(best_last)
            best_path.reverse()
            results.append(best_path)
        return results
