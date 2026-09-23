"""Tie-aware behavior cloning plus exact-teacher ranking losses.

For the complete optimal set A*:
  L_bc   = -mean(log_softmax(masked_logits)[a] for a in A*)
  z_tie  = logsumexp(logits[a] for a in A*) - log(|A*|)
  L_rank = mean(softplus(1 - z_tie + logits[n]) for n in hard_negatives)
Illegal actions carry zero probability and zero gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

RANK_MARGIN = 1.0


@dataclass(frozen=True)
class LossBreakdown:
    bc: float
    rank: float
    total: float
    tie_probability: float
    argmax_legal_optimal: bool


def masked_log_probs(logits: Tensor, legal_mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~legal_mask.bool(), float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def tie_bc_loss(log_probs: Tensor, tie_indices: Tensor) -> Tensor:
    return -(log_probs[0, tie_indices].mean())


def z_tie(logits: Tensor, tie_indices: Tensor) -> Tensor:
    return torch.logsumexp(logits[0, tie_indices], dim=0) - math.log(int(tie_indices.numel()))


def rank_loss(
    logits: Tensor, tie_indices: Tensor, negative_indices: Tensor, *, margin: float = RANK_MARGIN
) -> Tensor:
    if negative_indices.numel() == 0:
        return torch.zeros((), dtype=logits.dtype)
    target = torch.logsumexp(logits[0, tie_indices], dim=0) - math.log(
        int(tie_indices.numel())
    )
    penalties = torch.nn.functional.softplus(margin - target + logits[0, negative_indices])
    return penalties.mean()


def combined_loss(
    *,
    logits: Tensor,
    legal_mask: Tensor,
    tie_indices: Tensor,
    negative_indices: Tensor,
    lambda_rank: float,
) -> tuple[Tensor, LossBreakdown]:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("combined_loss requires one logit row")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal mask shape must equal logits shape")
    ties = [int(index) for index in tie_indices.tolist()]
    negatives = [int(index) for index in negative_indices.tolist()]
    if not ties or len(ties) != len(set(ties)):
        raise ValueError("optimal indices must be nonempty and unique")
    if len(negatives) != len(set(negatives)):
        raise ValueError("negative indices must be unique")
    if any(index < 0 or index >= logits.shape[1] for index in ties + negatives):
        raise ValueError("loss index outside the action space")
    if set(ties) & set(negatives):
        raise ValueError("an optimal action cannot be used as a negative")
    legal = legal_mask[0].bool()
    if not bool(legal.any()):
        raise ValueError("at least one legal action is required")
    if any(not bool(legal[index]) for index in ties):
        raise ValueError("every optimal action must be legal")
    if any(not bool(legal[index]) for index in negatives):
        raise ValueError("every negative action must be legal")
    log_probs = masked_log_probs(logits, legal_mask)
    bc = tie_bc_loss(log_probs, tie_indices)
    rank = rank_loss(logits, tie_indices, negative_indices)
    total = bc + lambda_rank * rank
    with torch.no_grad():
        masked = logits.masked_fill(~legal_mask.bool(), float("-inf"))
        argmax_index = int(torch.argmax(masked[0]))
        optimal = {int(i) for i in tie_indices.tolist()}
        tie_probability = float(torch.exp(log_probs[0, tie_indices]).sum())
    return total, LossBreakdown(
        bc=float(bc.detach()),
        rank=float(rank.detach()),
        total=float(total.detach()),
        tie_probability=tie_probability,
        argmax_legal_optimal=argmax_index in optimal,
    )


def select_hard_negative_indices(
    *,
    student_logits_row: Tensor,
    action_words: tuple[str, ...],
    tie_set: list[str],
    teacher_ranks: dict[str, int],
    exact_values: dict[str, tuple[int, int]],
    keyed_sample: list[str],
    legal_mask: Tensor,
    limit: int = 32,
) -> Tensor:
    from deep_green.examples import hard_negatives

    word_list = list(action_words)
    if legal_mask.ndim != 1 or legal_mask.shape[0] != len(word_list):
        raise ValueError("legal mask must be one-dimensional over the action space")
    student = {
        word: float(student_logits_row[index])
        for index, word in enumerate(word_list)
        if bool(legal_mask[index])
    }
    names = hard_negatives(
        student_logits=student,
        teacher_ranks=teacher_ranks,
        exact_values=exact_values,
        tie_set=tie_set,
        keyed_sample=keyed_sample,
        limit=limit,
    )
    index_of = {w: i for i, w in enumerate(word_list)}
    return torch.tensor([index_of[w] for w in names], dtype=torch.long)
