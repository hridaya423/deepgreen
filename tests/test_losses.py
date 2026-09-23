"""Loss tests: tie-aware BC and exact-teacher rank semantics."""

from __future__ import annotations

import pytest
import torch

from deep_green.losses import combined_loss, select_hard_negative_indices


def _setup(tie=(0, 1), negatives=(2, 3), vocab=6, legal=None):
    if legal is None:
        legal = [True] * vocab
    logits = torch.log(torch.tensor([[3.0, 3.0, 0.5, 0.4, 0.1, 0.05]]))
    mask = torch.tensor([legal])
    ties = torch.tensor(list(tie), dtype=torch.long)
    negs = torch.tensor(list(negatives), dtype=torch.long)
    return logits, mask, ties, negs


def test_bc_spread_equally_across_ties():
    logits = torch.zeros((1, 6))
    mask = torch.ones((1, 6), dtype=torch.bool)
    ties = torch.tensor([0, 1], dtype=torch.long)
    negs = torch.tensor([2, 3], dtype=torch.long)
    _, breakdown = combined_loss(
        logits=logits, legal_mask=mask, tie_indices=ties,
        negative_indices=negs, lambda_rank=0.0,
    )
    # With equal logits each action holds probability 1/6, so each tie's
    # log-prob is log(1/6) and BC (the mean over ties) is log(6).
    import math

    expected = math.log(6.0)
    assert abs(breakdown.tie_probability - 1 / 3) < 1e-5
    assert abs(breakdown.bc - expected) < 1e-4


def test_illegal_actions_have_zero_probability():
    logits, mask, ties, _ = _setup(legal=[True, True, False, False, True, True])
    _total, breakdown = combined_loss(
        logits=logits, legal_mask=mask, tie_indices=ties,
        negative_indices=torch.tensor([4, 5]), lambda_rank=0.0,
    )
    assert breakdown.argmax_legal_optimal
    masked = logits.masked_fill(~mask.bool(), float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    assert float(probs[0, 2]) == 0.0 and float(probs[0, 3]) == 0.0


def test_illegal_negative_is_rejected():
    logits, mask, ties, negatives = _setup(
        legal=[True, True, False, True, True, True]
    )
    with pytest.raises(ValueError, match="negative action must be legal"):
        combined_loss(
            logits=logits,
            legal_mask=mask,
            tie_indices=ties,
            negative_indices=negatives,
            lambda_rank=1.0,
        )


def test_rank_penalizes_negatives_above_tie_mass():
    logits, mask, ties, negs = _setup()
    _total_low, low = combined_loss(
        logits=logits, legal_mask=mask, tie_indices=ties,
        negative_indices=torch.tensor([4, 5]), lambda_rank=1.0,
    )
    _total_high, high = combined_loss(
        logits=logits, legal_mask=mask, tie_indices=ties,
        negative_indices=negs, lambda_rank=1.0,
    )
    assert high.rank > low.rank


def test_optimal_actions_never_in_negatives():
    logits = torch.zeros((1, 4))
    mask = torch.ones((1, 4), dtype=torch.bool)
    ties = torch.tensor([0], dtype=torch.long)
    negatives = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="optimal action"):
        combined_loss(
            logits=logits,
            legal_mask=mask,
            tie_indices=ties,
            negative_indices=negatives,
            lambda_rank=1.0,
        )


def test_hard_negative_selection_excludes_illegal_and_all_optimal_actions():
    words = ("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee")
    selected = select_hard_negative_indices(
        student_logits_row=torch.tensor([0.0, 100.0, 5.0, 4.0, 3.0]),
        action_words=words,
        tie_set=["aaaaa", "ccccc"],
        teacher_ranks={word: index for index, word in enumerate(words)},
        exact_values={word: (0, index) for index, word in enumerate(words)},
        keyed_sample=["bbbbb", "ddddd"],
        legal_mask=torch.tensor([True, False, True, True, True]),
        limit=4,
    )
    selected_words = {words[index] for index in selected.tolist()}
    assert not selected_words & {"aaaaa", "bbbbb", "ccccc"}
    assert selected_words == {"ddddd", "eeeee"}


def test_gradient_flows_only_through_legal_and_selected_paths():
    logits = torch.tensor([[1.0, 0.5, 0.25]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    ties = torch.tensor([0], dtype=torch.long)
    negs = torch.tensor([1], dtype=torch.long)
    total, _ = combined_loss(
        logits=logits, legal_mask=mask, tie_indices=ties,
        negative_indices=negs, lambda_rank=0.5,
    )
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 2]) == 0.0
