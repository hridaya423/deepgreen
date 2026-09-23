"""Metamorphic tests for the DeepSets policy and its tensor spec."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deep_green.backends.torch_policy import TorchWordlePolicy, select_action
from deep_green.model_spec import (
    RegisteredWidths,
    expected_shapes,
    parameter_count,
    save_config,
    tensor_names,
    verify_ladder_contract,
)

VOCAB = ["crane", "slate", "audio", "pouty", "irony", "metal"]


def _encode_board(rows: list[tuple[str, str]], turn: int):
    slots = max(len(rows), 1)
    letters = torch.zeros((1, slots, 5), dtype=torch.long)
    marks = torch.zeros((1, slots, 5), dtype=torch.long)
    positions = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(1, slots, 5).contiguous()
    row_mask = torch.zeros((1, slots), dtype=torch.bool)
    row_mask[0, : len(rows)] = True
    letter_codes = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    mark_codes = {"b": 0, "y": 1, "g": 2}
    for index, (guess_text, feedback_text) in enumerate(rows):
        for slot in range(5):
            letters[0, index, slot] = letter_codes[guess_text[slot]]
            marks[0, index, slot] = mark_codes[feedback_text[slot]]
    return letters, marks, positions, row_mask, turn


def _word_indices():
    letter_codes = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    wl = torch.tensor([[letter_codes[c] for c in word] for word in VOCAB], dtype=torch.long)
    wp = torch.arange(5).unsqueeze(0).expand(len(VOCAB), 5).contiguous()
    return wl, wp


def _logits(policy: TorchWordlePolicy, rows, mode: tuple[float, float]):
    rl, rm, rp, rmask, turn = _encode_board(rows, len(rows) + 1)
    wl, wp = _word_indices()
    turn_t = torch.tensor([float(turn)])
    mode_t = torch.tensor([mode], dtype=torch.float32)
    with torch.no_grad():
        return policy.score_all(
            row_letters=rl,
            row_marks=rm,
            row_positions=rp,
            row_mask=rmask,
            turn=turn_t,
            mode=mode_t,
            word_letters=wl,
            word_positions=wp,
        )


def test_ladder_contract_matches_frozen_counts_exactly():
    ladder = verify_ladder_contract()
    expected = {64: 64320, 96: 142560, 128: 251520, 192: 561600,
                256: 994560, 320: 1550400, 384: 2229120}
    for width, count in expected.items():
        assert parameter_count(width) == count
        assert width in ladder


def test_unregistered_width_is_refused():

    widths = RegisteredWidths.default().widths
    assert 448 not in widths
    with pytest.raises(ValueError):
        shapes = expected_shapes(448)  # noqa: F841 - refusal is the contract


def test_policy_parameter_count_matches_contract():
    policy = TorchWordlePolicy(64)
    report = policy.parameter_report()
    assert report["stored_scalars"] == 64320
    assert report["contract_count"] == 64320
    assert report["unique_parameters"] == len(tensor_names())


def test_row_permutation_invariance_at_fixed_turn():
    policy = TorchWordlePolicy(64)
    base = _logits(policy, [("crane", "bbybb"), ("slate", "byygb")], (1.0, 0.0))
    swapped = _logits(policy, [("slate", "byygb"), ("crane", "bbybb")], (1.0, 0.0))
    assert torch.equal(base, swapped)


def test_turn_input_changes_logits():
    early = _logits(policy := TorchWordlePolicy(64), [("crane", "bbybb")], (1.0, 0.0))
    later_rows = [("crane", "bbybb"), ("slate", "yyyyy"), ("audio", "yybbb")]
    late = _logits(policy, later_rows, (1.0, 0.0))
    assert not torch.allclose(early.mean(), late.mean(), atol=1e-9)


def test_mode_input_changes_logits():
    policy = TorchWordlePolicy(64)
    normal = _logits(policy, [("crane", "bbybb")], (1.0, 0.0))
    hard = _logits(policy, [("crane", "bbybb")], (0.0, 1.0))
    assert not torch.allclose(normal, hard, atol=1e-12)


def test_anagram_spellings_have_distinct_word_keys():
    policy = TorchWordlePolicy(64)
    letter_codes = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    words = ("aster", "stare")
    letters = torch.tensor([[letter_codes[c] for c in word] for word in words])
    positions = torch.arange(5).expand(2, 5).contiguous()
    with torch.no_grad():
        keys = policy._encode_words(letters, positions)
    assert not torch.equal(keys[0], keys[1])


def test_feedback_alignment_changes_row_encoding():
    policy = TorchWordlePolicy(64)
    first = _logits(policy, [("crane", "gbbbb")], (1.0, 0.0))
    second = _logits(policy, [("crane", "bgbbb")], (1.0, 0.0))
    assert not torch.equal(first, second)


def test_action_permutation_permutes_logits_by_spelling():
    policy = TorchWordlePolicy(64)
    rows = [("crane", "bbybb")]
    rl, rm, rp, rmask, turn = _encode_board(rows, len(rows) + 1)
    wl, wp = _word_indices()
    turn_t = torch.tensor([float(turn)])
    mode_t = torch.tensor([[1.0, 0.0]])
    with torch.no_grad():
        base = policy.score_all(row_letters=rl, row_marks=rm, row_positions=rp,
                                row_mask=rmask, turn=turn_t, mode=mode_t,
                                word_letters=wl, word_positions=wp)
        order = [2, 0, 5, 3, 1, 4]
        permuted = policy.score_all(row_letters=rl, row_marks=rm, row_positions=rp,
                                    row_mask=rmask, turn=turn_t, mode=mode_t,
                                    word_letters=wl[order], word_positions=wp[order])
    assert torch.allclose(base[0][order], permuted[0], atol=1e-6)


def test_chunk_size_invariance():
    """Per-word scores must not depend on how many words are scored at once."""
    policy = TorchWordlePolicy(64)
    rows = [("crane", "gggbb"), ("slate", "yyyyb"), ("pouty", "bbbbb")]
    rl, rm, rp, rmask, turn = _encode_board(rows, len(rows) + 1)
    wl, wp = _word_indices()
    turn_t = torch.tensor([float(turn)])
    mode_t = torch.tensor([[1.0, 0.0]])

    def run(word_slice):
        with torch.no_grad():
            return policy.score_all(
                row_letters=rl, row_marks=rm, row_positions=rp,
                row_mask=rmask, turn=turn_t, mode=mode_t,
                word_letters=wl[word_slice], word_positions=wp[word_slice],
            )

    full = run(slice(None))
    subset = run(slice(0, 3))
    assert torch.allclose(full[0][:3], subset[0], atol=1e-6)


def test_hidden_answer_swap_cannot_change_logits():
    """Public inputs fully determine logits; there is no answer input to vary."""
    policy = TorchWordlePolicy(64)
    rows = [("crane", "bbybb")]
    once = _logits(policy, rows, (1.0, 0.0))
    again = _logits(policy, rows, (1.0, 0.0))
    assert torch.equal(once, again)


def test_unknown_words_score_through_character_features(tmp_path):
    """Words outside the fixture vocabulary still score; no new parameters exist."""
    policy = TorchWordlePolicy(64)
    before = sum(p.numel() for p in policy.parameters())
    letter_codes = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    unknown = ["qajaq", "fuzzy"]
    wl = torch.tensor([[letter_codes[c] for c in w] for w in unknown], dtype=torch.long)
    wp = torch.arange(5).unsqueeze(0).expand(len(unknown), 5).contiguous()
    rows = [("crane", "bbybb")]
    rl, rm, rp, rmask, turn = _encode_board(rows, len(rows) + 1)
    with torch.no_grad():
        logits = policy.score_all(row_letters=rl, row_marks=rm, row_positions=rp,
                                  row_mask=rmask,
                                  turn=torch.tensor([float(turn)]),
                                  mode=torch.tensor([[1.0, 0.0]]),
                                  word_letters=wl, word_positions=wp)
    after = sum(p.numel() for p in policy.parameters())
    assert logits.shape == (1, len(unknown))
    assert before == after == 64320


def test_masked_padding_rows_contribute_nothing():
    """Masked-out slots with arbitrary values leave logits untouched."""
    policy = TorchWordlePolicy(64)
    rows = [("crane", "bbybb"), ("slate", "yyggb")]
    compact = _logits(policy, rows, (1.0, 0.0))
    rl, rm, rp, rmask, turn = _encode_board(rows, len(rows) + 1)
    generator = torch.Generator().manual_seed(99)
    pad = 3
    rl_pad = torch.cat([rl, torch.randint(0, 26, (1, pad, 5), generator=generator)], dim=1)
    rm_pad = torch.cat([rm, torch.randint(0, 3, (1, pad, 5), generator=generator)], dim=1)
    rp_pad = torch.cat(
        [rp, torch.arange(5).unsqueeze(0).expand(1, pad, 5)], dim=1
    )
    mask_pad = torch.cat([rmask, torch.zeros((1, pad), dtype=torch.bool)], dim=1)
    wl, wp = _word_indices()
    with torch.no_grad():
        padded = policy.score_all(
            row_letters=rl_pad, row_marks=rm_pad, row_positions=rp_pad,
            row_mask=mask_pad, turn=torch.tensor([float(turn)]),
            mode=torch.tensor([[1.0, 0.0]]), word_letters=wl, word_positions=wp,
        )
    assert torch.allclose(compact, padded, atol=1e-6)


def test_differently_filled_empty_boards_match():
    """Two all-false representations produce the exact same zero row pool."""
    policy = TorchWordlePolicy(64)
    compact_empty = _logits(policy, [], (1.0, 0.0))
    generator = torch.Generator().manual_seed(7)
    rl = torch.randint(0, 26, (1, 4, 5), generator=generator)
    rm = torch.randint(0, 3, (1, 4, 5), generator=generator)
    rp = torch.arange(5).unsqueeze(0).expand(1, 4, 5)
    rmask = torch.zeros((1, 4), dtype=torch.bool)
    wl, wp = _word_indices()
    with torch.no_grad():
        other_empty = policy.score_all(
            row_letters=rl, row_marks=rm, row_positions=rp, row_mask=rmask,
            turn=torch.tensor([1.0]), mode=torch.tensor([[1.0, 0.0]]),
            word_letters=wl, word_positions=wp,
        )
    assert torch.equal(compact_empty, other_empty)


def test_stable_lowest_index_argmax_on_exact_ties():
    logits = torch.tensor([[1.0, 3.0, 3.0, -1.0, 3.0]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    assert select_action(logits, mask) == 1


def test_save_config_round_trip(tmp_path):
    path = tmp_path / "model.json"
    save_config(path, width=64)
    import json

    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["parameter_count"] == 64320
    assert config["width"] == 64


def test_numpy_tensor_round_trip_via_safetensors(tmp_path):
    from safetensors.numpy import load_file

    from deep_green.backends.storage import save_policy_tensors

    policy = TorchWordlePolicy(64)
    tensors = {name: p.detach().numpy() for name, p in policy.tensors.items()}
    path = tmp_path / "weights.safetensors"
    save_policy_tensors(path, tensors, metadata={"width": "64"})
    restored = load_file(str(path))
    for name, value in tensors.items():
        np.testing.assert_array_equal(restored[name], value)
