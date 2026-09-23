"""Three-way parity: Torch, MLX, and NumPy share one scorer contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from deep_green.numeric_policy import score_logits as numpy_score_logits

VOCAB = ["crane", "slate", "audio", "pouty", "irony", "metal", "aster", "stare"]
WIDTH = 64
TOLERANCE = 1e-5


def _deterministic_tensors() -> dict[str, np.ndarray]:
    from deep_green.model_spec import expected_shapes

    rng = np.random.default_rng(20260913)
    tensors = {
        name: rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)
        for name, shape in expected_shapes(WIDTH).items()
    }
    tensors["score.bias"] = np.linspace(-0.5, 0.5, WIDTH, dtype=np.float32)
    return tensors


@pytest.fixture(scope="module")
def shared_tensors():
    """One deterministic weight set serialized through safetensors."""
    import tempfile
    from pathlib import Path

    from deep_green.backends.storage import (
        load_policy_tensors,
        save_policy_tensors,
    )

    tensors = _deterministic_tensors()
    path = Path(tempfile.mkdtemp()) / "weights.safetensors"
    save_policy_tensors(path, tensors, metadata={"width": str(WIDTH)})
    return load_policy_tensors(path, expected_width=WIDTH)


def _parity_batch() -> dict[str, np.ndarray]:
    """Padded batch: row counts 0..5 in normal and hard mode (12 boards).

    Masked padding slots carry arbitrary varied nonzero values; row_mask
    selects each board's valid prefix.
    """
    letter_codes = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    mark_codes = {"b": 0, "y": 1, "g": 2}
    guesses = ["crane", "slate", "audio", "pouty", "irony", "metal"]
    marks = ["bbybb", "yyggb", "ggbbb", "byygb", "yyyyg", "bbbbb"]
    mode_rows = {"normal": (1.0, 0.0), "hard": (0.0, 1.0)}

    slots = 5
    rng = np.random.default_rng(7)
    batch = {
        "row_letters": rng.integers(1, 26, size=(12, slots, 5)).astype(np.int64),
        "row_marks": rng.integers(1, 3, size=(12, slots, 5)).astype(np.int64),
        "row_positions": np.broadcast_to(np.arange(5), (12, slots, 5)).copy(),
        "row_mask": np.zeros((12, slots), dtype=bool),
        "turn": np.zeros(12, dtype=np.float32),
        "mode": np.zeros((12, 2), dtype=np.float32),
        "word_letters": np.array(
            [[letter_codes[c] for c in word] for word in VOCAB], dtype=np.int64
        ),
        "word_positions": np.broadcast_to(np.arange(5), (len(VOCAB), 5)).copy(),
    }
    index = 0
    for mode_name in ("normal", "hard"):
        for row_count in range(6):
            for row in range(row_count):
                guess = guesses[(index + row) % len(guesses)]
                batch["row_letters"][index, row] = [
                    letter_codes[c] for c in guess
                ]
                batch["row_marks"][index, row] = [
                    mark_codes[c] for c in marks[(index + 2 * row) % len(marks)]
                ]
            batch["row_mask"][index, :row_count] = True
            batch["turn"][index] = float(row_count + 1)
            batch["mode"][index] = mode_rows[mode_name]
            index += 1
    return batch


def _torch_policy(tensors):
    from deep_green.backends.torch_policy import TorchWordlePolicy

    policy = TorchWordlePolicy(WIDTH)
    policy.load_exact({k: torch.from_numpy(v) for k, v in tensors.items()})
    return policy


def _mlx_policy(tensors):
    from deep_green.backends.mlx_policy import MlxWordlePolicy

    policy = MlxWordlePolicy(WIDTH)
    policy.load_exact({k: mx.array(v) for k, v in tensors.items()})
    return policy


def _score_all_backends(
    tensors: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    word_slice=slice(None),
) -> dict[str, np.ndarray]:
    results = {}
    with torch.no_grad():
        results["torch"] = _torch_policy(tensors).score_all(
            row_letters=torch.tensor(batch["row_letters"]),
            row_marks=torch.tensor(batch["row_marks"]),
            row_positions=torch.tensor(batch["row_positions"]),
            row_mask=torch.tensor(batch["row_mask"]),
            turn=torch.tensor(batch["turn"]),
            mode=torch.tensor(batch["mode"]),
            word_letters=torch.tensor(batch["word_letters"][word_slice]),
            word_positions=torch.tensor(batch["word_positions"][word_slice]),
        ).numpy()
    results["numpy"] = numpy_score_logits(
        tensors=tensors,
        row_letters=batch["row_letters"],
        row_marks=batch["row_marks"],
        row_positions=batch["row_positions"],
        row_mask=batch["row_mask"],
        turn=batch["turn"],
        mode=batch["mode"],
        word_letters=batch["word_letters"][word_slice],
        word_positions=batch["word_positions"][word_slice],
    )
    if HAS_MLX:
        results["mlx"] = np.asarray(
            _mlx_policy(tensors)(
                row_letters=mx.array(batch["row_letters"]),
                row_marks=mx.array(batch["row_marks"]),
                row_positions=mx.array(batch["row_positions"]),
                row_mask=mx.array(batch["row_mask"]),
                turn=mx.array(batch["turn"]),
                mode=mx.array(batch["mode"]),
                word_letters=mx.array(batch["word_letters"][word_slice]),
                word_positions=mx.array(batch["word_positions"][word_slice]),
            )
        )
    return results


def _assert_cross_backend_close(results: dict[str, np.ndarray]) -> None:
    reference = results["numpy"]
    for name, logits in results.items():
        delta = float(np.abs(logits - reference).max())
        assert delta <= TOLERANCE, f"{name} vs numpy max delta {delta} > {TOLERANCE}"


def _select_all_backends(
    tensors: dict[str, np.ndarray],
    results: dict[str, np.ndarray],
    board: int,
    legal_mask: np.ndarray,
) -> dict[str, int]:
    del tensors
    from deep_green.backends.torch_policy import (
        select_action as torch_select_action,
    )
    from deep_green.numeric_policy import (
        select_action as numpy_select_action,
    )

    choices = {
        "numpy": numpy_select_action(results["numpy"][board], legal_mask),
        "torch": torch_select_action(
            torch.tensor(results["torch"][board : board + 1]),
            torch.tensor(legal_mask[None]),
        ),
    }
    if HAS_MLX:
        from deep_green.backends.mlx_policy import (
            select_action as mlx_select_action,
        )

        choices["mlx"] = int(
            mlx_select_action(
                mx.array(results["mlx"][board : board + 1]),
                mx.array(legal_mask[None]),
            )
        )
    return choices


def test_three_way_logits_parity_over_padded_boards(shared_tensors):
    results = _score_all_backends(shared_tensors, _parity_batch())
    _assert_cross_backend_close(results)


def test_anagram_spellings_remain_distinct_across_backends(shared_tensors):
    results = _score_all_backends(shared_tensors, _parity_batch())
    for name, logits in results.items():
        assert not np.array_equal(logits[:, 6], logits[:, 7]), name


def test_nonzero_score_bias_is_action_varying_and_board_independent(shared_tensors):
    tensors = dict(shared_tensors)
    tensors["score.bilinear"] = np.zeros_like(tensors["score.bilinear"])
    results = _score_all_backends(tensors, _parity_batch())
    _assert_cross_backend_close(results)
    for name, logits in results.items():
        expected = np.broadcast_to(logits[:1], logits.shape)
        assert np.array_equal(logits, expected), f"{name} bias depends on board"
        assert np.ptp(logits[0]) > 0, f"{name} bias is not action-varying"


def test_action_permutation_permuted_logits_match(shared_tensors):
    order = np.array([2, 0, 7, 5, 3, 1, 4, 6])
    batch = _parity_batch()
    base = _score_all_backends(shared_tensors, batch)
    permuted = _score_all_backends(shared_tensors, batch, word_slice=order)
    for name, logits in base.items():
        delta = float(np.abs(permuted[name] - logits[:, order]).max())
        assert delta <= TOLERANCE, f"{name} permutation delta {delta}"
    _assert_cross_backend_close(permuted)


def test_chunked_scoring_concatenates_to_full(shared_tensors):
    batch = _parity_batch()
    full = _score_all_backends(shared_tensors, batch)
    slices = [slice(0, 3), slice(3, 6), slice(6, len(VOCAB))]
    joined = {}
    for name in full:
        parts = [
            _score_all_backends(shared_tensors, batch, word_slice=s)[name]
            for s in slices
        ]
        joined[name] = np.concatenate(parts, axis=1)
        delta = float(np.abs(joined[name] - full[name]).max())
        assert delta <= TOLERANCE, f"{name} chunk delta {delta}"
    _assert_cross_backend_close(joined)


def test_exact_tie_argmax_selects_lowest_legal_index(shared_tensors):
    tensors = dict(shared_tensors)
    tensors["score.bilinear"] = np.zeros_like(tensors["score.bilinear"])
    tensors["score.bias"] = np.zeros_like(tensors["score.bias"])
    results = _score_all_backends(tensors, _parity_batch())
    for name, logits in results.items():
        assert (logits == 0).all(), f"{name} logits are not exact ties"
    legal = np.ones(len(VOCAB), dtype=bool)
    legal[0] = False
    choices = _select_all_backends(tensors, results, board=0, legal_mask=legal)
    for name, choice in choices.items():
        assert choice == 1, f"{name} picked {choice}, expected lowest legal 1"


def test_legal_mask_argmax_identical_across_backends(shared_tensors):
    results = _score_all_backends(shared_tensors, _parity_batch())
    masks = [
        np.array([False, True, False, True, True, False, True, False]),
        np.array([True, False, False, True, False, False, True, True]),
    ]
    for legal in masks:
        for board in range(12):
            choices = _select_all_backends(
                shared_tensors, results, board=board, legal_mask=legal
            )
            assert len(set(choices.values())) == 1, (
                f"board {board} mask {legal.tolist()}: {choices}"
            )


def _board_view(batch: dict[str, np.ndarray], board: int, slots: int):
    view = dict(batch)
    for key in ("row_letters", "row_marks", "row_positions", "row_mask"):
        view[key] = batch[key][board : board + 1, :slots]
    view["turn"] = batch["turn"][board : board + 1]
    view["mode"] = batch["mode"][board : board + 1]
    return view


def test_masked_padding_matches_compact_rows(shared_tensors):
    """Compact valid rows and the same rows plus garbage padding agree."""
    batch = _parity_batch()
    board = 8
    row_count = int(batch["row_mask"][board].sum())
    compact = _board_view(batch, board, row_count)
    padded = _board_view(batch, board, 5)
    compact_scores = _score_all_backends(shared_tensors, compact)
    padded_scores = _score_all_backends(shared_tensors, padded)
    for name in compact_scores:
        delta = float(np.abs(compact_scores[name] - padded_scores[name]).max())
        assert delta <= TOLERANCE, f"{name} padding delta {delta}"


def test_all_false_empty_representations_match(shared_tensors):
    """Differently filled all-false boards produce identical logits."""
    batch = _parity_batch()
    one_slot = _board_view(batch, 0, 1)
    one_slot["row_letters"][:] = 0
    one_slot["row_marks"][:] = 0
    five_slots = _board_view(batch, 0, 5)
    scores_one = _score_all_backends(shared_tensors, one_slot)
    scores_five = _score_all_backends(shared_tensors, five_slots)
    for name in scores_one:
        delta = float(np.abs(scores_one[name] - scores_five[name]).max())
        assert delta <= TOLERANCE, f"{name} empty-fill delta {delta}"


def test_safetensors_headers_carry_the_full_contract(shared_tensors):
    import tempfile
    from pathlib import Path

    from safetensors import safe_open

    from deep_green.backends.storage import save_policy_tensors
    from deep_green.model_spec import tensor_names

    path = Path(tempfile.mkdtemp()) / "weights.safetensors"
    save_policy_tensors(path, shared_tensors, metadata={"width": str(WIDTH)})
    with safe_open(str(path), framework="numpy") as handle:
        assert set(handle.keys()) == set(tensor_names())
