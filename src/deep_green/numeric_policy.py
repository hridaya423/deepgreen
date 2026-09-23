"""Backend-free numpy forward pass used by the sandboxed policy worker.

Mirrors the torch/MLX implementations exactly over the shared tensor contract.
Word keys come from each spelling's letters and positions; the accepted
lexicon is public benchmark data mounted inside the sandbox, never an answer
artifact.
"""

from __future__ import annotations

import numpy as np


def score_logits(
    *,
    tensors: dict[str, np.ndarray],
    row_letters: np.ndarray,
    row_marks: np.ndarray,
    row_positions: np.ndarray,
    row_mask: np.ndarray,
    turn: np.ndarray,
    mode: np.ndarray,
    word_letters: np.ndarray,
    word_positions: np.ndarray,
) -> np.ndarray:
    """Return logits [batch, action_count]; word_* encode the full vocabulary."""
    letter_part = (
        tensors["shared.letters"][row_letters] @ tensors["row.letter_mix"]
        + tensors["row.letter_bias"]
    )
    mark_part = (
        tensors["shared.marks"][row_marks] @ tensors["row.mark_mix"] + tensors["row.mark_bias"]
    )
    position_part = (
        tensors["shared.positions"][row_positions] @ tensors["row.position_mix"]
        + tensors["row.position_bias"]
    )
    per_slot = np.maximum(letter_part + mark_part + position_part, 0.0)
    per_row = per_slot.sum(axis=2)
    mask = row_mask.astype(bool)[..., None]
    summed = (per_row * mask).sum(axis=1)
    maxed = np.where(mask, per_row, -np.inf).max(axis=1)
    maxed = np.where(
        row_mask.astype(bool).any(axis=1, keepdims=True), maxed, np.zeros_like(maxed)
    )
    pooled = np.maximum(
        summed @ tensors["row.sum_mix"] + maxed @ tensors["row.max_mix"], 0.0
    )

    gated = pooled * _sigmoid(tensors["board.turn_gate"])
    extras = np.concatenate(
        [turn[:, None].astype(np.float32), mode.astype(np.float32)], axis=1
    )
    trunk_input = np.concatenate([gated, extras], axis=1).astype(np.float32)
    hidden = np.maximum(trunk_input @ tensors["board.project"] + tensors["board.b1"], 0.0)
    board_state = np.maximum(
        hidden @ tensors["board.w1"] + tensors["board.b2"], 0.0
    ) @ tensors["board.w2"]

    keys = _word_keys(tensors, word_letters, word_positions)
    transformed_keys = (tensors["score.bilinear"] @ keys.T).T
    return board_state @ transformed_keys.T + keys @ tensors["score.bias"]


def select_action(logits: np.ndarray, legal_mask: np.ndarray) -> int:
    """Lowest-index masked argmax, matching the torch/MLX selectors."""
    masked = np.where(np.asarray(legal_mask).astype(bool), logits, -np.inf)
    return int(np.argmax(masked))


def encode_lexicon(words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    codes = {chr(ord("a") + i): i for i in range(26)}
    letters = np.array([[codes[c] for c in w] for w in words], dtype=np.int64)
    positions = np.broadcast_to(np.arange(5), (len(words), 5)).copy()
    return letters, positions


def _word_keys(
    tensors: dict[str, np.ndarray],
    word_letters: np.ndarray,
    word_positions: np.ndarray,
) -> np.ndarray:
    letter_embed = tensors["shared.letters"][word_letters]
    position_embed = tensors["shared.positions"][word_positions]
    letter_part = (
        letter_embed @ tensors["word.letter_mix"] + tensors["word.letter_bias"]
    )
    position_part = position_embed @ tensors["word.position_mix"]
    per_position = np.maximum(letter_part + position_part, 0.0)
    summed = per_position.sum(axis=1)
    maxed = per_position.max(axis=1)
    pooled = np.maximum(
        summed @ tensors["word.sum_mix"] + maxed @ tensors["word.max_mix"], 0.0
    )
    keys = np.maximum(pooled @ tensors["word.key_mix"], 0.0)
    return keys @ tensors["word.out_mix"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))
