"""MLX implementation of the DeepSets policy from the shared tensor spec."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from deep_green.model_spec import (
    expected_shapes,
    parameter_count,
    tensor_names,
)

_relu = nn.relu


def _parameter_dict(width: int) -> dict[str, mx.array]:
    shapes = expected_shapes(width)
    return {
        name: mx.random.uniform(
            low=-0.05,
            high=0.05,
            shape=shape,
        ).astype(mx.float32)
        for name, shape in shapes.items()
    }


class MlxWordlePolicy(nn.Module):
    """Mirror of TorchWordlePolicy; identical tensor names and math."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        for name, value in _parameter_dict(width).items():
            setattr(self, name.replace(".", "_"), value)

    def parameters_dict(self) -> dict[str, mx.array]:
        return {
            name: getattr(self, name.replace(".", "_")) for name in tensor_names()
        }

    def load_exact(self, tensors: dict[str, mx.array]) -> None:
        expected = set(tensor_names())
        given = set(tensors)
        if given != expected:
            raise ValueError(
                f"tensor mismatch: missing={sorted(expected - given)} "
                f"extra={sorted(given - expected)}"
            )
        for name, value in tensors.items():
            attribute = name.replace(".", "_")
            target_shape = tuple(getattr(self, attribute).shape)
            if target_shape != tuple(value.shape):
                raise ValueError(f"{name}: shape {tuple(value.shape)} != {target_shape}")
            setattr(self, attribute, value.astype(mx.float32))

    def __call__(
        self,
        *,
        row_letters: mx.array,
        row_marks: mx.array,
        row_positions: mx.array,
        row_mask: mx.array,
        turn: mx.array,
        mode: mx.array,
        word_letters: mx.array,
        word_positions: mx.array,
    ) -> mx.array:
        pooled_board = self._encode_rows(row_letters, row_marks, row_positions, row_mask)
        board_state = self._encode_board(pooled_board, turn, mode)
        keys = self._encode_words(word_letters, word_positions)
        p = self.parameters_dict()
        transformed_keys = (p["score.bilinear"] @ keys.T).T
        logits = board_state @ transformed_keys.T + keys @ p["score.bias"]
        return logits

    def _encode_rows(self, row_letters, row_marks, row_positions, row_mask) -> mx.array:
        p = self.parameters_dict()
        letter_part = (
            p["shared.letters"][row_letters] @ p["row.letter_mix"] + p["row.letter_bias"]
        )
        mark_part = p["shared.marks"][row_marks] @ p["row.mark_mix"] + p["row.mark_bias"]
        position_part = (
            p["shared.positions"][row_positions] @ p["row.position_mix"]
            + p["row.position_bias"]
        )
        per_slot = _relu(letter_part + mark_part + position_part)
        per_row = per_slot.sum(axis=2)
        mask = row_mask.astype(mx.bool_)[:, :, None]
        summed = (per_row * mask).sum(axis=1)
        maxed = mx.where(mask, per_row, mx.array(-mx.inf)).max(axis=1)
        maxed = mx.where(
            mx.any(row_mask.astype(mx.bool_), axis=1, keepdims=True),
            maxed,
            mx.zeros_like(maxed),
        )
        return _relu(summed @ p["row.sum_mix"] + maxed @ p["row.max_mix"])

    def _encode_board(self, pooled_board, turn, mode) -> mx.array:
        p = self.parameters_dict()
        gated = pooled_board * mx.sigmoid(p["board.turn_gate"])
        extras = mx.concatenate([turn[:, None], mode], axis=1)
        trunk_input = mx.concatenate([gated, extras], axis=1)
        hidden = _relu(trunk_input @ p["board.project"] + p["board.b1"])
        return _relu(hidden @ p["board.w1"] + p["board.b2"]) @ p["board.w2"]

    def _encode_words(self, word_letters, word_positions) -> mx.array:
        p = self.parameters_dict()
        letter_embed = p["shared.letters"][word_letters]
        position_embed = p["shared.positions"][word_positions]
        letter_part = letter_embed @ p["word.letter_mix"] + p["word.letter_bias"]
        position_part = position_embed @ p["word.position_mix"]
        per_position = _relu(letter_part + position_part)
        summed = per_position.sum(axis=1)
        maxed = per_position.max(axis=1)
        pooled = _relu(summed @ p["word.sum_mix"] + maxed @ p["word.max_mix"])
        keys = _relu(pooled @ p["word.key_mix"])
        return keys @ p["word.out_mix"]

    def parameter_report(self) -> dict:
        names = tensor_names()
        arrays = self.parameters_dict()
        total = sum(int(a.size) for a in (arrays[n] for n in names))
        return {
            "tensor_count": len(names),
            "unique_parameters": len({n for n in names}),
            "stored_scalars": total,
            "contract_count": parameter_count(self.width),
        }


def select_action(logits: mx.array, legal_mask: mx.array) -> int:
    masked = mx.where(legal_mask.astype(mx.bool_), logits, mx.array(-mx.inf))
    return int(mx.argmax(masked[0]).item())
