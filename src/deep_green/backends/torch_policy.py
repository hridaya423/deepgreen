"""PyTorch CPU reference implementation of the DeepSets policy.

Forward pass, in order:
  1. Rows: each played row's five (letter, mark, position) slices embed
     through the shared tables, pass their own mixer+bias, combine through a
     rectifier, and sum/max-pool across rows through two mixers and a rectifier.
  2. Board: the pooled vector is gated by a turn gate, concatenated with the
     turn scalar and two mode scalars, projected to width d (relu), then one
     more d x d layer (relu).
  3. Words: each spelling's five (letter, position) slices embed, pass
     letter/position mixers (letter bias), combine through a rectifier, then
     sum/max-pool positions through rectified pool/key mixers and the output mixer.
  4. Scorer: logits[b, a] = board[b] . (W @ key[a]) + key[a] . score_bias.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from deep_green.model_spec import (
    expected_shapes,
    parameter_count,
    tensor_names,
)


class TorchWordlePolicy(nn.Module):
    """Scores every accepted spelling from public board bytes only."""

    def __init__(self, width: int) -> None:
        super().__init__()
        shapes = expected_shapes(width)
        self.width = width
        self.tensors: dict[str, Tensor] = {}
        for name in tensor_names():
            self.tensors[name] = nn.Parameter(torch.zeros(shapes[name]))
        with torch.no_grad():
            for parameter in self.tensors.values():
                parameter.uniform_(-0.05, 0.05)
        for name, parameter in self.tensors.items():
            setattr(self, f"t_{name.replace('.', '_')}", parameter)

    @property
    def device(self):
        return self.tensors[tensor_names()[0]].device

    def load_exact(self, tensors: dict[str, Tensor]) -> None:
        expected = set(tensor_names())
        given = set(tensors)
        if given != expected:
            raise ValueError(
                f"tensor mismatch: missing={sorted(expected - given)} "
                f"extra={sorted(given - expected)}"
            )
        with torch.no_grad():
            for name, value in tensors.items():
                target = self.tensors[name]
                if tuple(target.shape) != tuple(value.shape):
                    raise ValueError(
                        f"{name}: shape {tuple(value.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(value.to(dtype=target.dtype))

    def _encode_rows(
        self,
        row_letters: Tensor,
        row_marks: Tensor,
        row_positions: Tensor,
        row_mask: Tensor,
    ) -> Tensor:
        letter_part = (
            self.tensors["shared.letters"][row_letters] @ self.tensors["row.letter_mix"]
            + self.tensors["row.letter_bias"]
        )
        mark_part = (
            self.tensors["shared.marks"][row_marks] @ self.tensors["row.mark_mix"]
            + self.tensors["row.mark_bias"]
        )
        position_part = (
            self.tensors["shared.positions"][row_positions]
            @ self.tensors["row.position_mix"]
            + self.tensors["row.position_bias"]
        )
        per_slot = torch.relu(letter_part + mark_part + position_part)
        per_row = per_slot.sum(dim=2)
        mask = row_mask.bool().unsqueeze(-1)
        summed = (per_row * mask).sum(dim=1)
        maxed = per_row.masked_fill(~mask, -torch.inf).max(dim=1).values
        maxed = torch.where(
            row_mask.bool().any(dim=1, keepdim=True), maxed, torch.zeros_like(maxed)
        )
        return torch.relu(
            summed @ self.tensors["row.sum_mix"] + maxed @ self.tensors["row.max_mix"]
        )

    def _encode_board(self, pooled_board: Tensor, turn: Tensor, mode: Tensor) -> Tensor:
        gated = pooled_board * torch.sigmoid(self.tensors["board.turn_gate"])
        trunk_input = torch.cat([gated, turn.unsqueeze(1), mode], dim=1)
        hidden = torch.relu(trunk_input @ self.tensors["board.project"] + self.tensors["board.b1"])
        return torch.relu(hidden @ self.tensors["board.w1"] + self.tensors["board.b2"]) @ self.tensors["board.w2"]

    def _encode_words(self, word_letters: Tensor, word_positions: Tensor) -> Tensor:
        letter_embed = self.tensors["shared.letters"][word_letters]
        position_embed = self.tensors["shared.positions"][word_positions]
        letter_part = (
            letter_embed @ self.tensors["word.letter_mix"] + self.tensors["word.letter_bias"]
        )
        position_part = position_embed @ self.tensors["word.position_mix"]
        per_position = torch.relu(letter_part + position_part)
        summed = per_position.sum(dim=1)
        maxed = per_position.max(dim=1).values
        pooled = torch.relu(
            summed @ self.tensors["word.sum_mix"] + maxed @ self.tensors["word.max_mix"]
        )
        keys = torch.relu(pooled @ self.tensors["word.key_mix"])
        return keys @ self.tensors["word.out_mix"]

    def score_all(
        self,
        *,
        row_letters: Tensor,
        row_marks: Tensor,
        row_positions: Tensor,
        row_mask: Tensor,
        turn: Tensor,
        mode: Tensor,
        word_letters: Tensor,
        word_positions: Tensor,
    ) -> Tensor:
        """Return logits with shape [batch, action_count]."""
        pooled_board = self._encode_rows(row_letters, row_marks, row_positions, row_mask)
        board_state = self._encode_board(pooled_board, turn, mode)
        keys = self._encode_words(word_letters, word_positions)
        transformed_keys = (self.tensors["score.bilinear"] @ keys.T).T
        logits = board_state @ transformed_keys.T + keys @ self.tensors["score.bias"]
        return logits

    def parameter_report(self) -> dict:
        names = tensor_names()
        total = sum(self.tensors[n].numel() for n in names)
        unique = {id(self.tensors[n]) for n in names}
        return {
            "tensor_count": len(names),
            "unique_parameters": len(unique),
            "stored_scalars": total,
            "contract_count": parameter_count(self.width),
        }


def select_action(logits: Tensor, legal_mask: Tensor) -> int:
    masked = torch.where(
        legal_mask.bool(), logits, torch.full_like(logits, -float("inf"))
    )
    return int(torch.argmax(masked[0]))
