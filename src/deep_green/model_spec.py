"""Tensor specification shared by every policy backend.

Architecture (compositional DeepSets; no word-ID rows, answer embeddings,
frequency features, or history lookups):

- One shared table set: letter (26d), mark (3d), position (5d) embeddings.
- Row encoder: per-row letter, mark, and position slices pass through their
  own d x d mixers with biases, combine through a rectifier, then pool through
  sum and max mixers followed by a rectifier into a board vector.
- Board trunk: pooled board plus turn (1) and mode (2) scalars projects to
  width d through ``board.project``, then two more d x d layers.
- Word encoder: each spelling's five letter/position slices share the same
  tables, pass letter- and position-mixers with biases, combine through a
  rectifier, sum/max-pool through rectified mixers, then finish with a rectified
  key mixer and the output mixer.
- Scorer: bilinear key^T W b plus a per-word bias; no board-only or global
  shift terms exist because they cannot change a policy decision.

Parameter count is exactly 15d^2 + 45d at state width d: fourteen d x d
matrices plus one (d+3) x d projector that contributes the fifteenth d^2 and
carries the turn and mode scalars in its three extra input rows (row
letter/mark/position/sum/max = 5; board project/w1/w2 = 3; word
letter/position/sum/max/key/out = 6; score bilinear = 1), the shared tables
at (26+3+5)d = 34d, the projector's three scalar-input rows = 3d, and eight
width-d vectors (three row biases, two trunk biases, one turn gate, one word
bias, one score bias) = 8d. Mode enters only through the projector's extra
rows, so mode sensitivity is testable by zeroing those inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TENSOR_SPEC_VERSION = "wordle.policy-spec/v3"

LETTERS = 26
MARKS = 3
POSITIONS = 5
ROWS = 6
TURN_INPUTS = 1
MODE_INPUTS = 2


def parameter_count(width: int) -> int:
    """Exact learned-parameter count for a state width."""
    return 15 * width * width + 45 * width


@dataclass(frozen=True)
class RegisteredWidths:
    widths: tuple[int, ...]

    @staticmethod
    def default() -> RegisteredWidths:
        return RegisteredWidths((64, 96, 128, 192, 256, 320, 384))


def tensor_names() -> tuple[str, ...]:
    """The complete parameter-name contract shared by both backends."""
    return (
        "shared.letters",
        "shared.marks",
        "shared.positions",
        "row.letter_mix",
        "row.mark_mix",
        "row.position_mix",
        "row.sum_mix",
        "row.max_mix",
        "row.letter_bias",
        "row.mark_bias",
        "row.position_bias",
        "board.project",
        "board.w1",
        "board.w2",
        "board.b1",
        "board.b2",
        "board.turn_gate",
        "word.letter_mix",
        "word.position_mix",
        "word.sum_mix",
        "word.max_mix",
        "word.key_mix",
        "word.out_mix",
        "word.letter_bias",
        "score.bilinear",
        "score.bias",
    )


def expected_shapes(width: int) -> dict[str, tuple[int, ...]]:
    if width not in RegisteredWidths.default().widths:
        raise ValueError(
            f"width {width} is not on the registered model ladder "
            f"{RegisteredWidths.default().widths}"
        )
    shapes = {
        "shared.letters": (LETTERS, width),
        "shared.marks": (MARKS, width),
        "shared.positions": (POSITIONS, width),
        "row.letter_mix": (width, width),
        "row.mark_mix": (width, width),
        "row.position_mix": (width, width),
        "row.sum_mix": (width, width),
        "row.max_mix": (width, width),
        "row.letter_bias": (width,),
        "row.mark_bias": (width,),
        "row.position_bias": (width,),
        "board.project": (width + TURN_INPUTS + MODE_INPUTS, width),
        "board.w1": (width, width),
        "board.w2": (width, width),
        "board.b1": (width,),
        "board.b2": (width,),
        "board.turn_gate": (width,),
        "word.letter_mix": (width, width),
        "word.position_mix": (width, width),
        "word.sum_mix": (width, width),
        "word.max_mix": (width, width),
        "word.key_mix": (width, width),
        "word.out_mix": (width, width),
        "word.letter_bias": (width,),
        "score.bilinear": (width, width),
        "score.bias": (width,),
    }
    if len(shapes) != len(tensor_names()):
        raise ValueError("shape layout disagrees with tensor_names")
    return shapes


def count_parameters(shapes: dict[str, tuple[int, ...]]) -> int:
    total = 0
    for shape in shapes.values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    return total


def verify_ladder_contract() -> dict[int, dict[str, tuple[int, ...]]]:
    ladder: dict[int, dict[str, tuple[int, ...]]] = {}
    for width in RegisteredWidths.default().widths:
        shapes = expected_shapes(width)
        counted = count_parameters(shapes)
        if counted != parameter_count(width):
            raise ValueError(
                f"tensor spec at width {width} counts {counted} parameters but "
                f"the frozen contract requires exactly {parameter_count(width)}"
            )
        ladder[width] = shapes
    return ladder


def save_config(path: Path, *, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": TENSOR_SPEC_VERSION,
                "width": width,
                "parameter_count": parameter_count(width),
                "tensor_names": list(tensor_names()),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
