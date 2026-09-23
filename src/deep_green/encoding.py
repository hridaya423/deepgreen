"""Shared encoders turning public board bytes into backend-agnostic indices."""

from __future__ import annotations

from dataclasses import dataclass

LETTER_CODES = {chr(ord("a") + index): index for index in range(26)}
MARK_CODES = {"b": 0, "y": 1, "g": 2}
MODE_CODES = {"normal": (1.0, 0.0), "hard": (0.0, 1.0)}


class EncodingError(ValueError):
    pass


@dataclass(frozen=True)
class BoardEncoding:
    row_letters: list[list[int]]
    row_marks: list[list[int]]
    row_positions: tuple[int, ...]
    row_mask: list[bool]
    turn: int


def encode_rows(rows: list[tuple[str, str]], max_turns: int = 6) -> BoardEncoding:
    if len(rows) >= max_turns:
        raise EncodingError("board already at the horizon; no pending turn")
    letters: list[list[int]] = []
    marks: list[list[int]] = []
    for guess_text, feedback_text in rows:
        if len(guess_text) != 5 or len(feedback_text) != 5:
            raise EncodingError(f"malformed row {guess_text!r}/{feedback_text!r}")
        letters.append([LETTER_CODES[c] for c in guess_text])
        marks.append([MARK_CODES[c] for c in feedback_text])
    return BoardEncoding(
        row_letters=letters,
        row_marks=marks,
        row_positions=tuple(range(5)),
        row_mask=[True] * len(rows),
        turn=len(rows) + 1,
    )


def encode_word(word: str) -> tuple[list[int], list[int]]:
    if len(word) != 5:
        raise EncodingError(f"word {word!r} is not five letters")
    return [LETTER_CODES[c] for c in word], list(range(5))
