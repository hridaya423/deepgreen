import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

WORD_LENGTH = 5
MAX_ROWS = 6
LOWERCASE_ASCII = frozenset("abcdefghijklmnopqrstuvwxyz")


class Mode(StrEnum):
    NORMAL = "normal"
    HARD = "hard"


class Tile(StrEnum):
    ABSENT = "b"
    PRESENT = "y"
    CORRECT = "g"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        canonical = (Tile.ABSENT, Tile.PRESENT, Tile.CORRECT)
        return canonical.index(self) < canonical.index(other)


@dataclass(frozen=True)
class Word5:
    value: str

    @staticmethod
    def parse(text: str) -> "Word5":
        if len(text) != WORD_LENGTH or any(char not in LOWERCASE_ASCII for char in text):
            raise ValueError(f"word must be exactly {WORD_LENGTH} lowercase ASCII letters: {text!r}")
        return Word5(text)


@dataclass(frozen=True)
class Feedback:
    tiles: tuple[Tile, ...]

    @staticmethod
    def parse(marks: str) -> "Feedback":
        tiles = tuple(_tile_for(mark) for mark in marks)
        if len(tiles) != WORD_LENGTH:
            raise ValueError(f"feedback must have exactly {WORD_LENGTH} marks: {marks!r}")
        return Feedback(tiles)


def _tile_for(mark: str) -> Tile:
    try:
        return Tile(mark)
    except ValueError:
        raise ValueError(f"unknown tile mark: {mark!r}") from None


@dataclass(frozen=True)
class GuessRow:
    guess: Word5
    feedback: Feedback


def _is_solved(row: GuessRow) -> bool:
    return all(tile is Tile.CORRECT for tile in row.feedback.tiles)


@dataclass(frozen=True)
class Board:
    mode: Mode
    rows: tuple[GuessRow, ...]

    @staticmethod
    def create(mode: Mode, rows: tuple[GuessRow, ...], next_turn: int) -> "Board":
        rows = tuple(rows)
        if len(rows) > MAX_ROWS:
            raise ValueError(f"board cannot hold more than {MAX_ROWS} rows")
        solved_seen = False
        for row in rows:
            if solved_seen:
                raise ValueError("board cannot hold rows after a solved state")
            solved_seen = _is_solved(row)
        if next_turn != len(rows) + 1:
            raise ValueError(f"next_turn must equal {len(rows) + 1} for this board")
        return Board(mode, rows)


def score_guess(guess: Word5, answer: Word5) -> Feedback:
    marks = [Tile.ABSENT] * WORD_LENGTH
    remaining = Counter(answer.value)
    for index, letter in enumerate(guess.value):
        if answer.value[index] == letter:
            marks[index] = Tile.CORRECT
            remaining[letter] -= 1
    for index, letter in enumerate(guess.value):
        if marks[index] is Tile.CORRECT:
            continue
        if remaining[letter] > 0:
            marks[index] = Tile.PRESENT
            remaining[letter] -= 1
    return Feedback(tuple(marks))


@dataclass(frozen=True)
class ActionSpace:
    words: tuple[str, ...]
    digest: str

    @staticmethod
    def parse(words: Iterable[str]) -> "ActionSpace":
        seen: set[str] = set()
        canonical: list[str] = []
        for word in words:
            parsed = Word5.parse(word)
            if parsed.value in seen:
                raise ValueError(f"duplicate action word: {parsed.value!r}")
            seen.add(parsed.value)
            canonical.append(parsed.value)
        canonical.sort()
        spellings = tuple(canonical)
        canonical_bytes = ("\n".join(canonical) + "\n").encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()
        return ActionSpace(spellings, digest)

    def mask(
        self,
        values: Sequence[bool],
        *,
        action_space_digest: str | None = None,
    ) -> tuple[bool, ...]:
        if len(values) != len(self.words):
            raise ValueError(
                f"legality mask must have exactly {len(self.words)} entries for this action space"
            )
        expected_digest = action_space_digest if action_space_digest is not None else self.digest
        if expected_digest != self.digest:
            raise ValueError("action-space digest mismatch")
        return tuple(bool(value) for value in values)


def _candidate_spellings(actions: ActionSpace | Sequence[str]) -> Sequence[str]:
    if isinstance(actions, ActionSpace):
        return actions.words
    return actions


def is_legal_normal(board: Board, actions: ActionSpace | Sequence[str]) -> list[bool]:
    del board
    return [True] * len(_candidate_spellings(actions))


def _revealed_requirements(rows: tuple[GuessRow, ...]) -> dict[str, int]:
    required: dict[str, int] = {}
    for row in rows:
        revealed: Counter[str] = Counter()
        for letter, tile in zip(row.guess.value, row.feedback.tiles):
            if tile is Tile.PRESENT or tile is Tile.CORRECT:
                revealed[letter] += 1
        for letter, count in revealed.items():
            required[letter] = max(required.get(letter, 0), count)
    return required


def is_legal_nyt_standard_hard(board: Board, actions: ActionSpace | Sequence[str]) -> list[bool]:
    words = _candidate_spellings(actions)
    greens: list[tuple[int, str]] = []
    for row in board.rows:
        for index, (letter, tile) in enumerate(zip(row.guess.value, row.feedback.tiles)):
            if tile is Tile.CORRECT:
                greens.append((index, letter))
    required = _revealed_requirements(board.rows)
    legal: list[bool] = []
    for word in words:
        ok = True
        for index, letter in greens:
            if word[index] != letter:
                ok = False
                break
        if ok:
            available = Counter(word)
            for letter, need in required.items():
                if available[letter] < need:
                    ok = False
                    break
        legal.append(ok)
    return legal


def board_from_public_state(*, mode: str, turn: int, rows: Sequence[Sequence[str]]) -> Board:
    if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= MAX_ROWS:
        raise ValueError(f"pending turn must be an integer from 1 through {MAX_ROWS}")
    parsed_rows: list[GuessRow] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each public row must contain exactly guess and feedback")
        guess, marks = row
        if not isinstance(guess, str) or not isinstance(marks, str):
            raise TypeError("public row guess and feedback must be strings")
        parsed_rows.append(GuessRow(Word5.parse(guess), Feedback.parse(marks)))
    try:
        parsed_mode = Mode(mode)
    except ValueError:
        raise ValueError(f"unknown mode: {mode!r}") from None
    return Board.create(parsed_mode, tuple(parsed_rows), turn)


def legal_actions(board: Board, actions: ActionSpace | Sequence[str]) -> list[bool]:
    if board.mode is Mode.HARD:
        return is_legal_nyt_standard_hard(board, actions)
    return is_legal_normal(board, actions)


def legality_digest(mask: Sequence[bool]) -> str:
    if any(not isinstance(value, bool) for value in mask):
        raise ValueError("legality mask values must be bools")
    return hashlib.sha256("".join("1" if value else "0" for value in mask).encode()).hexdigest()
