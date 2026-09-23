"""End-to-end oracle smoke proof over a tiny fixed vocabulary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from deep_green.oracle.adapter import OracleRequest, OracleResult, run_oracle
from deep_green.rules import (
    ActionSpace,
    Board,
    Feedback,
    GuessRow,
    Mode,
    Tile,
    Word5,
    is_legal_normal,
    is_legal_nyt_standard_hard,
    score_guess,
)

SMOKE_ACTIONS = ("audio", "crane", "pouty", "slate")
SMOKE_ANSWERS = ("slate", "pouty")
SMOKE_PRIOR = "uniform-v1"
INFINITY = 10**9


class SmokeMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeReport:
    mode: str
    solver: OracleResult
    python_objective: int
    optimal_first_words: tuple[str, ...]


def _sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _partition(
    guess: Word5, candidates: tuple[str, ...]
) -> dict[tuple[int, ...], list[str]]:
    classes: dict[tuple[int, ...], list[str]] = {}
    for answer in candidates:
        feedback = score_guess(guess, Word5.parse(answer))
        key = tuple(tile.value for tile in feedback.tiles)
        classes.setdefault(key, []).append(answer)
    return classes


def _is_solved(key: tuple[int, ...]) -> bool:
    return all(value == Tile.CORRECT.value for value in key)


def exact_policy_value(
    candidates: tuple[str, ...],
    *,
    mode: Mode,
    action_words: tuple[str, ...],
    board_rows: tuple[GuessRow, ...] = (),
    depth: int = 6,
) -> int:
    """Independent exhaustive minimum total-guess count over the tiny cell."""
    if not candidates:
        return 0
    if depth <= 0:
        return INFINITY
    if len(candidates) == 1:
        return 1
    board = Board.create(mode, board_rows, len(board_rows) + 1)
    legality = (
        is_legal_nyt_standard_hard(board, action_words)
        if mode is Mode.HARD
        else is_legal_normal(board, action_words)
    )
    best = INFINITY
    for index, word in enumerate(action_words):
        if not legality[index]:
            continue
        guess = Word5.parse(word)
        total = 0
        for key, group in _partition(guess, candidates).items():
            if _is_solved(key):
                total += len(group)
                continue
            sub_value = exact_policy_value(
                tuple(group),
                mode=mode,
                action_words=action_words,
                board_rows=_extended(board_rows, guess, key),
                depth=depth - 1,
            )
            if sub_value >= INFINITY:
                total = INFINITY
                break
            total += len(group) + sub_value
            if total >= best:
                break
        best = min(best, total)
    return best


def _extended(
    board_rows: tuple[GuessRow, ...], guess: Word5, key: tuple[int, ...]
) -> tuple[GuessRow, ...]:
    tiles = {
        Tile.ABSENT.value: Tile.ABSENT,
        Tile.PRESENT.value: Tile.PRESENT,
        Tile.CORRECT.value: Tile.CORRECT,
    }
    marks = tuple(tiles[value] for value in key)
    return board_rows + (GuessRow(guess, Feedback(marks)),)


def optimal_first_words(
    *, mode: Mode, action_words: tuple[str, ...], answers: tuple[str, ...], depth: int = 6
) -> tuple[int, tuple[str, ...]]:
    """Return (optimal objective, sorted tie set of optimal opening words)."""
    scores: list[tuple[int, str]] = []
    for word in action_words:
        guess = Word5.parse(word)
        total = 0
        for key, group in _partition(guess, answers).items():
            if _is_solved(key):
                total += len(group)
                continue
            sub_value = exact_policy_value(
                tuple(group),
                mode=mode,
                action_words=action_words,
                board_rows=_extended((), guess, key),
                depth=depth - 1,
            )
            total += len(group) + sub_value
        scores.append((total, word))
    best_total = min(score for score, _ in scores)
    winners = tuple(sorted(word for score, word in scores if score == best_total))
    return best_total, winners


def run_smoke(*, repo_root: Path, work_dir: Path, mode: str) -> SmokeReport:
    action_space = ActionSpace.parse(SMOKE_ACTIONS)
    from deep_green.oracle.build import ensure_solver, vendor_dir

    exe_path, binary_digest, compiler_id = ensure_solver(
        vendor_dir(repo_root), work_dir / "build"
    )
    del exe_path
    request = OracleRequest(
        mode=mode,
        horizon=6,
        rules_sha256=_sha_hex(f"openwordle-rules-v1:{mode}"),
        prior_sha256=_sha_hex(f"{SMOKE_PRIOR}:{mode}"),
        action_space_sha256=action_space.digest,
        answer_set_sha256=_sha_hex("\n".join(SMOKE_ANSWERS)),
        executable_sha256=binary_digest,
        compiler_id=compiler_id,
        solver_source_sha256=_sha_hex(
            (vendor_dir(repo_root) / "wordle.cpp").read_text(encoding="utf-8")
        ),
        action_words=action_space.words,
        answer_words=SMOKE_ANSWERS,
    )
    result = run_oracle(request, repo_root=repo_root, work_dir=work_dir / "runs")

    mode_value = Mode.NORMAL if mode == "normal" else Mode.HARD
    python_objective, winners = optimal_first_words(
        mode=mode_value, action_words=action_space.words, answers=SMOKE_ANSWERS
    )
    if python_objective != result.objective:
        raise SmokeMismatch(
            f"independent python objective {python_objective} contradicts solver "
            f"objective {result.objective} in {mode} mode"
        )
    if result.best_first_word not in winners:
        raise SmokeMismatch(
            f"solver opening {result.best_first_word!r} is outside the optimal tie set {winners}"
        )
    return SmokeReport(
        mode=mode,
        solver=result,
        python_objective=python_objective,
        optimal_first_words=winners,
    )


def format_report(report: SmokeReport) -> str:
    lines = [
        f"oracle smoke mode={report.mode}",
        (
            f"  solver objective={report.solver.objective} "
            f"best_first={report.solver.best_first_word} nodes={report.solver.nodes}"
        ),
        f"  python objective={report.python_objective}",
        f"  optimal openings={list(report.optimal_first_words)}",
        "  OK",
    ]
    return "\n".join(lines)
