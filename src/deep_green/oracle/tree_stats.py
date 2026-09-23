"""Per-answer statistics for a parsed policy tree, via rules.py feedback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_green.rules import Feedback, Tile, Word5, score_guess


class TreeStatsError(ValueError):
    pass


def count_policy_states(tree: dict) -> int:
    """Count nonterminal public states in a verified policy tree."""
    if "terminal" in tree:
        return 0
    children = tree.get("children", {})
    if not isinstance(children, dict):
        raise TreeStatsError("children must be a mapping")
    return 1 + sum(count_policy_states(child) for child in children.values())


@dataclass(frozen=True)
class TreeStats:
    nodes: int
    answers_reached: int
    total_score: int
    losses: int
    worst_score: int


_CODE_TO_TILE = {0: Tile.ABSENT, 1: Tile.PRESENT, 2: Tile.CORRECT}


def _key_of(feedback: Feedback) -> str:
    order = {Tile.ABSENT: "0", Tile.PRESENT: "1", Tile.CORRECT: "2"}
    return "".join(order[tile] for tile in feedback.tiles)


def collect_tree_stats(
    tree: dict,
    *,
    frozen_corpus_dir: Path,
    answers_file: str,
    horizon_turns: int,
) -> TreeStats:
    answer_words = tuple(
        (frozen_corpus_dir / answers_file).read_text(encoding="utf-8").split()
    )
    stats = TreeStats(0, 0, 0, 0, 0)

    def walk(node: dict, remaining: tuple[str, ...], turn: int) -> None:
        nonlocal stats
        stats = TreeStats(
            stats.nodes + 1,
            stats.answers_reached,
            stats.total_score,
            stats.losses,
            stats.worst_score,
        )
        guess_text = node.get("guess")
        if not isinstance(guess_text, str):
            raise TreeStatsError("branch node missing guess word")
        guess_word = Word5.parse(guess_text)
        children = node.get("children", {})
        if not isinstance(children, dict):
            raise TreeStatsError("children must be a mapping")
        partition: dict[str, list[str]] = {}
        for answer in remaining:
            key = _key_of(score_guess(guess_word, Word5.parse(answer)))
            partition.setdefault(key, []).append(answer)
        covered = set(children.keys())
        expected = set(partition.keys())
        if expected - covered:
            raise TreeStatsError(
                f"incomplete partition at turn {turn}: {sorted(expected - covered)}"
            )
        if covered - expected:
            raise TreeStatsError(
                f"impossible classes at turn {turn}: {sorted(covered - expected)}"
            )
        for key, child in children.items():
            group = partition[key]
            terminal = child.get("terminal")
            if key == "22222" and terminal is not None:
                stats = TreeStats(
                    stats.nodes + len(group),
                    stats.answers_reached + len(group),
                    stats.total_score + int(child.get("turn", turn)) * len(group),
                    stats.losses,
                    max(stats.worst_score, int(child.get("turn", turn))),
                )
                continue
            if terminal == "loss":
                stats = TreeStats(
                    stats.nodes,
                    stats.answers_reached,
                    stats.total_score,
                    stats.losses + len(group),
                    stats.worst_score,
                )
                continue
            if child.get("guess") is None:
                raise TreeStatsError(f"unsolved class {key!r} has no subtree at turn {turn}")
            walk(child, tuple(group), turn + 1)

    walk(tree, answer_words, 1)
    if stats.answers_reached != len(answer_words):
        raise TreeStatsError(
            f"tree covers {stats.answers_reached} of {len(answer_words)} answers"
        )
    return stats
