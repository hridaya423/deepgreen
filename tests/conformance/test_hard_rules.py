"""Differential conformance battery for feedback scoring and hard-mode legality.

Self-contained by contract: this module embeds a second, independent
implementation of Wordle feedback and NYT-standard-hard legality, imports
nothing from ``deep_green`` outside the public API under test, and drives
exactly ``3**15 == 14,348,907`` board/candidate comparisons against the
production predicate.
"""

import hashlib
from collections import Counter

CASE_COUNT = 3**15
assert CASE_COUNT == 14_348_907

ABSENT = "b"
PRESENT = "y"
CORRECT = "g"
DIGIT_MARKS = (ABSENT, PRESENT, CORRECT)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ACTION_COUNT = 5201


def reference_score(guess: str, answer: str) -> str:
    marks = [ABSENT] * 5
    remaining = Counter(answer)
    for index in range(5):
        letter = guess[index]
        if answer[index] == letter:
            marks[index] = CORRECT
            remaining[letter] -= 1
    for index in range(5):
        letter = guess[index]
        if marks[index] != CORRECT and remaining[letter] > 0:
            marks[index] = PRESENT
            remaining[letter] -= 1
    return "".join(marks)


def reference_is_legal(rows: tuple[tuple[str, str], ...], candidate: str) -> bool:
    required: dict[str, int] = {}
    for guess, marks in rows:
        for index in range(5):
            if marks[index] == CORRECT and candidate[index] != guess[index]:
                return False
        revealed: dict[str, int] = {}
        for letter, mark in zip(guess, marks):
            if mark in (PRESENT, CORRECT):
                revealed[letter] = revealed.get(letter, 0) + 1
        for letter, need in revealed.items():
            required[letter] = max(required.get(letter, 0), need)
    available = Counter(candidate)
    return all(available[letter] >= need for letter, need in required.items())


def _action_words() -> tuple[str, ...]:
    state = int.from_bytes(hashlib.sha256(b"openwordle-conformance-v1-actions").digest(), "big")
    words = []
    seen = set()
    while len(words) < ACTION_COUNT:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        word = "".join(ALPHABET[(state >> shift) % 26] for shift in (5, 11, 17, 23, 29))
        if word not in seen:
            seen.add(word)
            words.append(word)
    return tuple(sorted(words))


WORDS = _action_words()


def _marks_for(assignment: int) -> str:
    digits = []
    value = assignment
    for _position in range(5):
        digits.append(DIGIT_MARKS[value % 3])
        value //= 3
    return "".join(digits)


def build_case(case_index: int) -> tuple[tuple[tuple[str, str], ...], str]:
    rows = []
    value = case_index
    for offset in range(3):
        guess = WORDS[(case_index + offset) % ACTION_COUNT]
        marks = _marks_for(value % 3**5)
        value //= 3**5
        rows.append((guess, marks))
        if marks == CORRECT * 5:
            break
    candidate = WORDS[(case_index + 3) % ACTION_COUNT]
    return tuple(rows), candidate


def test_conformance_module_is_self_contained():
    import ast
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    class Scanner(ast.NodeVisitor):
        def __init__(self):
            self.function_stack = []

        def visit_FunctionDef(self, node):
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def check_import(self, node):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                if name == "deep_green" or name.startswith("deep_green."):
                    assert self.function_stack and self.function_stack[-1].startswith("test_"), (
                        f"production import outside a test entrypoint: {name!r} in {self.function_stack}"
                    )

        visit_Import = check_import
        visit_ImportFrom = check_import

    Scanner().visit(tree)

    reference_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("reference_")
    ]
    assert {node.name for node in reference_functions} == {"reference_score", "reference_is_legal"}
    for function_node in reference_functions:
        for node in ast.walk(function_node):
            assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
                f"reference helper {function_node.name} must stay dependency-free"
            )


def test_reference_scorer_agrees_with_published_fixtures():
    assert reference_score("allee", "apple") == "gybbg"
    assert reference_score("belle", "label") == "yyyyb"
    assert reference_score("treat", "crate") == "ygyyb"


def test_full_differential_conformance():
    from deep_green.rules import (
        Board,
        Feedback,
        GuessRow,
        Mode,
        Word5,
        is_legal_nyt_standard_hard,
    )

    word_cache: dict[str, Word5] = {}
    feedback_cache: dict[str, Feedback] = {}

    def cached_word(text: str) -> Word5:
        node = word_cache.get(text)
        if node is None:
            node = Word5.parse(text)
            word_cache[text] = node
        return node

    def cached_feedback(marks: str) -> Feedback:
        node = feedback_cache.get(marks)
        if node is None:
            node = Feedback.parse(marks)
            feedback_cache[marks] = node
        return node

    mismatch_count = 0
    mismatch_examples = []
    for case_index in range(CASE_COUNT):
        rows, candidate = build_case(case_index)
        board_rows = tuple(GuessRow(cached_word(guess), cached_feedback(marks)) for guess, marks in rows)
        board = Board.create(Mode.HARD, board_rows, len(board_rows) + 1)
        produced = is_legal_nyt_standard_hard(board, [candidate])[0]
        expected = reference_is_legal(rows, candidate)
        if produced != expected:
            mismatch_count += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((rows, candidate, produced, expected))
    assert mismatch_count == 0, (
        f"mismatches={mismatch_count} examples={mismatch_examples}"
    )
    print(f"cases={CASE_COUNT} mismatches={mismatch_count}")
