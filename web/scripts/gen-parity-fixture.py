"""Regenerate web/testdata/parity.json from the certified release.

Mirrors policy_worker.py (score_logits/select_action over the legality mask)
and webapp.py (solve loop, assist session rules including the consistent<=2
fallback) directly against load_release — no sandbox needed because this is a
fixture generator, not the serving path.

    uv run python web/scripts/gen-parity-fixture.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from deep_green.numeric_policy import (  # noqa: E402
    encode_lexicon,
    score_logits,
    select_action,
)
from deep_green.proof import load_public_proof  # noqa: E402
from deep_green.release import load_release  # noqa: E402
from deep_green.rules import (  # noqa: E402
    MAX_ROWS,
    Board,
    Feedback,
    GuessRow,
    Mode,
    Word5,
    legal_actions,
    score_guess,
)

RELEASE_DIR = REPO / "artifacts/releases/champion-5022-certified"
PROOF_PATH = REPO / "artifacts/proof/champion-5022-certified/proof.json"
TIER_PATH = (
    REPO
    / "benchmarks/openwordle-v1/frozen-universe/corpus/tiers/tier-5022.txt"
)
OUT = REPO / "web/testdata/parity.json"

payload = load_release(RELEASE_DIR)
WORDS = list(payload.action_words)
TIER = TIER_PATH.read_text(encoding="utf-8").split()
TIER_SHA = hashlib.sha256(TIER_PATH.read_bytes()).hexdigest()
LETTER = {chr(ord("a") + i): i for i in range(26)}
MARK = {"b": 0, "y": 1, "g": 2}
MODE_IN = {"normal": [1.0, 0.0], "hard": [0.0, 1.0]}
WORD_LETTERS, WORD_POSITIONS = encode_lexicon(WORDS)

proof = load_public_proof(PROOF_PATH, RELEASE_DIR)
CERTIFIED_SCOPE = (
    proof.get("status") == "VERIFIED"
    and set(proof.get("certificates", {})) == {"normal", "hard"}
    and proof.get("claim", {}).get("tier") == len(TIER)
    and all(
        record.get("answers_sha256") == TIER_SHA
        for record in proof["certificates"].values()
    )
)


def next_guess(mode: str, rows: list[tuple[str, str]]) -> str:
    board = Board.create(
        Mode(mode),
        tuple(
            GuessRow(Word5.parse(g), Feedback.parse(m)) for g, m in rows
        ),
        len(rows) + 1,
    )
    mask = np.array(legal_actions(board, WORDS), dtype=bool)
    slot_count = max(len(rows), 1)
    row_letters = np.zeros((1, slot_count, 5), dtype=np.int64)
    row_marks = np.zeros((1, slot_count, 5), dtype=np.int64)
    row_mask = np.zeros((1, slot_count), dtype=bool)
    row_mask[0, : len(rows)] = True
    for index, (guess, marks) in enumerate(rows):
        row_letters[0, index] = [LETTER[c] for c in guess]
        row_marks[0, index] = [MARK[c] for c in marks]
    logits = score_logits(
        tensors=payload.tensors,
        row_letters=row_letters,
        row_marks=row_marks,
        row_positions=np.broadcast_to(np.arange(5), row_letters.shape).copy(),
        row_mask=row_mask,
        turn=np.array([float(len(rows) + 1)], dtype=np.float32),
        mode=np.array([MODE_IN[mode]], dtype=np.float32),
        word_letters=WORD_LETTERS,
        word_positions=WORD_POSITIONS,
    )
    return WORDS[select_action(logits[0], mask)]


def consistent(rows: list[tuple[str, str]]) -> list[str]:
    parsed = [(Word5.parse(g), Feedback.parse(m)) for g, m in rows]
    return [
        a
        for a in TIER
        if all(score_guess(g, Word5.parse(a)) == f for g, f in parsed)
    ]


def solve(mode: str, word: str) -> dict:
    rows: list[tuple[str, str]] = []
    counts: list[int] = []
    while len(rows) < MAX_ROWS:
        guess = next_guess(mode, rows)
        marks = "".join(t.value for t in score_guess(Word5.parse(guess), Word5.parse(word)).tiles)
        rows.append((guess, marks))
        counts.append(len(consistent(rows)))
        if marks == "ggggg":
            break
    in_tier = word in TIER
    return {
        "mode": mode,
        "word": word,
        "rows": [{"guess": g, "marks": m} for g, m in rows],
        "consistent_counts": counts,
        "outcome": "SOLVED" if rows[-1][1] == "ggggg" else "UNSOLVED",
        "certification": (
            "CERTIFIED" if in_tier and CERTIFIED_SCOPE else
            "UNVERIFIED" if in_tier else "EXHIBITION"
        ),
    }


def assist_trace(mode: str, word: str) -> dict:
    """Assisted game with real marks; exercises the fallback rule."""
    rows: list[tuple[str, str]] = []
    steps = [{"marks": None, "pending": next_guess(mode, rows), "consistent_count": len(TIER), "terminal": False, "outcome": None}]
    pending = steps[0]["pending"]
    counts: list[int] = []
    terminal = False
    outcome = None
    while not terminal:
        marks = "".join(
            t.value for t in score_guess(Word5.parse(pending), Word5.parse(word)).tiles
        )
        rows.append((pending, marks))
        pool = consistent(rows)
        counts.append(len(pool))
        if marks == "ggggg":
            terminal, outcome = True, "WON"
            pending = None
        elif len(rows) >= MAX_ROWS:
            terminal, outcome = True, "LOST"
            pending = None
        elif not pool:
            pending = None
        else:
            guess = next_guess(mode, rows)
            played = {g for g, _ in rows}
            if guess in played or (len(pool) <= 2 and guess not in pool):
                guess = pool[0]
            pending = guess
        steps.append(
            {
                "marks": marks,
                "pending": pending,
                "consistent_count": len(pool),
                "terminal": terminal,
                "outcome": outcome,
            }
        )
    return {
        "mode": mode,
        "word": word,
        "consistent_counts": counts,
        "steps": steps,
    }


def assist_script(mode: str, marks_seq: list[str]) -> dict:
    """Fixed marks feed; may end early when the board goes out of scope."""
    rows: list[tuple[str, str]] = []
    pending = next_guess(mode, rows)
    pool = list(TIER)
    counts: list[int] = []
    terminal = False
    outcome = None
    steps = [
        {
            "marks": None,
            "pending": pending,
            "consistent_count": len(pool),
            "terminal": False,
            "outcome": None,
        }
    ]
    for marks in marks_seq:
        if terminal or not pending:
            break
        rows.append((pending, marks))
        pool = consistent(rows)
        counts.append(len(pool))
        if marks == "ggggg":
            terminal, outcome = True, "WON"
            pending = None
        elif len(rows) >= MAX_ROWS:
            terminal, outcome = True, "LOST"
            pending = None
        elif not pool:
            pending = None
        else:
            guess = next_guess(mode, rows)
            played = {g for g, _ in rows}
            if guess in played or (len(pool) <= 2 and guess not in pool):
                guess = pool[0]
            pending = guess
        steps.append(
            {
                "marks": marks,
                "pending": pending,
                "consistent_count": len(pool),
                "terminal": terminal,
                "outcome": outcome,
            }
        )
    return {"mode": mode, "consistent_counts": counts, "steps": steps}


def sample_words() -> list[str]:
    picks = {WORDS[i] for i in range(0, len(WORDS), 211)}
    picks |= {"tares", "tramp", "fjord", "crane", "adieu", "queue"} & set(WORDS)
    return sorted(picks)


def main() -> int:
    words = sample_words()
    fixture = {
        "weights_sha256": payload.weights_sha256,
        "manifest_sha256": payload.manifest_sha256,
        "solves": [solve(mode, w) for mode in ("normal", "hard") for w in words],
        "assists": [
            assist_trace(mode, w) for mode in ("normal", "hard") for w in words[:8]
        ],
        "scripts": [
            assist_script("normal", ["bbbbb", "bbbbb", "bbbbb", "bbbbb", "bbbbb", "bbbbb"]),
            assist_script("hard", ["ggggg"]),
            assist_script("normal", ["ybybb", "gbgby", "bygbg"]),
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT}: {len(fixture['solves'])} solves, "
        f"{len(fixture['assists'])} assist traces, {len(fixture['scripts'])} scripts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
