"""Step 8 product contract tests: proof, Assist, Solve, envelopes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
FROZEN = REPO_ROOT / "benchmarks/openwordle-v1/frozen"
TIER = FROZEN / "corpus/tiers/tier-512.txt"
RELEASE = REPO_ROOT / "artifacts/releases/champion-512-certified"
PROOF = REPO_ROOT / "artifacts/proof/champion-512-certified/proof.json"
ACTIONS = RELEASE / "accepted-guesses.txt"

TIER_ANSWER = min(TIER.read_text(encoding="utf-8").split())
OFF_TIER_ANSWER = next(
    word
    for word in ACTIONS.read_text(encoding="utf-8").split()
    if word not in set(TIER.read_text(encoding="utf-8").split())
)


class FakePolicy:
    def __init__(self, guesses=()):
        self._guesses = list(guesses)
        self.calls = []

    def next_guess(self, mode, rows):
        self.calls.append((mode, tuple(tuple(row) for row in rows)))
        if self._guesses:
            return self._guesses.pop(0)
        return "stare" if mode == "normal" else "slate"


def make_client(*, policy=None, proof_path=PROOF, raise_server_exceptions=True):
    from deep_green.webapp import create_app

    app = create_app(
        release_dir=RELEASE,
        proof_path=proof_path,
        tier_answers_path=TIER,
        policy=policy or FakePolicy(),
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _assist(client, mode="normal"):
    response = client.post("/api/assist/sessions", json={"mode": mode})
    assert response.status_code == 200
    return response.json()


def _solve(client, word, mode="normal"):
    return client.post("/api/solve", json={"mode": mode, "word": word})


def test_proof_endpoint_verified():
    client = make_client()
    response = client.get("/api/proof")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["schema"] == "wordle.public-proof/v1"
    assert body["status"] == "VERIFIED"
    assert body["label"] == "Certified @512"
    assert body["model"]["parameter_count"] == 251_520


def test_proof_endpoint_tampered(tmp_path):
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    proof["release"]["weights_sha256"] = "0" * 64
    tampered = tmp_path / "proof.json"
    tampered.write_text(json.dumps(proof), encoding="utf-8")
    client = make_client(proof_path=tampered)
    response = client.get("/api/proof")
    assert response.status_code == 200
    assert response.json() == {
        "schema": "wordle.public-proof/v1",
        "status": "UNVERIFIED",
        "label": "Unverified build",
    }


def test_assist_normal_and_hard_roots():
    client = make_client()
    session = _assist(client, "normal")
    assert session["schema"] == "wordle.assist/v1"
    assert session["mode"] == "normal"
    assert session["turn"] == 1
    assert session["rows"] == []
    assert session["pending_guess"] == "stare"
    assert session["terminal"] is False
    assert session["outcome"] is None
    assert session["certification"] == "EXHIBITION"
    assert session["identity"]["proof_status"] == "VERIFIED"
    assert "answer" not in json.dumps(session)
    hard = _assist(client, "hard")
    assert hard["pending_guess"] == "slate"


def test_assist_feedback_progression_and_win():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 1,
            "pending_guess": "stare",
            "marks": "bbybb",
            "submission_id": "f1",
        },
    )
    body = response.json()
    assert body["turn"] == 2
    assert body["rows"] == [{"guess": "stare", "marks": "bbybb"}]
    assert body["terminal"] is False
    # the fake policy repeats 'stare'; the gate substitutes a consistent word
    assert body["pending_guess"] is not None
    assert body["pending_guess"] != "stare"
    assert body["consistent_count"] > 0
    assert body["consistent_counts"] == [body["consistent_count"]]
    assert body["vocabulary_size"] == len(TIER.read_text().split())
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 2,
            "pending_guess": body["pending_guess"],
            "marks": "ggggg",
            "submission_id": "f2",
        },
    )
    body = response.json()
    assert body["terminal"] is True
    assert body["outcome"] == "WON"
    assert body["pending_guess"] is None
    assert "answer" not in json.dumps(body)
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 3,
            "pending_guess": "stare",
            "marks": "bbbbb",
            "submission_id": "f3",
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "session-closed"


def test_assist_loss_at_turn_six():
    client = make_client(
        policy=FakePolicy(
            guesses=["dummy", "dizzy", "fizzy", "mimic", "juicy", "ninny"]
        )
    )
    session = _assist(client)
    sid = session["session_id"]
    body = session
    for turn in range(1, 7):
        response = client.post(
            f"/api/assist/sessions/{sid}/feedback",
            json={
                "expected_turn": turn,
                "pending_guess": body["pending_guess"],
                "marks": "bbbbb",
                "submission_id": f"turn-{turn}",
            },
        )
        body = response.json()
    assert body["terminal"] is True
    assert body["outcome"] == "LOST"


def test_assist_out_of_scope_board_stops_guessing():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    body = session
    for turn in range(1, 4):
        response = client.post(
            f"/api/assist/sessions/{sid}/feedback",
            json={
                "expected_turn": turn,
                "pending_guess": body["pending_guess"],
                "marks": "bbbbb",
                "submission_id": f"turn-{turn}",
            },
        )
        body = response.json()
    assert body["terminal"] is False
    assert body["out_of_scope"] is True
    assert body["consistent_count"] == 0
    assert body["consistent_preview"] == []
    assert body["pending_guess"] is None
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 4,
            "pending_guess": "slate",
            "marks": "bbbbb",
            "submission_id": "turn-4",
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "pending-mismatch"


def test_assist_candidate_fallback_wins():
    from deep_green.rules import Word5, score_guess

    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    body = session
    answer = Word5.parse("abbot")
    for turn in range(1, 7):
        pending = body["pending_guess"]
        marks = "".join(
            tile.value
            for tile in score_guess(Word5.parse(pending), answer).tiles
        )
        response = client.post(
            f"/api/assist/sessions/{sid}/feedback",
            json={
                "expected_turn": turn,
                "pending_guess": pending,
                "marks": marks,
                "submission_id": f"turn-{turn}",
            },
        )
        body = response.json()
        if body["terminal"]:
            break
    assert body["terminal"] is True
    assert body["outcome"] == "WON"
    assert body["rows"][-1]["guess"] == "abbot"


def test_assist_stale_turn_and_pending_mismatch():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 2,
            "pending_guess": "stare",
            "marks": "bbbbb",
            "submission_id": "s1",
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "stale-turn"
    response = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 1,
            "pending_guess": "slate",
            "marks": "bbbbb",
            "submission_id": "s2",
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "pending-mismatch"


def test_assist_submission_idempotency():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    request = {
        "expected_turn": 1,
        "pending_guess": "stare",
        "marks": "bbybb",
        "submission_id": "same",
    }
    first = client.post(f"/api/assist/sessions/{sid}/feedback", json=request)
    replay = client.post(f"/api/assist/sessions/{sid}/feedback", json=request)
    assert replay.content == first.content
    conflict = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={**request, "marks": "bbbbb"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "submission-conflict"


def test_assist_malformed_and_unknown():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    extra = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 1,
            "pending_guess": "stare",
            "marks": "bbbbb",
            "submission_id": "m1",
            "extra": 1,
        },
    )
    assert extra.status_code == 422
    assert extra.json()["code"] == "malformed"
    bad_marks = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={
            "expected_turn": 1,
            "pending_guess": "stare",
            "marks": "gggg",
            "submission_id": "m2",
        },
    )
    assert bad_marks.status_code == 422
    unknown = client.post(
        "/api/assist/sessions/nope/feedback",
        json={
            "expected_turn": 1,
            "pending_guess": "stare",
            "marks": "bbbbb",
            "submission_id": "m3",
        },
    )
    assert unknown.status_code == 404
    body = unknown.json()
    assert body["schema"] == "wordle.web-error/v1"
    assert body["code"] == "unknown-session"
    assert body["identity"]["proof_status"] == "VERIFIED"


def test_solve_certified_one_turn():
    client = make_client(policy=FakePolicy(guesses=[TIER_ANSWER]))
    response = _solve(client, TIER_ANSWER)
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "wordle.solve/v1"
    assert body["mode"] == "normal"
    assert body["word"] == TIER_ANSWER
    assert body["rows"] == [{"guess": TIER_ANSWER, "marks": "ggggg"}]
    assert body["turns"] == 1
    assert body["outcome"] == "SOLVED"
    assert body["certification"] == "CERTIFIED"
    assert body["identity"]["proof_status"] == "VERIFIED"
    assert "session_id" not in body


def test_solve_multi_turn_rollout():
    client = make_client(
        policy=FakePolicy(guesses=["stare", "mound", TIER_ANSWER])
    )
    body = _solve(client, TIER_ANSWER).json()
    assert [row["guess"] for row in body["rows"]] == [
        "stare",
        "mound",
        TIER_ANSWER,
    ]
    assert body["turns"] == 3
    assert body["rows"][-1]["marks"] == "ggggg"
    assert body["outcome"] == "SOLVED"
    assert body["consistent_counts"][-1] == 1
    assert body["vocabulary_size"] == len(TIER.read_text().split())


def test_solve_exhibition_off_tier():
    client = make_client(policy=FakePolicy(guesses=[OFF_TIER_ANSWER]))
    body = _solve(client, OFF_TIER_ANSWER).json()
    assert body["outcome"] == "SOLVED"
    assert body["certification"] == "EXHIBITION"


def test_solve_unsolved_at_six():
    guesses = ["stare", "mound", "climb", "fjord", "vexes", "wrung"]
    client = make_client(policy=FakePolicy(guesses=guesses))
    body = _solve(client, TIER_ANSWER).json()
    assert body["turns"] == 6
    assert len(body["rows"]) == 6
    assert body["outcome"] == "UNSOLVED"
    assert all(row["marks"] != "ggggg" for row in body["rows"])


def test_solve_unknown_word_rejected():
    client = make_client()
    response = _solve(client, "zzzzz")
    assert response.status_code == 422
    body = response.json()
    assert body["schema"] == "wordle.web-error/v1"
    assert body["code"] == "unknown-word"
    assert body["identity"]["proof_status"] == "VERIFIED"


def test_solve_malformed_requests():
    client = make_client()
    short = _solve(client, "abc")
    assert short.status_code == 422
    assert short.json()["code"] == "malformed"
    extra = client.post(
        "/api/solve", json={"mode": "normal", "word": "stare", "extra": 1}
    )
    assert extra.status_code == 422
    bad_mode = client.post(
        "/api/solve", json={"mode": "weird", "word": "stare"}
    )
    assert bad_mode.status_code == 422


def test_solve_hard_mode_passed_to_policy():
    policy = FakePolicy(guesses=[TIER_ANSWER])
    client = make_client(policy=policy)
    response = _solve(client, TIER_ANSWER, mode="hard")
    assert response.status_code == 200
    assert policy.calls and all(call[0] == "hard" for call in policy.calls)


def test_solve_is_stateless_and_deterministic():
    client = make_client()
    first = _solve(client, OFF_TIER_ANSWER).json()
    second = _solve(client, OFF_TIER_ANSWER).json()
    assert first == second


def test_solve_in_tier_but_unverified_proof(tmp_path):
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    proof["release"]["weights_sha256"] = "0" * 64
    tampered = tmp_path / "proof.json"
    tampered.write_text(json.dumps(proof), encoding="utf-8")
    client = make_client(
        policy=FakePolicy(guesses=[TIER_ANSWER]), proof_path=tampered
    )
    body = _solve(client, TIER_ANSWER).json()
    assert body["certification"] == "UNVERIFIED"


def test_policy_receives_only_public_state():
    policy = FakePolicy()
    client = make_client(policy=policy)
    response = _solve(client, TIER_ANSWER)
    assert response.status_code == 200
    serialized = json.dumps(policy.calls)
    assert TIER_ANSWER not in serialized
    for call in policy.calls:
        assert len(call) == 2
        assert call[0] in ("normal", "hard")
        for row in call[1]:
            assert len(row) == 2
            guess, marks = row
            assert len(guess) == 5 and guess.islower()
            assert len(marks) == 5 and set(marks) <= set("byg")


def test_error_envelope_carries_identity_not_internals():
    client = make_client()
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["schema"] == "wordle.web-error/v1"
    assert body["status"] == "ERROR"
    assert body["code"] == "not-found"
    assert "identity" in body
    assert "detail" not in body
    assert "Traceback" not in json.dumps(body)


def test_concurrent_sessions_isolated():
    client = make_client()
    sessions = [_assist(client) for _ in range(4)]
    ids = [s["session_id"] for s in sessions]
    assert len(set(ids)) == 4

    def play(sid, index):
        return client.post(
            f"/api/assist/sessions/{sid}/feedback",
            json={
                "expected_turn": 1,
                "pending_guess": "stare",
                "marks": "bbbbb" if index % 2 else "bbybb",
                "submission_id": f"par-{index}",
            },
        ).json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda args: play(*args), zip(ids, range(4))))
    for index, body in enumerate(results):
        expected = "bbybb" if index % 2 == 0 else "bbbbb"
        assert body["rows"] == [{"guess": "stare", "marks": expected}]
        assert body["turn"] == 2


def test_assist_terminal_replay_returns_stored_response():
    client = make_client()
    session = _assist(client)
    sid = session["session_id"]
    winning = {
        "expected_turn": 1,
        "pending_guess": "stare",
        "marks": "ggggg",
        "submission_id": "win",
    }
    first = client.post(f"/api/assist/sessions/{sid}/feedback", json=winning)
    assert first.json()["terminal"] is True
    replay = client.post(f"/api/assist/sessions/{sid}/feedback", json=winning)
    assert replay.status_code == 200
    assert replay.content == first.content
    new_id = client.post(
        f"/api/assist/sessions/{sid}/feedback",
        json={**winning, "submission_id": "after-close"},
    )
    assert new_id.status_code == 409
    assert new_id.json()["code"] == "session-closed"





def test_unhandled_exception_envelope():
    class BoomPolicy:
        def next_guess(self, mode, rows):
            raise RuntimeError("sensitive internals should not leak")

    client = make_client(
        policy=BoomPolicy(), raise_server_exceptions=False
    )
    response = _solve(client, "stare")
    assert response.status_code == 500
    body = response.json()
    assert body["schema"] == "wordle.web-error/v1"
    assert body["status"] == "ERROR"
    assert body["code"] == "internal-error"
    assert "sensitive internals" not in json.dumps(body)
    assert "Traceback" not in json.dumps(body)


def test_answer_loading_requires_verified_manifest(tmp_path):
    from deep_green.webapp import create_app

    answers = tmp_path / "answers"
    answers.mkdir()
    tier = answers / "tier-512.txt"
    tier.write_text(TIER.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="manifest"):
        create_app(
            release_dir=RELEASE,
            proof_path=PROOF,
            tier_answers_path=tier,
            policy=FakePolicy(),
        )
    (answers / "manifest.json").write_text("not json")
    with pytest.raises(ValueError, match="manifest"):
        create_app(
            release_dir=RELEASE,
            proof_path=PROOF,
            tier_answers_path=tier,
            policy=FakePolicy(),
        )
    (answers / "manifest.json").write_text(
        json.dumps({"files": {"tier-512.txt": "0" * 64}})
    )
    with pytest.raises(ValueError, match="drift"):
        create_app(
            release_dir=RELEASE,
            proof_path=PROOF,
            tier_answers_path=tier,
            policy=FakePolicy(),
        )
