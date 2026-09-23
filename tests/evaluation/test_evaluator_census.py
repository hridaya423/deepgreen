"""Evaluation tests: exact census metrics, certified parity, firewall audit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
RELEASE = REPO_ROOT / "tests/fixtures/release"
FROZEN = REPO_ROOT / "benchmarks/openwordle-v1/frozen"
CERTIFICATE = (
    REPO_ROOT
    / "artifacts/oracle/openwordle-v1"
    / "openwordle-v1.uniform.normal.full-compiled@512/certificate.json"
)
MINI_ANSWERS = ("audio", "crane", "irony", "pouty", "slate")
MINI_ACTIONS = (
    "audio",
    "crane",
    "irony",
    "metal",
    "pouty",
    "slate",
)

TURN_GUESSES = ("crane", "slate", "audio", "pouty", "irony", "metal")


class ScriptedRuntime:
    """Deterministic fake runtime: guesses a fixed word per turn."""

    def __init__(self):
        self.requests = []
        self.serialized = []
        self.peak_rss_bytes = 0

    def guess(self, request):
        self.requests.append(request)
        from deep_green.protocol import serialize_request

        self.serialized.append(
            serialize_request(request, action_words=MINI_ACTIONS)
        )
        return MINI_ACTIONS.index(TURN_GUESSES[request.turn - 1])


def _expected_census(answers, actions, mode="normal"):
    """Independent recomputation of every census metric."""
    from deep_green.rules import (
        ActionSpace,
        Board,
        GuessRow,
        Mode,
        Word5,
        legal_actions,
        score_guess,
    )

    mode_value = Mode.NORMAL if mode == "normal" else Mode.HARD
    space = ActionSpace.parse(actions)
    scores = []
    per_answer = {}
    for answer_text in sorted(set(answers)):
        answer = Word5.parse(answer_text)
        rows: list[GuessRow] = []
        lines = []
        outcome = 7
        for turn in range(1, 7):
            board = Board.create(mode_value, tuple(rows), turn)
            mask = legal_actions(board, space)
            del mask
            guess = Word5.parse(TURN_GUESSES[turn - 1])
            feedback = score_guess(guess, answer)
            lines.append(f"{turn}:{guess.value}:{_marks(feedback)}\n")
            rows.append(GuessRow(guess, feedback))
            if all(tile.value == "g" for tile in feedback.tiles):
                outcome = turn
                break
        scores.append(outcome)
        per_answer[answer_text] = hashlib.sha256(
            "".join(lines).encode("utf-8")
        ).hexdigest()
    trace = hashlib.sha256(
        "".join(
            f"{a}:{per_answer[a]}\n" for a in sorted(per_answer)
        ).encode("utf-8")
    ).hexdigest()
    ordered = sorted(scores)
    n = len(ordered)
    losses = sum(1 for s in scores if s == 7)
    solved = n - losses
    return {
        "answers": n,
        "losses": losses,
        "score7_total": sum(scores),
        "score7_mean": sum(scores) / n,
        "solved": solved,
        "solved_only_mean": (sum(s for s in scores if s < 7) / solved)
        if solved
        else None,
        "histogram": tuple(
            sum(1 for s in scores if s == bucket) for bucket in range(1, 8)
        ),
        "p95": ordered[max(0, math.ceil(0.95 * n) - 1)],
        "worst_score": max(scores),
        "trace_digest": trace,
        "per_answer_trace_digests": per_answer,
        "guesses": sum(min(s, 6) for s in scores),
    }


def _marks(feedback):
    text = {"b": "b", "y": "y", "g": "g"}
    return "".join(text[tile.value] for tile in feedback.tiles)


def test_release_fixture_verifies_clean():
    sys.path.insert(0, str(REPO_ROOT / "tools/release"))
    from build_fixture import verify_release

    assert verify_release(RELEASE) == []


def test_census_runs_end_to_end_and_is_deterministic():
    from deep_green.evaluate import run_census
    from deep_green.inference import PolicyRuntime

    words = tuple((RELEASE / "accepted-guesses.txt").read_text().split())
    with PolicyRuntime(release_dir=RELEASE, action_words=words) as runtime:
        first = run_census(
            runtime=runtime, action_words=words, answer_words=MINI_ANSWERS,
            mode="normal",
        )
        second = run_census(
            runtime=runtime, action_words=words, answer_words=MINI_ANSWERS,
            mode="normal",
        )
    volatile = {"latency_ms", "peak_rss_bytes"}
    first_fixed = {k: v for k, v in first.as_dict().items() if k not in volatile}
    second_fixed = {k: v for k, v in second.as_dict().items() if k not in volatile}
    assert first_fixed == second_fixed
    assert first.answers == len(MINI_ANSWERS)
    assert 0 <= first.losses <= len(MINI_ANSWERS)
    assert first.latency_ms["count"] == second.latency_ms["count"]
    assert first.peak_rss_bytes > 0


def test_census_metrics_are_exact_against_hand_computed_traces():
    from deep_green.evaluate import run_census

    runtime = ScriptedRuntime()
    result = run_census(
        runtime=runtime,
        action_words=MINI_ACTIONS,
        answer_words=MINI_ANSWERS,
        mode="normal",
    )
    expected = _expected_census(MINI_ANSWERS, MINI_ACTIONS)
    assert result.answers == expected["answers"]
    assert result.losses == expected["losses"]
    assert result.score7_total == expected["score7_total"]
    assert result.score7_mean == pytest.approx(expected["score7_mean"])
    assert result.solved == expected["solved"]
    assert result.solved_only_mean == pytest.approx(expected["solved_only_mean"])
    assert result.histogram == expected["histogram"]
    assert result.p95 == expected["p95"]
    assert result.worst_score == expected["worst_score"]
    assert result.illegal_actions == 0
    assert result.trace_digest == expected["trace_digest"]
    assert result.per_answer_trace_digests == expected["per_answer_trace_digests"]
    assert set(result.latency_ms) == {"count", "mean", "p50", "p95", "max"}
    assert result.latency_ms["count"] == expected["guesses"]


def test_no_hidden_answer_appears_in_any_worker_request():
    from deep_green.evaluate import run_census

    runtime = ScriptedRuntime()
    run_census(
        runtime=runtime,
        action_words=MINI_ACTIONS,
        answer_words=MINI_ANSWERS,
        mode="normal",
    )
    assert runtime.requests, "census should issue requests"
    allowed = {
        "schema",
        "mode",
        "turn",
        "played_rows",
        "action_space_sha256",
        "action_count",
        "legality_mask",
        "request_hash",
    }
    for line in runtime.serialized:
        body = json.loads(line)
        assert set(body) == allowed
    for answer in MINI_ANSWERS:
        solo = ScriptedRuntime()
        run_census(
            runtime=solo,
            action_words=MINI_ACTIONS,
            answer_words=(answer,),
            mode="normal",
        )
        for line in solo.serialized:
            assert f'"{answer}"' not in line, (
                f"answer {answer!r} leaked into a worker request"
            )


def test_census_hard_mode_uses_hard_legality():
    from deep_green.evaluate import run_census

    runtime = ScriptedRuntime()
    result = run_census(
        runtime=runtime,
        action_words=MINI_ACTIONS,
        answer_words=("crane",),
        mode="hard",
    )
    assert result.answers == 1
    for request in runtime.requests:
        assert request.mode == "hard"


def test_illegal_action_is_counted_then_fails_evaluation():
    from deep_green.evaluate import EvaluationError, run_census
    from deep_green.inference import PolicyIllegalActionError

    class RogueRuntime:
        peak_rss_bytes = 0

        def guess(self, request):
            raise PolicyIllegalActionError("worker chose an illegal action")

    with pytest.raises(EvaluationError, match="illegal"):
        run_census(
            runtime=RogueRuntime(),
            action_words=MINI_ACTIONS,
            answer_words=("crane",),
            mode="normal",
        )


def test_runtime_fault_translates_to_evaluation_error():
    from deep_green.evaluate import EvaluationError, run_census
    from deep_green.inference import PolicyRuntimeError

    class DeadRuntime:
        peak_rss_bytes = 0

        def guess(self, request):
            raise PolicyRuntimeError("worker died")

    with pytest.raises(EvaluationError):
        run_census(
            runtime=DeadRuntime(),
            action_words=MINI_ACTIONS,
            answer_words=("crane",),
            mode="normal",
        )


def _matching_result(certificate):
    from deep_green.evaluate import CensusResult

    objective = certificate["objective_value"]
    return CensusResult(
        answers=512,
        losses=objective["losses"],
        score7_total=objective["score7_total"],
        score7_mean=objective["score7_total"] / 512,
        solved=512 - objective["losses"],
        solved_only_mean=1.0,
        histogram=(512, 0, 0, 0, 0, 0, objective["losses"]),
        p95=1,
        worst_score=objective["worst_score"],
        illegal_actions=0,
        trace_digest="0" * 64,
        per_answer_trace_digests={},
        latency_ms={"count": 1, "mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0},
        peak_rss_bytes=0,
    )


def test_require_certificate_parity_accepts_exact_objective():
    from deep_green.evaluate import require_certificate_parity

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    require_certificate_parity(_matching_result(certificate), certificate)


def test_perturbed_policy_result_fails_parity():
    from dataclasses import replace

    from deep_green.evaluate import (
        EvaluationError,
        require_certificate_parity,
    )

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    result = _matching_result(certificate)
    perturbed = replace(result, score7_total=result.score7_total + 1)
    with pytest.raises(EvaluationError, match="score7_total"):
        require_certificate_parity(perturbed, certificate)
    perturbed = replace(result, losses=result.losses + 1)
    with pytest.raises(EvaluationError, match="losses"):
        require_certificate_parity(perturbed, certificate)
    perturbed = replace(result, worst_score=result.worst_score + 1)
    with pytest.raises(EvaluationError, match="worst_score"):
        require_certificate_parity(perturbed, certificate)
    perturbed = replace(result, illegal_actions=1)
    with pytest.raises(EvaluationError, match="illegal"):
        require_certificate_parity(perturbed, certificate)


def test_certified_input_mismatch_fails_before_census(tmp_path, monkeypatch):
    from deep_green import evaluate

    def boom(**kwargs):
        raise AssertionError("census must not run on mismatched inputs")

    monkeypatch.setattr(evaluate, "run_census", boom)
    monkeypatch.setattr(
        evaluate,
        "PolicyRuntime",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("runtime")),
    )
    from deep_green.evaluate import EvaluationError, evaluate_certified

    with pytest.raises(EvaluationError):
        evaluate_certified(
            release_dir=tmp_path / "no-release",
            certificate_path=CERTIFICATE,
            frozen_root=FROZEN,
        )
    with pytest.raises(EvaluationError):
        evaluate_certified(
            release_dir=RELEASE,
            certificate_path=CERTIFICATE,
            frozen_root=tmp_path,
        )
    bad_cert = tmp_path / "cert.json"
    bad_cert.write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(EvaluationError):
        evaluate_certified(
            release_dir=RELEASE,
            certificate_path=bad_cert,
            frozen_root=FROZEN,
        )


def test_evaluate_certified_returns_exact_identity(monkeypatch):
    from deep_green import evaluate
    from deep_green.evaluate import evaluate_certified

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    result = _matching_result(certificate)
    monkeypatch.setattr(evaluate, "run_census", lambda **kwargs: result)

    class FakeRuntime:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(evaluate, "PolicyRuntime", FakeRuntime)
    census_result, identity = evaluate_certified(
        release_dir=RELEASE,
        certificate_path=CERTIFICATE,
        frozen_root=FROZEN,
    )
    assert census_result is result
    assert set(identity) == {
        "benchmark_id",
        "view_id",
        "tier",
        "mode",
        "answers_sha256",
        "action_space_sha256",
        "rules_sha256",
        "prior_id",
        "track",
        "inference_class",
        "artifact_sha256",
        "manifest_sha256",
        "certificate_sha256",
    }
    assert identity["benchmark_id"] == "openwordle-v1"
    assert identity["view_id"] == "openwordle-v1.uniform.normal.full-compiled"
    assert identity["tier"] == 512
    assert identity["mode"] == "normal"
    assert identity["inference_class"] == "numpy-f32-pure-neural"
    assert identity["certificate_sha256"] == hashlib.sha256(
        CERTIFICATE.read_bytes()
    ).hexdigest()


def test_census_report_is_canonical_v2_and_atomic(tmp_path):
    from deep_green.evaluate import save_census_report

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    result = _matching_result(certificate)
    out = tmp_path / "nested" / "census.json"
    save_census_report(out, result, {"benchmark_id": "openwordle-v1"})
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "wordle.census/v2"
    assert set(payload) == {"schema", "identity", "result"}
    raw = out.read_text(encoding="utf-8")
    assert raw == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_worker_cli_requires_exactly_release_dir():
    result = subprocess.run(
        [sys.executable, "-m", "deep_green.policy_worker"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_worker_rejects_missing_release(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_green.policy_worker",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_worker_fails_closed_on_protocol_violation():
    line = (
        '{"schema":"wordle.policy-ipc/v1","mode":"normal","turn":1,'
        '"played_rows":[],"action_space_sha256":"0","action_count":5022,'
        '"legality_mask":[1],"request_hash":"0","answer_id":"slate"}\n'
    )
    result = subprocess.run(
        [sys.executable, "-m", "deep_green.policy_worker", str(RELEASE)],
        input=line,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert '"status":"FAIL"' in result.stdout.replace(" ", "")


def test_runtime_refuses_vocabulary_mismatch():
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with pytest.raises(PolicyRuntimeError, match="vocabulary mismatch"):
        PolicyRuntime(release_dir=RELEASE, action_words=("aaaaa", "bbbbb"))


def test_runtime_refuses_missing_release(tmp_path):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with pytest.raises(PolicyRuntimeError):
        PolicyRuntime(release_dir=tmp_path / "nothing", action_words=("crane",))


def test_firewall_audit_passes_on_the_release_fixture(tmp_path):
    from deep_green.firewall_audit import audit_release

    audit_release(RELEASE, output_path=tmp_path / "firewall.json")
    saved = json.loads((tmp_path / "firewall.json").read_text())
    assert saved["schema"] == "wordle.firewall/v1"
    assert saved["status"] == "PASS", saved["checks"]
    checks = {c["check"]: c["status"] for c in saved["checks"]}
    for required in (
        "strict-release-load",
        "actual-seatbelt-launch",
        "request-determinism",
        "reset-determinism",
        "full-vocabulary-count",
        "forbidden-field-parser-battery",
        "network-denied",
        "file-write-denied",
        "fail-closed-empty-mask",
    ):
        assert checks.get(required) == "PASS", f"{required}: {checks}"
    assert saved["weights_sha256"] == hashlib.sha256(
        (RELEASE / "weights.safetensors").read_bytes()
    ).hexdigest()
    assert len(saved["manifest_sha256"]) == 64
    assert len(saved["sandbox_profile_sha256"]) == 64


def test_evidence_binds_the_weights_hash():
    from deep_green.firewall_audit import audit_release

    evidence = audit_release(RELEASE)
    actual = hashlib.sha256((RELEASE / "weights.safetensors").read_bytes()).hexdigest()
    assert evidence["weights_sha256"] == actual


def test_fake_runtime_bool_index_rejected():
    from deep_green.evaluate import EvaluationError, run_census

    class _BoolRuntime:
        peak_rss_bytes = 0

        def guess(self, request):
            return True

    with pytest.raises(EvaluationError, match="non-integer"):
        run_census(
            runtime=_BoolRuntime(),
            action_words=MINI_ACTIONS,
            answer_words=MINI_ANSWERS,
            mode="normal",
        )


def test_fake_runtime_illegal_masked_index_rejected():
    from deep_green.evaluate import EvaluationError, run_census

    class _IllegalRuntime:
        peak_rss_bytes = 0
        calls = 0

        def guess(self, request):
            self.calls += 1
            if self.calls > 1:
                for index, flag in enumerate(request.legality_mask):
                    if not flag:
                        return index
            return request.legality_mask.index(True)

    with pytest.raises(EvaluationError, match="illegal"):
        run_census(
            runtime=_IllegalRuntime(),
            action_words=MINI_ACTIONS,
            answer_words=("crane",),
            mode="hard",
        )


def test_certificate_parity_rejects_malformed_objective():
    from deep_green.evaluate import EvaluationError, require_certificate_parity

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    result = _matching_result(certificate)
    for bad in (
        {},
        {"losses": 0, "score7_total": 1},
        {"losses": 0, "score7_total": 1, "worst_score": 2, "extra": 3},
        {"losses": True, "score7_total": 1, "worst_score": 2},
        {"losses": -1, "score7_total": 1, "worst_score": 2},
        {"losses": "0", "score7_total": 1, "worst_score": 2},
    ):
        broken = dict(certificate)
        broken["objective_value"] = bad
        with pytest.raises(EvaluationError, match="objective"):
            require_certificate_parity(result, broken)


def test_firewall_audit_returns_fail_evidence_when_runtime_fails(
    tmp_path, monkeypatch
):
    import deep_green.firewall_audit as audit_mod
    import deep_green.inference as inference_mod
    from deep_green.inference import PolicyRuntimeError

    def _explode(*args, **kwargs):
        raise PolicyRuntimeError("sandboxed worker exited at launch (rc=3)")

    monkeypatch.setattr(inference_mod, "PolicyRuntime", _explode)
    out = tmp_path / "firewall.json"
    evidence = audit_mod.audit_release(RELEASE, output_path=out)
    assert evidence["status"] == "FAIL"
    checks = {c["check"]: c["status"] for c in evidence["checks"]}
    assert checks["strict-release-load"] == "PASS"
    assert checks["actual-seatbelt-launch"] == "FAIL"
    assert checks["network-denied"] == "PASS"
    assert checks["file-write-denied"] == "PASS"
    saved = json.loads(out.read_text())
    assert saved["status"] == "FAIL"
    assert len(saved["weights_sha256"]) == 64
    assert len(saved["manifest_sha256"]) == 64
