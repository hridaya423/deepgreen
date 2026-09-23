"""Adapter tests: strict event parsing and hash-bound requests."""

from __future__ import annotations

import json

import pytest

from deep_green.oracle.adapter import (
    OracleActionValue,
    OracleEventsError,
    OracleRequest,
    OracleRequestError,
    parse_events,
)

VALID_IDENTITY = {
    "type": "run_identity",
    "schema": "alex1770.wordle-events/v1",
    "commitdesc": "8fd3f1b-upstream-pin-local-build",
    "hardmode": 0,
    "exhaustive": True,
    "maxguesses": 6,
    "beta": None,
    "all_count": 2,
    "hidden_count": 2,
    "depthonly": 0,
}
VALID_ACTION_VALUES = (
    {
        "type": "action_value",
        "schema": "alex1770.wordle-events/v1",
        "action_index": 0,
        "word": "crane",
        "bound": 3,
        "exact": True,
    },
    {
        "type": "action_value",
        "schema": "alex1770.wordle-events/v1",
        "action_index": 1,
        "word": "slate",
        "bound": 5,
        "exact": False,
    },
)
VALID_RESULT = {
    "type": "result",
    "schema": "alex1770.wordle-events/v1",
    "objective": 3,
    "average": 1.5,
    "best_first": 0,
    "nodes": 1,
    "time_seconds": 0.01,
    "depthonly": False,
    "checkpoints": 0,
}
VALID_COMPLETION = {
    "type": "completion",
    "schema": "alex1770.wordle-events/v1",
    "status": "completed",
    "covered_actions": 2,
    "declared_actions": 2,
}


def _stream(*events) -> str:
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _valid_stream() -> str:
    return _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        VALID_COMPLETION,
    )


def _request(**overrides):
    fields = {
        "mode": "normal",
        "horizon": 6,
        "rules_sha256": "a" * 64,
        "prior_sha256": "b" * 64,
        "action_space_sha256": "c" * 64,
        "answer_set_sha256": "d" * 64,
        "executable_sha256": "e" * 64,
        "compiler_id": "apple-clang",
        "solver_source_sha256": "f" * 64,
        "action_words": ("crane", "slate"),
        "answer_words": ("slate",),
    }
    fields.update(overrides)
    return OracleRequest(**fields)


def test_parse_events_accepts_a_wellformed_stream():
    record = parse_events(_valid_stream())
    assert record.identity["hidden_count"] == 2
    assert record.result["objective"] == 3
    assert record.completion["status"] == "completed"
    assert record.action_values == (
        OracleActionValue(action_index=0, word="crane", bound=3, exact=True),
        OracleActionValue(action_index=1, word="slate", bound=5, exact=False),
    )


def test_unknown_event_type_fails_distinctly():
    stream = _stream(
        VALID_IDENTITY,
        {"type": "mystery", "schema": "alex1770.wordle-events/v1"},
        VALID_RESULT,
    )
    with pytest.raises(OracleEventsError, match="unknown event type"):
        parse_events(stream)


def test_missing_field_fails_distinctly():
    broken = {k: v for k, v in VALID_RESULT.items() if k != "nodes"}
    stream = _stream(VALID_IDENTITY, broken)
    with pytest.raises(OracleEventsError, match="missing fields"):
        parse_events(stream)


def test_wrong_schema_fails():
    impostor = dict(VALID_IDENTITY, schema="other.v9")
    with pytest.raises(OracleEventsError, match="unsupported schema"):
        parse_events(json.dumps(impostor))


def test_truncated_stream_fails():
    stream = _stream(VALID_IDENTITY, VALID_RESULT)
    with pytest.raises(OracleEventsError, match="truncated"):
        parse_events(stream)


def test_checkpoint_count_mismatch_fails():
    stream = _stream(
        VALID_IDENTITY,
        {"type": "checkpoint", "schema": "alex1770.wordle-events/v1", "generation": 1},
        dict(VALID_RESULT, checkpoints=2),
        dict(VALID_COMPLETION, covered_actions=0),
    )
    with pytest.raises(OracleEventsError, match="checkpoint"):
        parse_events(stream)


def test_duplicate_action_index_fails():
    stream = _stream(
        VALID_IDENTITY,
        VALID_ACTION_VALUES[0],
        dict(VALID_ACTION_VALUES[0], word="slate"),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="duplicate action_index"):
        parse_events(stream)


def test_duplicate_action_word_fails():
    stream = _stream(
        VALID_IDENTITY,
        VALID_ACTION_VALUES[0],
        dict(VALID_ACTION_VALUES[1], word="crane"),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="duplicate action word"):
        parse_events(stream)


def test_action_value_index_out_of_range_fails():
    stream = _stream(
        VALID_IDENTITY,
        dict(VALID_ACTION_VALUES[0], action_index=7),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="action_index"):
        parse_events(stream)


def test_action_value_rejects_non_boolean_exact():
    stream = _stream(
        VALID_IDENTITY,
        dict(VALID_ACTION_VALUES[0], exact=1),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="boolean"):
        parse_events(stream)


def test_action_value_rejects_negative_bound():
    stream = _stream(
        VALID_IDENTITY,
        dict(VALID_ACTION_VALUES[0], bound=-1),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="bound"):
        parse_events(stream)


def test_completion_covered_count_mismatch_fails():
    stream = _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        dict(VALID_COMPLETION, covered_actions=1),
    )
    with pytest.raises(OracleEventsError, match="covered_actions"):
        parse_events(stream)


def test_completion_declared_count_mismatch_fails():
    stream = _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        dict(VALID_COMPLETION, declared_actions=9),
    )
    with pytest.raises(OracleEventsError, match="declared_actions"):
        parse_events(stream)


def test_action_value_after_result_fails():
    stream = _stream(
        VALID_IDENTITY,
        VALID_RESULT,
        VALID_ACTION_VALUES[0],
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="after the result"):
        parse_events(stream)


def test_checkpoint_after_result_fails():
    stream = _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        dict(VALID_RESULT, checkpoints=1),
        {"type": "checkpoint", "schema": "alex1770.wordle-events/v1", "generation": 0},
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="after the result"):
        parse_events(stream)


@pytest.mark.parametrize("bad", [-1, True, "3"])
def test_checkpoint_generation_must_be_a_nonnegative_int(bad):
    stream = _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        {"type": "checkpoint", "schema": "alex1770.wordle-events/v1", "generation": bad},
        dict(VALID_RESULT, checkpoints=1),
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="invalid checkpoint generation"):
        parse_events(stream)


@pytest.mark.parametrize("bad", ["2", True, -1])
def test_run_identity_all_count_must_be_a_nonnegative_int(bad):
    stream = _stream(
        dict(VALID_IDENTITY, all_count=bad),
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="all_count"):
        parse_events(stream)


@pytest.mark.parametrize("field", ("covered_actions", "declared_actions"))
@pytest.mark.parametrize("bad", ["2", True, -1])
def test_completion_counts_must_be_nonnegative_ints(field, bad):
    stream = _stream(
        VALID_IDENTITY,
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        dict(VALID_COMPLETION, **{field: bad}),
    )
    with pytest.raises(OracleEventsError, match="nonnegative integer"):
        parse_events(stream)


def test_action_value_missing_field_fails():
    broken = {k: v for k, v in VALID_ACTION_VALUES[0].items() if k != "bound"}
    stream = _stream(VALID_IDENTITY, broken, VALID_RESULT, VALID_COMPLETION)
    with pytest.raises(OracleEventsError, match="missing fields"):
        parse_events(stream)


def test_action_value_unknown_field_fails():
    stream = _stream(
        VALID_IDENTITY,
        dict(VALID_ACTION_VALUES[0], flavor="sweet"),
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="unknown fields"):
        parse_events(stream)


def test_run_identity_unknown_field_fails():
    stream = _stream(
        dict(VALID_IDENTITY, flavor="sweet"),
        *VALID_ACTION_VALUES,
        VALID_RESULT,
        VALID_COMPLETION,
    )
    with pytest.raises(OracleEventsError, match="unknown fields"):
        parse_events(stream)


def test_request_rejects_unset_fields():
    with pytest.raises(OracleRequestError, match="no defaults"):
        _request(rules_sha256="")


def test_request_rejects_bad_mode_and_horizon():
    with pytest.raises(OracleRequestError, match="mode"):
        _request(mode="ultrahard")
    with pytest.raises(OracleRequestError, match="horizon"):
        _request(horizon=9)


def test_request_rejects_malformed_hashes():
    with pytest.raises(OracleRequestError, match="sha256"):
        _request(prior_sha256="zz")


def test_request_requires_explicit_word_sets():
    with pytest.raises(OracleRequestError, match="action_words"):
        _request(action_words=())


def test_identity_digest_is_stable_and_sensitive():
    base = _request()
    assert base.identity_digest() == _request().identity_digest()
    changed = _request(answer_words=("crane",))
    assert base.identity_digest() != changed.identity_digest()


def test_cwd_decoys_do_not_change_the_request(tmp_path, monkeypatch):
    decoy = tmp_path / "decoy-rules.json"
    decoy.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    from_cwd = _request()
    assert from_cwd.rules_sha256 == "a" * 64
