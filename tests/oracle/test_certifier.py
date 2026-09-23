"""Certifier production path: a -s run covers every action with exact bounds."""

from __future__ import annotations

import shutil

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
VENDOR_DIR = REPO_ROOT / "vendor" / "alex1770-wordle"

ACTIONS = ("audio", "crane", "pouty", "slate")
ANSWERS = ("crane", "pouty", "slate")


def test_select_canonical_winner_prefers_lowest_index_on_ties():
    from deep_green.oracle.certifier import (
        ActionOutcome,
        CertificationError,
        _select_canonical_winner,
    )

    outcomes = [
        ActionOutcome(action_index=2, word="slate", bound=3, exact=True),
        ActionOutcome(action_index=0, word="crane", bound=4, exact=True),
        ActionOutcome(action_index=1, word="pouty", bound=3, exact=True),
    ]
    winner = _select_canonical_winner(outcomes, objective=3)
    assert winner.word == "pouty" and winner.action_index == 1
    with pytest.raises(CertificationError, match="no exact action record"):
        _select_canonical_winner(outcomes, objective=2)
    inexact_only = [
        ActionOutcome(action_index=0, word="crane", bound=3, exact=False),
    ]
    with pytest.raises(CertificationError, match="no exact action record"):
        _select_canonical_winner(inexact_only, objective=3)


def _reported_winner_fixtures():
    from deep_green.oracle.certifier import ActionOutcome

    solver_order = ["crane", "pouty", "slate"]
    outcomes = [
        ActionOutcome(action_index=0, word="crane", bound=4, exact=True),
        ActionOutcome(action_index=1, word="pouty", bound=3, exact=True),
        ActionOutcome(action_index=2, word="slate", bound=3, exact=True),
    ]
    return solver_order, outcomes


def test_validate_reported_winner_accepts_an_exact_objective_record():
    from deep_green.oracle.certifier import _validate_reported_winner

    solver_order, outcomes = _reported_winner_fixtures()
    _validate_reported_winner(1, solver_order, outcomes, objective=3)


@pytest.mark.parametrize("bad", [None, True, -1, 3])
def test_validate_reported_winner_rejects_malformed_best_first(bad):
    from deep_green.oracle.certifier import (
        CertificationError,
        _validate_reported_winner,
    )

    solver_order, outcomes = _reported_winner_fixtures()
    with pytest.raises(CertificationError, match="malformed"):
        _validate_reported_winner(bad, solver_order, outcomes, objective=3)


def test_validate_reported_winner_rejects_a_non_objective_record():
    from deep_green.oracle.certifier import (
        CertificationError,
        _validate_reported_winner,
    )

    solver_order, outcomes = _reported_winner_fixtures()
    with pytest.raises(CertificationError, match="not an exact record"):
        _validate_reported_winner(0, solver_order, outcomes, objective=3)


def _solver_identity(**overrides):
    identity = {
        "hardmode": 0,
        "maxguesses": 6,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
        "exhaustive": True,
    }
    identity.update(overrides)
    return identity


def test_check_solver_identity_accepts_the_requested_cell():
    from deep_green.oracle.certifier import _check_solver_identity

    _check_solver_identity(
        _solver_identity(), mode="normal", horizon=6, action_count=4, answer_count=3
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("hardmode", 1),
        ("maxguesses", 5),
        ("all_count", 5),
        ("hidden_count", 2),
    ],
)
def test_check_solver_identity_rejects_cell_mismatch(field, bad):
    from deep_green.oracle.certifier import (
        CertificationError,
        _check_solver_identity,
    )

    with pytest.raises(CertificationError, match="does not match requested"):
        _check_solver_identity(
            _solver_identity(**{field: bad}),
            mode="normal",
            horizon=6,
            action_count=4,
            answer_count=3,
        )


def test_check_solver_identity_rejects_non_bool_exhaustive():
    from deep_green.oracle.certifier import (
        CertificationError,
        _check_solver_identity,
    )

    with pytest.raises(CertificationError, match="exhaustive"):
        _check_solver_identity(
            _solver_identity(exhaustive=1),
            mode="normal",
            horizon=6,
            action_count=4,
            answer_count=3,
        )


@pytest.mark.parametrize("bad", ["3", True, -1])
def test_solver_objective_must_be_a_nonnegative_int(bad):
    from deep_green.oracle.certifier import (
        CertificationError,
        _validated_objective,
    )

    with pytest.raises(CertificationError, match="malformed"):
        _validated_objective({"objective": bad})


def test_run_exact_search_rejects_unsupported_mode(tmp_path):
    from deep_green.oracle.certifier import CertificationError, run_exact_search

    with pytest.raises(CertificationError, match="mode"):
        run_exact_search(
            repo_root=REPO_ROOT,
            work_dir=tmp_path / "work",
            action_words=ACTIONS,
            answer_words=ANSWERS,
            mode="weird",
            horizon=6,
            with_tree=False,
        )


def test_archive_existing_certificate_returns_none_without_file(tmp_path):
    from deep_green.oracle_commands import _archive_existing_certificate

    cert_path = tmp_path / "certificate.json"
    assert _archive_existing_certificate(cert_path) is None
    assert list(tmp_path.iterdir()) == []


def test_archive_existing_certificate_preserves_exact_bytes(tmp_path):
    import hashlib

    from deep_green.oracle_commands import _archive_existing_certificate

    payload = b'{"legacy": true}\n'
    cert_path = tmp_path / "certificate.json"
    cert_path.write_bytes(payload)
    archive = _archive_existing_certificate(cert_path)
    digest = hashlib.sha256(payload).hexdigest()
    assert archive == tmp_path / f"certificate.legacy-{digest}.json"
    assert archive.read_bytes() == payload
    assert cert_path.read_bytes() == payload


def test_archive_existing_certificate_is_idempotent(tmp_path):
    from deep_green.oracle_commands import _archive_existing_certificate

    cert_path = tmp_path / "certificate.json"
    cert_path.write_bytes(b"same bytes\n")
    first = _archive_existing_certificate(cert_path)
    second = _archive_existing_certificate(cert_path)
    assert first == second
    assert len(list(tmp_path.iterdir())) == 2


def test_archive_existing_certificate_collision_fails_closed(tmp_path):
    import hashlib

    from deep_green.oracle_commands import _archive_existing_certificate

    payload = b"new bytes\n"
    cert_path = tmp_path / "certificate.json"
    cert_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / f"certificate.legacy-{digest}.json"
    archive.write_bytes(b"impostor bytes\n")
    with pytest.raises(RuntimeError, match="collision"):
        _archive_existing_certificate(cert_path)
    assert archive.read_bytes() == b"impostor bytes\n"
    assert cert_path.read_bytes() == payload


def test_count_policy_states_counts_only_branch_nodes():
    from deep_green.oracle.tree_stats import count_policy_states

    tree = {
        "guess": "stare",
        "children": {
            "22222": {"terminal": "solved", "turn": 1},
            "00000": {
                "guess": "lucid",
                "children": {"22222": {"terminal": "solved", "turn": 2}},
            },
            "11111": {"terminal": "loss"},
        },
    }
    assert count_policy_states(tree) == 2
    assert count_policy_states({"terminal": "solved"}) == 0


def test_run_exact_search_covers_every_action_with_exact_bounds(tmp_path):
    from deep_green.oracle.certifier import run_exact_search

    vendor_copy = tmp_path / "vendor" / "alex1770-wordle"
    shutil.copytree(VENDOR_DIR, vendor_copy)
    record = run_exact_search(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        action_words=ACTIONS,
        answer_words=ANSWERS,
        mode="normal",
        horizon=6,
        with_tree=True,
    )
    assert len(record.outcomes) == len(ACTIONS)
    assert {outcome.action_index for outcome in record.outcomes} == set(
        range(len(ACTIONS))
    )
    assert all(outcome.exact for outcome in record.outcomes)
    winner = next(
        outcome for outcome in record.outcomes if outcome.word == record.best_first_word
    )
    assert winner.exact and winner.bound == record.objective
    assert len(record.executable_sha256) == 64
    from deep_green.oracle.certifier import parse_tree_text

    assert record.tree_text
    assert parse_tree_text(record.tree_text)["guess"] == record.best_first_word


def _shard_stream(identity, values, objective, best_first, nodes, seconds):
    import json

    lines = [json.dumps(identity)]
    lines += [
        json.dumps(
            {
                "type": "action_value",
                "schema": "alex1770.wordle-events/v1",
                "action_index": index,
                "word": word,
                "bound": bound,
                "exact": exact,
            }
        )
        for index, word, bound, exact in values
    ]
    lines.append(
        json.dumps(
            {
                "type": "result",
                "schema": "alex1770.wordle-events/v1",
                "objective": objective,
                "average": objective / 2,
                "best_first": best_first,
                "nodes": nodes,
                "time_seconds": seconds,
                "depthonly": 0,
                "checkpoints": 0,
            }
        )
    )
    lines.append(
        json.dumps(
            {
                "type": "completion",
                "schema": "alex1770.wordle-events/v1",
                "status": "completed",
                "covered_actions": len(values),
                "declared_actions": 4,
            }
        )
    )
    return "\n".join(lines) + "\n"


def test_merge_shard_events_unions_values_and_synthesizes_result():
    from deep_green.oracle.adapter import parse_events
    from deep_green.oracle.certifier import _merge_shard_events

    identity = {
        "type": "run_identity",
        "schema": "alex1770.wordle-events/v1",
        "commitdesc": "test",
        "hardmode": 0,
        "exhaustive": True,
        "maxguesses": 6,
        "beta": None,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
    }
    shard0 = _shard_stream(
        identity,
        [(0, "audio", 99, False), (2, "pouty", 50, True)],
        50,
        2,
        nodes=10,
        seconds=1.5,
    )
    shard1 = _shard_stream(
        identity,
        [(1, "crane", 50, True), (3, "slate", 77, False)],
        50,
        1,
        nodes=20,
        seconds=2.5,
    )
    record = parse_events(_merge_shard_events([shard0, shard1], 4))
    assert len(record.action_values) == 4
    assert record.result["objective"] == 50
    assert record.result["best_first"] == 1
    assert record.result["nodes"] == 30
    assert record.result["time_seconds"] == 2.5
    assert record.completion["covered_actions"] == 4


def test_merge_shard_events_accepts_truncated_shard():
    import json

    from deep_green.oracle.adapter import parse_events
    from deep_green.oracle.certifier import _merge_shard_events

    identity = {
        "type": "run_identity",
        "schema": "alex1770.wordle-events/v1",
        "commitdesc": "test",
        "hardmode": 0,
        "exhaustive": True,
        "maxguesses": 6,
        "beta": 55,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
    }
    complete = _shard_stream(
        identity,
        [(0, "audio", 99, False), (2, "pouty", 50, True)],
        50,
        2,
        nodes=10,
        seconds=1.5,
    )
    truncated_lines = [json.dumps(identity)]
    truncated_lines += [
        json.dumps(
            {
                "type": "action_value",
                "schema": "alex1770.wordle-events/v1",
                "action_index": index,
                "word": word,
                "bound": bound,
                "exact": exact,
            }
        )
        for index, word, bound, exact in [
            (1, "crane", 50, True),
            (3, "slate", 77, False),
        ]
    ]
    truncated = "\n".join(truncated_lines) + "\n"
    record = parse_events(_merge_shard_events([complete, truncated], 4))
    assert len(record.action_values) == 4
    assert record.result["objective"] == 50
    assert record.result["best_first"] == 1
    assert record.result["nodes"] == 10
    assert record.completion["covered_actions"] == 4


def test_merge_shard_events_rejects_duplicate_actions():
    from deep_green.oracle.certifier import CertificationError, _merge_shard_events

    identity = {
        "type": "run_identity",
        "schema": "alex1770.wordle-events/v1",
        "commitdesc": "test",
        "hardmode": 0,
        "exhaustive": True,
        "maxguesses": 6,
        "beta": None,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
    }
    shard0 = _shard_stream(identity, [(0, "audio", 50, True)], 50, 0, 1, 1.0)
    shard1 = _shard_stream(identity, [(0, "audio", 50, True)], 50, 0, 1, 1.0)
    with pytest.raises(CertificationError, match="multiple shards"):
        _merge_shard_events([shard0, shard1], 4)


def test_merge_shard_events_rejects_divergent_identities():
    from deep_green.oracle.certifier import CertificationError, _merge_shard_events

    identity = {
        "type": "run_identity",
        "schema": "alex1770.wordle-events/v1",
        "commitdesc": "test",
        "hardmode": 0,
        "exhaustive": True,
        "maxguesses": 6,
        "beta": None,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
    }
    shard0 = _shard_stream(identity, [(0, "audio", 50, True)], 50, 0, 1, 1.0)
    other = dict(identity, hidden_count=2)
    shard1 = _shard_stream(other, [(1, "crane", 60, True)], 60, 1, 1, 1.0)
    with pytest.raises(CertificationError, match="identities diverge"):
        _merge_shard_events([shard0, shard1], 4)


def test_run_exact_search_sharded_covers_every_action(tmp_path):
    from deep_green.oracle.certifier import run_exact_search

    vendor_copy = tmp_path / "vendor" / "alex1770-wordle"
    shutil.copytree(VENDOR_DIR, vendor_copy)
    record = run_exact_search(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        action_words=ACTIONS,
        answer_words=ANSWERS,
        mode="normal",
        horizon=6,
        with_tree=True,
        jobs=2,
    )
    assert len(record.outcomes) == len(ACTIONS)
    assert {outcome.action_index for outcome in record.outcomes} == set(
        range(len(ACTIONS))
    )
    winner = next(
        outcome for outcome in record.outcomes if outcome.word == record.best_first_word
    )
    assert winner.exact and winner.bound == record.objective


def test_run_exact_search_resumes_from_complete_events(tmp_path):
    import json

    from deep_green.oracle.certifier import run_exact_search

    vendor_copy = tmp_path / "vendor" / "alex1770-wordle"
    shutil.copytree(VENDOR_DIR, vendor_copy)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    identity = {
        "type": "run_identity",
        "schema": "alex1770.wordle-events/v1",
        "commitdesc": "test",
        "hardmode": 0,
        "exhaustive": True,
        "maxguesses": 6,
        "beta": None,
        "all_count": 4,
        "hidden_count": 3,
        "depthonly": 0,
    }
    solver_order = ["crane", "pouty", "slate", "audio"]
    bounds = {"crane": 42, "pouty": 42, "slate": 99, "audio": 88}
    lines = [json.dumps(identity)]
    lines += [
        json.dumps(
            {
                "type": "action_value",
                "schema": "alex1770.wordle-events/v1",
                "action_index": index,
                "word": word,
                "bound": bounds[word],
                "exact": True,
            }
        )
        for index, word in enumerate(solver_order)
    ]
    lines.append(
        json.dumps(
            {
                "type": "result",
                "schema": "alex1770.wordle-events/v1",
                "objective": 42,
                "average": 21,
                "best_first": 0,
                "nodes": 7,
                "time_seconds": 0.5,
                "depthonly": 0,
                "checkpoints": 0,
            }
        )
    )
    lines.append(
        json.dumps(
            {
                "type": "completion",
                "schema": "alex1770.wordle-events/v1",
                "status": "completed",
                "covered_actions": 4,
                "declared_actions": 4,
            }
        )
    )
    (work_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = run_exact_search(
        repo_root=tmp_path,
        work_dir=work_dir,
        action_words=ACTIONS,
        answer_words=ANSWERS,
        mode="normal",
        horizon=6,
        with_tree=False,
        jobs=3,
    )
    # Fabricated bounds prove the solver sweep never ran.
    assert record.objective == 42
    assert record.best_first_word == "crane"
    assert {outcome.word: outcome.bound for outcome in record.outcomes} == bounds
    assert not list(work_dir.glob("events.shard*.jsonl"))
    assert not (work_dir / "joblist.txt").exists()
