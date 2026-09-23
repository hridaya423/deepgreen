"""Oracle-example schema tests: every leak class fails validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmark import repo_path
from deep_green.examples import (
    ExampleError,
    canonical_example_bytes,
    example_id_for,
    generate_root_examples,
    load_examples,
    project_batch,
    validate_example,
)
from deep_green.rules import ActionSpace

REPO_ROOT = repo_path()
FIXTURE = repo_path("tests/fixtures/oracle_examples/tiny.jsonl")
_WORDS = tuple(
    repo_path("benchmarks/openwordle-v1/frozen/corpus/words/accepted-guesses.txt")
    .read_text()
    .split()
)


@pytest.fixture(scope="module")
def examples():
    return load_examples(FIXTURE)


def test_fixture_loads_and_validates(examples):
    assert len(examples) == 2
    assert {record["public_state"]["mode"] for record in examples} == {"normal", "hard"}
    expected = {
        "normal": (
            "ab763d900306a83e9407885d210186db6239778054b1a5290f7f0d8b84abccfe",
            ["stare"],
        ),
        "hard": (
            "2fbb51fd7224f6070fc6c11ad19d9b92364af9cfce489551b8ad81bc6792c269",
            ["slate", "stare"],
        ),
    }
    for record in examples:
        mode = record["public_state"]["mode"]
        assert validate_example(record) == record
        assert record["identity"]["certificate_sha256"] == expected[mode][0]
        assert record["optimal_tie_set"] == expected[mode][1]
        assert record["example_id"] == example_id_for(record)
        assert len(record["sparse_action_values"]) == 5022


def _mutated(base, **changes):
    record = json.loads(json.dumps(base))
    record.update(changes)
    return record


def test_rejects_answer_identifier_field(examples):
    with pytest.raises(ExampleError, match="forbidden field"):
        validate_example(_mutated(examples[0], answer_id="slate"))


def test_rejects_posterior_features(examples):
    with pytest.raises(ExampleError, match="forbidden field"):
        validate_example(_mutated(examples[0], posterior=[0.25, 0.75]))


def test_rejects_untyped_infinity(examples):
    turn = examples[0]["public_state"]["turn"]
    raw = json.dumps(examples[0]).replace(f'"turn": {turn}', '"turn": Infinity')
    assert raw != json.dumps(examples[0]), "replacement failed to apply"
    with pytest.raises(ExampleError, match="non-finite|Invalid JSON|invalid JSON"):
        load_examples_line(raw)


def load_examples_line(line: str) -> list[dict]:
    target = Path("/tmp/wordle-bad-example.jsonl")
    target.write_text(line + "\n")
    return load_examples(target)


def test_rejects_nan(examples):
    turn = examples[0]["public_state"]["turn"]
    raw = json.dumps(examples[0]).replace(f'"turn": {turn}', '"turn": NaN')
    with pytest.raises(ExampleError, match="non-finite|Invalid JSON|invalid JSON"):
        load_examples_line(raw)


def test_rejects_unknown_fields(examples):
    with pytest.raises(ExampleError, match="unknown fields"):
        validate_example(_mutated(examples[0], weather="sunny"))


def test_rejects_unproven_answer_fold_claim(examples):
    bad = json.loads(json.dumps(examples[0]))
    bad["identity"]["answer_fold"] = "train"
    with pytest.raises(ExampleError, match="identity block"):
        validate_example(bad)


def test_rejects_example_id_drift(examples):
    bad = json.loads(json.dumps(examples[0]))
    bad["optimal_tie_set"] = ["crane"]
    with pytest.raises(ExampleError, match="example_id"):
        validate_example(bad)


def test_rejects_malformed_tie_set(examples):
    with pytest.raises(ExampleError):
        validate_example(_mutated(examples[0], optimal_tie_set=[]))
    with pytest.raises(ExampleError):
        validate_example(_mutated(examples[0], optimal_tie_set=["abc"]))
    with pytest.raises(ExampleError):
        validate_example(
            _mutated(examples[0], optimal_tie_set=[examples[0]["optimal_tie_set"][0]] * 2)
        )


def test_rejects_wrong_track(examples):
    bad = json.loads(json.dumps(examples[0]))
    bad["identity"]["exposure_track"] = "answer-exposed"
    with pytest.raises(ExampleError, match="compiled track"):
        validate_example(bad)


def test_legal_mask_digest_binds_recompute(examples):
    bad = json.loads(json.dumps(examples[0]))
    bad["legal_mask_digest"] = "f" * 64
    bad["example_id"] = example_id_for(bad)
    with pytest.raises(ExampleError, match="legal_mask_digest"):
        project_batch([bad], _WORDS)


def test_projection_serializes_without_answer_fields(examples):
    batch = project_batch(examples[:3], _WORDS)
    assert batch.to_public_dict()["action_space_sha256"] == ActionSpace.parse(_WORDS).digest
    assert batch.to_public_dict()["row_masks"] == batch.row_masks
    for mask, letters_rows in zip(batch.row_masks, batch.row_letters):
        assert mask == [True] * len(letters_rows)
    blob = json.dumps(batch.to_public_dict())
    for marker in ("answer_id", "audit_answer", "posterior", "sparse_action_values"):
        assert marker not in blob


def test_verified_certificates_reproduce_fixture_byte_for_byte():
    certificate_root = REPO_ROOT / "artifacts" / "oracle" / "openwordle-v1"
    certificates = [
        certificate_root
        / "openwordle-v1.uniform.normal.full-compiled@512"
        / "certificate.json",
        certificate_root
        / "openwordle-v1.uniform.hard.full-compiled@512"
        / "certificate.json",
    ]
    generated = generate_root_examples(
        certificates,
        frozen_root=REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen",
        repo_root=REPO_ROOT,
    )
    generated_bytes = b"".join(
        canonical_example_bytes(record) + b"\n" for record in generated
    )
    assert generated_bytes == FIXTURE.read_bytes()


def test_hard_legality_uses_rules_profile_without_gray_letter_bans(examples):
    from deep_green.rules import legality_digest

    actions = ActionSpace.parse(("alley", "crane", "llama"))
    record = json.loads(json.dumps(next(
        item for item in examples if item["public_state"]["mode"] == "hard"
    )))
    record["identity"]["action_space_sha256"] = actions.digest
    record["public_state"] = {
        "mode": "hard",
        "rows": [["allee", "yybbb"]],
        "turn": 2,
    }
    record["optimal_tie_set"] = ["alley"]
    record["sparse_action_values"] = [
        {"word": "alley", "losses": 0, "score7_total": 1}
    ]
    record["legal_mask_digest"] = legality_digest((True, False, True))
    record["example_id"] = example_id_for(record)
    batch = project_batch([record], actions.words)
    assert batch.legal_masks == [(True, False, True)]
