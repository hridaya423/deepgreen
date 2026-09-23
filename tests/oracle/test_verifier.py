"""Verifier tests: a synthetic tiny certificate plus a corruption matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ACTIONS = ("crane", "pouty", "slate")
ANSWERS = ("pouty", "slate")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _feedback_key(guess: str, answer: str) -> str:
    from deep_green.rules import Tile, Word5, score_guess

    code = {Tile.ABSENT: "0", Tile.PRESENT: "1", Tile.CORRECT: "2"}
    feedback = score_guess(Word5.parse(guess), Word5.parse(answer))
    return "".join(code[tile] for tile in feedback.tiles)


@pytest.fixture
def cert_env(tmp_path):
    frozen_root = tmp_path / "frozen"
    corpus = frozen_root / "corpus"
    (corpus / "words").mkdir(parents=True)
    (corpus / "tiers").mkdir(parents=True)
    actions_bytes = ("\n".join(ACTIONS) + "\n").encode("utf-8")
    answers_bytes = ("\n".join(ANSWERS) + "\n").encode("utf-8")
    (corpus / "words" / "accepted-guesses.txt").write_bytes(actions_bytes)
    (corpus / "tiers" / "tier-2.txt").write_bytes(answers_bytes)

    vendor_root = tmp_path / "vendor" / "alex1770-wordle"
    (vendor_root / "patches").mkdir(parents=True)
    (vendor_root / "compat").mkdir(parents=True)
    (vendor_root / "wordle.cpp").write_bytes(b"// synthetic pinned source\n")
    (vendor_root / "patches" / "events-v1.patch").write_bytes(b"--- synthetic patch\n")
    (vendor_root / "compat" / "error.h").write_bytes(b"// synthetic shim\n")

    from deep_green.oracle.build import (
        BINARY_NAME,
        COMPILE_FLAGS,
        compiler_identity,
        solver_digest,
    )

    build_cache = tmp_path / "build-cache"
    build_cache.mkdir()
    binary_path = build_cache / BINARY_NAME
    binary_path.write_bytes(b"synthetic solver binary\n")
    binary_sha256 = solver_digest(binary_path)
    compiler_id = compiler_identity()
    marker = {
        "source_sha256": _sha((vendor_root / "wordle.cpp").read_bytes()),
        "patch_sha256": _sha(
            (vendor_root / "patches" / "events-v1.patch").read_bytes()
        ),
        "shim_sha256": _sha((vendor_root / "compat" / "error.h").read_bytes()),
        "flags": list(COMPILE_FLAGS),
        "compiler_id": compiler_id,
        "binary_sha256": binary_sha256,
    }
    (build_cache / "build-marker.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )

    from deep_green.rules import ActionSpace

    action_space = ActionSpace.parse(ACTIONS)
    view = {
        "view_id": "tiny-v1.uniform.normal.test",
        "tier": 2,
        "mode": "normal",
        "track": "full-compiled",
        "rules_sha256": "a" * 64,
        "prior_id": "uniform-v1",
        "action_space_sha256": _sha(actions_bytes),
        "answers_sha256": _sha(answers_bytes),
        "answers_file": "tiers/tier-2.txt",
        "horizon_turns": 6,
        "legality_profile_id": None,
        "tie_order": "lowest_canonical_action_index_wins",
        "loss_scoring": "turns_to_solve_unsolved_scores_seven",
        "objective": "lexicographic_minimize_losses_then_maximize_score7_total_then_minimize_worst_score",
    }
    views_dir = frozen_root / "views"
    views_dir.mkdir(parents=True)
    (views_dir / "views.json").write_text(json.dumps([view]), encoding="utf-8")

    manifest = {
        "benchmark_id": "tiny-v1",
        "files": {
            "views/views.json": _sha((views_dir / "views.json").read_bytes()),
            "corpus/words/accepted-guesses.txt": _sha(actions_bytes),
            "corpus/tiers/tier-2.txt": _sha(answers_bytes),
        },
    }
    (frozen_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    slate_key = _feedback_key("pouty", "slate")
    certificate = {
        "schema": "wordle.oracle-certificate/v1",
        "identity": {**view, "benchmark_id": "tiny-v1"},
        "binding": {
            "action_space_sha256": _sha(actions_bytes),
            "answers_sha256": _sha(answers_bytes),
            "action_space_canonical_digest": action_space.digest,
            "answers_canonical_digest": hashlib.sha256(
                "\n".join(ANSWERS).encode("utf-8")
            ).hexdigest(),
            "solver_source_sha256": _sha((vendor_root / "wordle.cpp").read_bytes()),
            "patch_sha256": _sha(
                (vendor_root / "patches" / "events-v1.patch").read_bytes()
            ),
            "compiler_id": compiler_id,
            "executable_sha256": binary_sha256,
        },
        "first_action_proof": {
            "declared_actions": 3,
            "winner_index": 1,
            "winner_word": "pouty",
            "winner_score7_total": 3,
            "actions": [
                {"action_index": 0, "word": "crane", "bound": 4, "exact": True},
                {"action_index": 1, "word": "pouty", "bound": 3, "exact": True},
                {"action_index": 2, "word": "slate", "bound": 3, "exact": True},
            ],
        },
        "search": {
            "exhaustive": True,
            "best_first_word": "pouty",
            "nodes": 3,
            "time_seconds": 0.0,
        },
        "objective_value": {"losses": 0, "score7_total": 3, "worst_score": 2},
        "policy_tree": {
            "guess": "pouty",
            "children": {
                "22222": {"terminal": "solved", "turn": 1},
                slate_key: {
                    "guess": "slate",
                    "children": {"22222": {"terminal": "solved", "turn": 2}},
                },
            },
        },
    }
    cert_path = tmp_path / "certificate.json"
    cert_path.write_text(
        json.dumps(certificate, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(
        tmp_path=tmp_path,
        frozen_root=frozen_root,
        vendor_root=vendor_root,
        build_cache=build_cache,
        certificate=certificate,
        cert_path=cert_path,
    )


def _verify(env, path: Path):
    from deep_green.oracle.verifier import verify_certificate

    return verify_certificate(
        path,
        frozen_root=env.frozen_root,
        vendor_root=env.vendor_root,
        build_cache=env.build_cache,
    )


def _write(env, cert: dict) -> Path:
    path = env.tmp_path / "corrupt.json"
    path.write_text(
        json.dumps(cert, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _corrupt(env, mutate) -> Path:
    cert = copy.deepcopy(env.certificate)
    mutate(cert)
    return _write(env, cert)


def test_clean_certificate_verifies(cert_env):
    stats = _verify(cert_env, cert_env.cert_path)
    assert stats.answers_reached == len(ANSWERS)
    assert stats.losses == 0
    assert stats.total_score == 3
    assert stats.worst_score == 2


def test_missing_vendor_root_fails(cert_env, tmp_path):
    from deep_green.oracle.verifier import CertificateError, verify_certificate

    with pytest.raises(CertificateError, match="vendor root"):
        verify_certificate(
            cert_env.cert_path,
            frozen_root=cert_env.frozen_root,
            vendor_root=tmp_path / "no-such-vendor",
        )


def test_corrupt_action_outside_vocab_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    slate_key = _feedback_key("pouty", "slate")

    def mutate(cert):
        cert["policy_tree"]["children"][slate_key]["guess"] = "zzzzz"

    with pytest.raises(VerificationError, match="outside the frozen action vocabulary"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_child_partition_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        children = cert["policy_tree"]["children"]
        del children[min(children)]

    with pytest.raises(VerificationError, match="partition incomplete"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_feedback_edge_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    slate_key = _feedback_key("pouty", "slate")

    def mutate(cert):
        children = cert["policy_tree"]["children"]
        flipped = str(int(slate_key[0]) ^ 1) + slate_key[1:]
        children[flipped] = children.pop(slate_key)

    with pytest.raises(VerificationError, match="impossible feedback|partition incomplete"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_identity_field_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["identity"]["answers_sha256"] = "0" * 64

    with pytest.raises(CertificateError, match="does not match"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_source_hash_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["binding"]["solver_source_sha256"] = "0" * 64

    with pytest.raises(VerificationError, match="drifts"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_patch_hash_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["binding"]["patch_sha256"] = "0" * 64

    with pytest.raises(VerificationError, match="drifts"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_objective_value_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["objective_value"]["score7_total"] += 1

    with pytest.raises(VerificationError, match="score7_total mismatch"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_non_exhaustive_search_flag_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["search"]["exhaustive"] = False

    with pytest.raises(CertificateError, match="non-exhaustive"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_missing_proof_action_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        del cert["first_action_proof"]["actions"][0]

    with pytest.raises(CertificateError, match="cover every action"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_duplicate_proof_action_index_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["first_action_proof"]["actions"][2]["action_index"] = 1

    with pytest.raises(CertificateError, match="duplicates action_index"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_proof_word_index_drift_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["first_action_proof"]["actions"][0]["word"] = "slate"

    with pytest.raises(CertificateError, match="does not match canonical"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_nonexact_winner_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["first_action_proof"]["actions"][1]["exact"] = False

    with pytest.raises(CertificateError, match="exact bound"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_winner_bound_mismatch_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["first_action_proof"]["actions"][1]["bound"] = 4

    with pytest.raises(CertificateError, match="winner_score7_total"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_action_bound_below_winner_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["first_action_proof"]["actions"][0]["bound"] = 2

    with pytest.raises(VerificationError, match="undercuts"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_tree_root_winner_mismatch_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["policy_tree"]["guess"] = "slate"

    with pytest.raises(VerificationError, match="proven winner"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_missing_first_action_proof_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        del cert["first_action_proof"]

    with pytest.raises(CertificateError, match="first_action_proof"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_noncanonical_tied_winner_is_rejected(cert_env):
    """pouty and slate tie at bound 3; canonical order requires pouty (index 1)."""
    from deep_green.oracle.verifier import CertificateError

    pouty_key = _feedback_key("slate", "pouty")

    def mutate(cert):
        proof = cert["first_action_proof"]
        proof["winner_index"] = 2
        proof["winner_word"] = "slate"
        cert["search"]["best_first_word"] = "slate"
        cert["policy_tree"] = {
            "guess": "slate",
            "children": {
                "22222": {"terminal": "solved", "turn": 1},
                pouty_key: {
                    "guess": "pouty",
                    "children": {"22222": {"terminal": "solved", "turn": 2}},
                },
            },
        }

    with pytest.raises(CertificateError, match="canonical tie order"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_missing_vendor_file_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    (cert_env.vendor_root / "patches" / "events-v1.patch").unlink()
    with pytest.raises(CertificateError, match="vendor file missing"):
        _verify(cert_env, cert_env.cert_path)


def test_corrupt_benchmark_id_is_rejected(cert_env):
    from deep_green.oracle.verifier import CertificateError

    def mutate(cert):
        cert["identity"]["benchmark_id"] = "other-benchmark"

    with pytest.raises(CertificateError, match="benchmark_id"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_executable_hash_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["binding"]["executable_sha256"] = "0" * 64

    with pytest.raises(VerificationError, match="executable"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_compiler_id_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["binding"]["compiler_id"] = "forged-compiler"

    with pytest.raises(VerificationError, match="compiler"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_corrupt_search_best_first_word_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    def mutate(cert):
        cert["search"]["best_first_word"] = "slate"

    with pytest.raises(VerificationError, match="best_first_word"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_coordinated_views_and_certificate_drift_is_rejected(cert_env):
    """views.json edited to agree with a corrupted cert still fails the manifest."""
    from deep_green.oracle.verifier import VerificationError

    views_path = cert_env.frozen_root / "views" / "views.json"
    views = json.loads(views_path.read_text(encoding="utf-8"))
    views[0]["horizon_turns"] = 5
    views_path.write_text(json.dumps(views), encoding="utf-8")

    def mutate(cert):
        cert["identity"]["horizon_turns"] = 5

    with pytest.raises(VerificationError, match="manifest"):
        _verify(cert_env, _corrupt(cert_env, mutate))


def test_frozen_action_file_drift_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    path = cert_env.frozen_root / "corpus" / "words" / "accepted-guesses.txt"
    path.write_bytes(path.read_bytes() + b"zzzzz\n")
    with pytest.raises(VerificationError, match="manifest"):
        _verify(cert_env, cert_env.cert_path)


def test_frozen_answer_file_drift_is_rejected(cert_env):
    from deep_green.oracle.verifier import VerificationError

    path = cert_env.frozen_root / "corpus" / "tiers" / "tier-2.txt"
    path.write_bytes(path.read_bytes() + b"zzzzz\n")
    with pytest.raises(VerificationError, match="manifest"):
        _verify(cert_env, cert_env.cert_path)


def test_manifest_missing_entry_fails_closed(cert_env):
    from deep_green.oracle.verifier import CertificateError

    manifest_path = cert_env.frozen_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["views/views.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CertificateError, match="manifest"):
        _verify(cert_env, cert_env.cert_path)


def test_malformed_manifest_fails_closed(cert_env):
    from deep_green.oracle.verifier import CertificateError

    (cert_env.frozen_root / "manifest.json").write_text(
        json.dumps({"benchmark_id": "tiny-v1", "files": "nope"}), encoding="utf-8"
    )
    with pytest.raises(CertificateError, match="manifest"):
        _verify(cert_env, cert_env.cert_path)


def test_missing_build_cache_fails_without_building(cert_env, tmp_path):
    from deep_green.oracle.verifier import VerificationError, verify_certificate

    missing = tmp_path / "no-cache"
    with pytest.raises(VerificationError, match="solver build identity"):
        verify_certificate(
            cert_env.cert_path,
            frozen_root=cert_env.frozen_root,
            vendor_root=cert_env.vendor_root,
            build_cache=missing,
        )
    assert not missing.exists()


def test_stale_build_marker_fails_without_rebuild(cert_env):
    from deep_green.oracle.verifier import VerificationError

    marker = cert_env.build_cache / "build-marker.json"
    marker.write_text(json.dumps({"binary_sha256": "0" * 64}), encoding="utf-8")
    tampered = marker.read_bytes()
    with pytest.raises(VerificationError, match="solver build identity"):
        _verify(cert_env, cert_env.cert_path)
    assert marker.read_bytes() == tampered
