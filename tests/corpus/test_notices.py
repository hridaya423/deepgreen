"""VAL-BENCH-007 hardening: notices are registered, digest-bound, and verified."""

from __future__ import annotations

import pytest

from tests.benchmark import repo_path
from tests.corpus.conftest import read_candidate_json

CANDIDATE_DIR = repo_path("artifacts", "corpus", "openwordle-v1-candidate")

REQUIRED_NOTICES = ("NOTICE-SCOWL.txt", "NOTICE-WordNet.txt")


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_candidate_manifest_registers_both_notice_files():
    manifest = read_candidate_json(CANDIDATE_DIR, "manifest.json")
    for notice in REQUIRED_NOTICES:
        assert notice in manifest["files"], f"manifest files table missing {notice}"


def test_registered_notice_digests_match_the_candidate_bytes():
    manifest = read_candidate_json(CANDIDATE_DIR, "manifest.json")
    for notice in REQUIRED_NOTICES:
        data = (CANDIDATE_DIR / notice).read_bytes()
        assert manifest["files"][notice] == _sha256_bytes(data), f"{notice} digest drift"


def test_verify_candidate_rejects_a_corrupted_notice(tmp_path):
    from deep_green.corpus import VerificationError, verify_candidate

    target = tmp_path / "candidate"
    target.mkdir()
    for item in CANDIDATE_DIR.rglob("*"):
        rel = item.relative_to(CANDIDATE_DIR)
        if item.is_dir():
            (target / rel).mkdir(parents=True, exist_ok=True)
        else:
            (target / rel).write_bytes(item.read_bytes())
    notice = target / "NOTICE-WordNet.txt"
    notice.write_text(notice.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerificationError, match="NOTICE-WordNet"):
        verify_candidate(target)


def test_verify_candidate_rejects_a_deleted_notice(tmp_path):
    from deep_green.corpus import VerificationError, verify_candidate

    target = tmp_path / "candidate"
    target.mkdir()
    for item in CANDIDATE_DIR.rglob("*"):
        rel = item.relative_to(CANDIDATE_DIR)
        if item.is_dir():
            (target / rel).mkdir(parents=True, exist_ok=True)
        else:
            (target / rel).write_bytes(item.read_bytes())
    (target / "NOTICE-SCOWL.txt").unlink()
    with pytest.raises(VerificationError, match="NOTICE-SCOWL"):
        verify_candidate(target)


def test_verify_candidate_accepts_the_manifest_json_path_form():
    """VAL-BENCH-007 hardening: the documented path form must verify intact trees."""

    from deep_green.corpus import verify_candidate

    report = verify_candidate(CANDIDATE_DIR / "manifest.json")
    assert report["verified"] is True


def test_verify_candidate_on_directory_and_path_forms_agree():
    from deep_green.corpus import verify_candidate

    by_dir = verify_candidate(CANDIDATE_DIR)
    by_file = verify_candidate(CANDIDATE_DIR / "manifest.json")
    assert by_dir == by_file
