"""Atomic freeze publish, complete inventory, standalone verification, deletion matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from tests.benchmark import repo_path
from tests.benchmark.test_candidate_manifest import CANDIDATE_DIR
from tests.benchmark.test_views import RELEASE_CELL_IDS

PROTOCOL_PATH = repo_path("benchmarks", "openwordle-v1", "protocol.toml")
RULES_PATH = repo_path("benchmarks", "rules", "openwordle-rules-v1.json")
PRIOR_PATH = repo_path("benchmarks", "openwordle-v1", "priors", "uniform-v1.json")

INVENTORY = (
    ("rules.json", RULES_PATH.read_bytes()),
    ("prior.json", PRIOR_PATH.read_bytes()),
    ("protocol.toml", PROTOCOL_PATH.read_bytes()),
)

DELETION_MATRIX = (
    "corpus/NOTICE-SCOWL.txt",
    "corpus/tiers/tier-1024.txt",
    "views/views.json",
    "corpus/words/accepted-guesses.txt",
    "corpus/splits/ah-test.txt",
    "rules.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def staged_bundle(tmp_path_factory):
    from deep_green.benchmark import load_protocol, stage_freeze

    protocol = load_protocol(PROTOCOL_PATH)
    staging = tmp_path_factory.mktemp("stage") / "frozen-staging"
    stage_freeze(protocol, CANDIDATE_DIR, staging)
    return staging


def test_staged_directory_holds_the_full_inventory(staged_bundle):
    for rel, expected_bytes in INVENTORY:
        assert (staged_bundle / rel).is_file(), f"missing {rel}"
        assert (staged_bundle / rel).read_bytes() == expected_bytes


def test_staged_corpus_provenance_copies_words_tiers_splits_notices_and_manifest(staged_bundle):
    manifest = json.loads((staged_bundle / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads((CANDIDATE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == original["schema"]
    for rel, digest in sorted(original["files"].items()):
        frozen_copy = staged_bundle / "corpus" / rel
        assert frozen_copy.is_file(), f"frozen bundle lacks corpus file {rel}"
        assert sha256_bytes(frozen_copy.read_bytes()) == digest
    for notice in ("NOTICE-SCOWL.txt", "NOTICE-WordNet.txt"):
        assert (staged_bundle / "corpus" / notice).is_file()
        assert (staged_bundle / "corpus" / notice).read_bytes() == (
            CANDIDATE_DIR / notice
        ).read_bytes()


def test_staged_views_cover_the_release_cells(staged_bundle):
    records = json.loads((staged_bundle / "views" / "views.json").read_text(encoding="utf-8"))
    ids = {record["view_id"] for record in records}
    for cell in RELEASE_CELL_IDS:
        assert cell in ids


def test_publish_is_atomic_stage_fsync_rename_and_failure_leaves_nothing_readable(
    tmp_path, monkeypatch
):
    from deep_green import benchmark as benchmark_module
    from deep_green.benchmark import load_protocol, publish_freeze

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"

    calls: list[str] = []

    real_fsync = os.fsync

    def spy_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    monkeypatch.setattr(benchmark_module.os, "fsync", spy_fsync)

    publish_freeze(protocol, CANDIDATE_DIR, target)
    assert calls[:2] == ["fsync", "fsync"], "publish must fsync files and the staged tree before rename"
    assert target.is_dir()
    assert (target / "manifest.json").is_file()

    calls.clear()

    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("injected rename failure")

    monkeypatch.setattr(benchmark_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        publish_freeze(protocol, CANDIDATE_DIR, tmp_path / "frozen-retry")
    assert not (tmp_path / "frozen-retry").exists(), "a failed rename must leave no readable partial bundle"
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert not any(name.startswith("frozen-retry") for name in leftovers), (
        f"staging debris survived the failure: {leftovers}"
    )
    monkeypatch.setattr(benchmark_module.os, "replace", real_replace)
    publish_freeze(protocol, CANDIDATE_DIR, tmp_path / "frozen-retry")
    assert (tmp_path / "frozen-retry" / "manifest.json").is_file()


def test_republish_over_an_existing_frozen_dir_is_refused(tmp_path):
    from deep_green.benchmark import BundleError, load_protocol, publish_freeze

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    with pytest.raises(BundleError, match="already exists"):
        publish_freeze(protocol, CANDIDATE_DIR, target)


def test_verify_passes_inside_the_repo_tree(tmp_path):
    from deep_green.benchmark import (
        load_protocol,
        publish_freeze,
        verify_frozen_bundle,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    report = verify_frozen_bundle(target / "manifest.json")
    assert report.view_count >= len(RELEASE_CELL_IDS) + 4
    cells = {record["view_id"] if isinstance(record, dict) else record.view_id for record in report.views}
    for cell in RELEASE_CELL_IDS:
        assert cell in cells


def test_verify_is_standalone_from_a_bare_copy_of_frozen(tmp_path):
    from deep_green.benchmark import (
        load_protocol,
        publish_freeze,
        verify_frozen_bundle,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    copy_root = tmp_path / "standalone"
    copy_root.mkdir()
    shutil.copytree(target, copy_root / "frozen")

    report = verify_frozen_bundle(copy_root / "frozen" / "manifest.json")
    assert report.verified is True


@pytest.mark.parametrize("victim", DELETION_MATRIX)
def test_deletion_matrix_every_removed_file_fails_verification_naming_it(tmp_path, victim):
    from deep_green.benchmark import (
        BundleError,
        load_protocol,
        publish_freeze,
        verify_frozen_bundle,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    copy_root = tmp_path / "matrix"
    copy_root.mkdir()
    shutil.copytree(target, copy_root / "frozen")
    (copy_root / "frozen" / victim).unlink()

    with pytest.raises(BundleError) as excinfo:
        verify_frozen_bundle(copy_root / "frozen" / "manifest.json")
    message = str(excinfo.value)
    assert Path(victim).name in message or victim in message, (
        f"error must name the deleted artifact: {message}"
    )


def test_single_byte_corruption_in_a_frozen_file_fails_verification(tmp_path):
    from deep_green.benchmark import (
        BundleError,
        load_protocol,
        publish_freeze,
        verify_frozen_bundle,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    copy_root = tmp_path / "tampered"
    copy_root.mkdir()
    shutil.copytree(target, copy_root / "frozen")
    tier = copy_root / "frozen" / "corpus" / "tiers" / "tier-512.txt"
    data = bytearray(tier.read_bytes())
    data[-3] ^= 0x01
    tier.write_bytes(bytes(data))

    with pytest.raises(BundleError, match="tier-512"):
        verify_frozen_bundle(copy_root / "frozen" / "manifest.json")


def test_loader_writes_view_access_audit_events(tmp_path):
    from deep_green.benchmark import (
        load_protocol,
        publish_freeze,
        verify_frozen_bundle,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    audit_path = repo_path("artifacts", "audit", "view-access.log")
    audit_before = len(audit_path.read_text().splitlines()) if audit_path.is_file() else 0
    verify_frozen_bundle(target / "manifest.json")
    lines = audit_path.read_text().splitlines()
    new_entries = lines[audit_before:]
    assert new_entries, "loading a frozen view must append an access event"
    assert any(RELEASE_CELL_IDS[0] in line for line in new_entries)
