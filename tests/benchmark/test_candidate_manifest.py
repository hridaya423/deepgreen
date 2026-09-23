"""Independent candidate preflight: trust no count or digest from the manifest."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from tests.benchmark import repo_path

CANDIDATE_DIR = repo_path("artifacts", "corpus", "openwordle-v1-candidate")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FROZEN_COUNTS = {
    "words/accepted-guesses.txt": 5022,
    "intermediates/scowl-audit-35.txt": 3421,
    "intermediates/wordnet-five-letter-hex.txt": 5358,
    "answers/primary-2095.txt": 2095,
    "answers/broad-3048.txt": 3048,
    "tiers/tier-512.txt": 512,
    "tiers/tier-1024.txt": 1024,
    "tiers/tier-2048.txt": 2048,
    "tiers/tier-2095.txt": 2095,
    "splits/ah-train.txt": 1677,
    "splits/ah-dev.txt": 209,
    "splits/ah-test.txt": 209,
    "NOTICE-SCOWL.txt": 6,
    "NOTICE-WordNet.txt": 8,
}


def run_preflight(candidate):
    from deep_green.benchmark import run_preflight as _run

    return _run(candidate)


def copy_candidate(tmp_path: Path) -> Path:
    target = tmp_path / "candidate"
    shutil.copytree(CANDIDATE_DIR, target)
    return target


def refresh_manifest(candidate: Path) -> None:
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    files = {}
    for rel in sorted(manifest["files"]):
        files[rel] = sha256_bytes((candidate / rel).read_bytes())
    counts = {rel: len((candidate / rel).read_text(encoding="utf-8").splitlines()) for rel in files}
    manifest["files"] = files
    manifest["counts"] = counts
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_intact_candidate_passes_with_recomputed_counts():
    report = run_preflight(CANDIDATE_DIR)
    assert report.recomputed_counts == FROZEN_COUNTS
    for rel, digest in report.digests.items():
        data = (CANDIDATE_DIR / rel).read_bytes()
        assert digest == sha256_bytes(data)


def test_inflated_manifest_count_is_caught_by_the_recount(tmp_path):
    candidate = copy_candidate(tmp_path)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["words/accepted-guesses.txt"] = 5023
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception) as excinfo:
        run_preflight(candidate)
    assert "words/accepted-guesses.txt" in str(excinfo.value)
    assert "5023" in str(excinfo.value)


def test_swapped_file_with_consistent_manifest_is_caught_by_recomputation(tmp_path):
    candidate = copy_candidate(tmp_path)
    primary = (candidate / "answers/primary-2095.txt").read_text(encoding="utf-8").splitlines()
    first, second = primary[0], primary[1]
    primary[0], primary[1] = second, first
    (candidate / "answers/primary-2095.txt").write_text(
        "\n".join(primary) + "\n", encoding="utf-8"
    )
    refresh_manifest(candidate)
    with pytest.raises(Exception, match="tier-512|sorted|rank"):
        run_preflight(candidate)


def test_fold_content_swap_of_equal_size_is_caught_by_rederivation(tmp_path):
    candidate = copy_candidate(tmp_path)
    dev = (candidate / "splits/ah-dev.txt").read_text(encoding="utf-8").splitlines()
    train = (candidate / "splits/ah-train.txt").read_text(encoding="utf-8").splitlines()
    dev[0], dev[-1] = train[0], train[-1]
    (candidate / "splits/ah-dev.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    refresh_manifest(candidate)
    with pytest.raises(Exception, match="fold"):
        run_preflight(candidate)


def test_answer_outside_accepted_vocabulary_is_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    accepted = (candidate / "words/accepted-guesses.txt").read_text(encoding="utf-8").splitlines()
    primary = (candidate / "answers/primary-2095.txt").read_text(encoding="utf-8").splitlines()
    outside = "zzzzz"
    while outside in set(accepted):
        outside = outside[:-1] + chr(ord(outside[-1]) + 1)
    accepted.remove(primary[0])
    accepted.append(outside)
    accepted.sort()
    (candidate / "words/accepted-guesses.txt").write_text(
        "\n".join(accepted) + "\n", encoding="utf-8"
    )
    primary[0] = outside
    primary.sort()
    (candidate / "answers/primary-2095.txt").write_text(
        "\n".join(primary) + "\n", encoding="utf-8"
    )
    refresh_manifest(candidate)
    with pytest.raises(Exception, match="accepted"):
        run_preflight(candidate)


def test_tier_that_breaks_the_hash_rank_prefix_is_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    tier = (candidate / "tiers/tier-1024.txt").read_text(encoding="utf-8").splitlines()
    tier[700], tier[701] = tier[701], tier[700]
    (candidate / "tiers/tier-1024.txt").write_text("\n".join(tier) + "\n", encoding="utf-8")
    refresh_manifest(candidate)
    with pytest.raises(Exception, match="rank|tier"):
        run_preflight(candidate)


def test_overlapping_ah_folds_are_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    dev = (candidate / "splits/ah-dev.txt").read_text(encoding="utf-8").splitlines()
    test = (candidate / "splits/ah-test.txt").read_text(encoding="utf-8").splitlines()
    train = (candidate / "splits/ah-train.txt").read_text(encoding="utf-8").splitlines()
    moved = dev.pop()
    test.append(moved)
    dev.append(train.pop())
    for name, words in (("dev", sorted(dev)), ("test", sorted(test)), ("train", train)):
        (candidate / f"splits/ah-{name}.txt").write_text(
            "\n".join(words) + "\n", encoding="utf-8"
        )
    refresh_manifest(candidate)
    with pytest.raises(Exception, match="fold"):
        run_preflight(candidate)


def test_missing_notice_file_is_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    (candidate / "NOTICE-SCOWL.txt").unlink()
    with pytest.raises(Exception, match="NOTICE"):
        run_preflight(candidate)


def test_unregistered_extra_output_file_is_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    (candidate / "words").mkdir(exist_ok=True)
    (candidate / "words/extra-diagnostic-answers.txt").write_text("crane\n", encoding="utf-8")
    with pytest.raises(Exception, match="unregistered"):
        run_preflight(candidate)


def test_foreign_source_entry_in_the_manifest_is_refused(tmp_path):
    candidate = copy_candidate(tmp_path)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    manifest["sources"] = {
        "scowl_v1": {"commit": manifest["scowl_commit"]},
        "wordfreq": {"version": "0.9.1"},
    }
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="wordfreq"):
        run_preflight(candidate)


def test_divergent_salt_interpretation_cannot_pass(tmp_path):
    candidate = copy_candidate(tmp_path)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    manifest["salt_hex"] = "openwordle-v1-as-text"
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="salt"):
        run_preflight(candidate)


def test_preflight_reads_no_count_or_digest_from_the_manifest():
    from deep_green import benchmark as benchmark_module

    text = Path(benchmark_module.__file__).resolve().read_text(encoding="utf-8")
    preflight_start = text.index("def run_preflight")
    preflight_end = text.index("def view_id_for")
    body = text[preflight_start:preflight_end]
    assert not re.search(r"manifest\[.counts.\]", body), (
        "preflight must recompute counts from bytes; comparing against the manifest is fine, "
        "trusting it is not"
    )
    assert "counts[rel] = len(lines)" in body or "len(lines)" in body


def test_module_verify_entrypoint_still_stands_alone_on_the_candidate():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_green.corpus",
            "verify",
            str(CANDIDATE_DIR),
        ],
        cwd=repo_path(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
