"""Pin verification: the vendor tree matches UPSTREAM.toml exactly."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.benchmark import repo_path

VENDOR_DIR = repo_path("vendor", "alex1770-wordle")


def _load_spec():
    from deep_green.oracle.upstream import load_upstream_spec

    return load_upstream_spec(VENDOR_DIR / "UPSTREAM.toml")


def test_upstream_pin_records_the_expected_commit_and_license():
    spec = _load_spec()
    assert spec.commit == "8fd3f1bdf884ff6757c24ad1227dc561fab7787d"
    assert spec.upstream_url == "https://github.com/alex1770/wordle"
    assert spec.license == "MIT"


def test_every_committed_file_hash_matches_the_pin():
    from deep_green.oracle.upstream import validate_vendor_tree

    spec = _load_spec()
    validate_vendor_tree(VENDOR_DIR, spec)


def test_wordlist_data_files_are_recorded_as_withheld():
    spec = _load_spec()
    statuses = {entry.path: entry.status for entry in spec.files}
    withheld = [path for path, status in statuses.items() if status == "withheld"]
    assert withheld, "the pin must record withheld upstream files to stay auditable"
    assert "wordlist_orig_hidden" in withheld
    assert "wordlist_orig_all" in withheld
    committed_paths = {entry.path for entry in spec.files if entry.status == "committed"}
    assert "wordle.cpp" in committed_paths
    assert "LICENSE" in committed_paths


def test_committed_source_matches_its_recorded_pin_hash():
    """Committed wordle.cpp is the byte-pristine upstream blob; the patch applies at build time."""
    import hashlib

    spec = _load_spec()
    source_entry = next(entry for entry in spec.files if entry.path == "wordle.cpp")
    current = hashlib.sha256((VENDOR_DIR / "wordle.cpp").read_bytes()).hexdigest()
    assert current == source_entry.sha256, (
        "vendored wordle.cpp must stay byte-pristine; local changes live only in patches/"
    )
    assert current == "a3bd676c581703a63e29bfe5e922cfe1409e5c3b276cd0ddbbe20ca5353d1a93"
    patch_text = (VENDOR_DIR / "patches" / "events-v1.patch").read_text(encoding="utf-8")
    assert "case 'E'" in patch_text
    assert "action_value" in patch_text
    assert "covered_actions" in patch_text
    assert "declared_actions" in patch_text


def test_unrecorded_file_is_rejected(tmp_path):
    from deep_green.oracle.upstream import UpstreamSpecError, validate_vendor_tree

    copy_dir = tmp_path / "vendor-copy"
    shutil.copytree(VENDOR_DIR, copy_dir)
    (copy_dir / "sneaky-extra.txt").write_text("unrecorded", encoding="utf-8")
    try:
        validate_vendor_tree(copy_dir, _load_spec())
    except UpstreamSpecError as exc:
        assert "unrecorded" in str(exc)
    else:
        raise AssertionError("an unrecorded file must fail pin validation")


def test_hash_drift_is_rejected(tmp_path):
    from deep_green.oracle.upstream import UpstreamSpecError, validate_vendor_tree

    copy_dir = tmp_path / "vendor-drift"
    shutil.copytree(VENDOR_DIR, copy_dir)
    target = copy_dir / "wordle.cpp"
    text = target.read_text(encoding="utf-8")
    target.write_text(text + "\n// drift\n", encoding="utf-8")
    try:
        validate_vendor_tree(copy_dir, _load_spec())
    except UpstreamSpecError as exc:
        assert "drift" in str(exc)
    else:
        raise AssertionError("hash drift must fail pin validation")


def test_missing_pinned_file_is_rejected(tmp_path):
    from deep_green.oracle.upstream import UpstreamSpecError, validate_vendor_tree

    copy_dir = tmp_path / "vendor-missing"
    shutil.copytree(VENDOR_DIR, copy_dir)
    (copy_dir / "compat" / "error.h").unlink()
    try:
        validate_vendor_tree(copy_dir, _load_spec())
    except UpstreamSpecError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("a missing pinned file must fail validation")


def test_solver_compiles_from_the_patched_tree(tmp_path):
    from deep_green.oracle.build import build_solver

    exe = build_solver(VENDOR_DIR, tmp_path)
    assert exe.is_file() and exe.stat().st_size > 10000


def test_inspect_solver_accepts_a_valid_cache(tmp_path):
    from deep_green.oracle.build import ensure_solver, inspect_solver

    cache = tmp_path / "cache"
    built = ensure_solver(VENDOR_DIR, cache)
    assert inspect_solver(VENDOR_DIR, cache) == built


def test_inspect_solver_rejects_a_tampered_binary(tmp_path):
    from deep_green.oracle.build import (
        BINARY_NAME,
        OracleBuildError,
        ensure_solver,
        inspect_solver,
    )

    cache = tmp_path / "cache"
    ensure_solver(VENDOR_DIR, cache)
    binary = cache / BINARY_NAME
    binary.write_bytes(binary.read_bytes() + b"drift")
    with pytest.raises(OracleBuildError, match="does not match"):
        inspect_solver(VENDOR_DIR, cache)


def test_inspect_solver_rejects_a_malformed_marker(tmp_path):
    from deep_green.oracle.build import (
        OracleBuildError,
        ensure_solver,
        inspect_solver,
    )

    cache = tmp_path / "cache"
    ensure_solver(VENDOR_DIR, cache)
    (cache / "build-marker.json").write_text("not json", encoding="utf-8")
    with pytest.raises(OracleBuildError, match="marker"):
        inspect_solver(VENDOR_DIR, cache)


def test_inspect_solver_never_builds(tmp_path):
    from deep_green.oracle.build import OracleBuildError, inspect_solver

    cache = tmp_path / "empty-cache"
    with pytest.raises(OracleBuildError, match="missing"):
        inspect_solver(VENDOR_DIR, cache)
    assert not cache.exists()


def test_exhaustive_behavior_identical_without_n_flag(tmp_path):
    """The port preserves exhaustive semantics when -n/-N are omitted."""
    from deep_green.oracle.build import build_solver

    exe = build_solver(VENDOR_DIR, tmp_path)
    all_txt = tmp_path / "all.txt"
    hidden_txt = tmp_path / "hidden.txt"
    all_txt.write_text("crane\nslate\naudio\npouty\n", encoding="utf-8")
    hidden_txt.write_text("slate\npouty\n", encoding="utf-8")
    completed = subprocess.run(
        [str(exe), "-a", str(all_txt), "-h", str(hidden_txt)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert "Best first guess score = 3" in completed.stdout
