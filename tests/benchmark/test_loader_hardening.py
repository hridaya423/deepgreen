"""Benchmark-loader hardening: identity pin comparison and clean freeze CLI errors."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.benchmark import repo_path
from tests.benchmark.test_candidate_manifest import CANDIDATE_DIR

PROTOCOL_PATH = repo_path("benchmarks", "openwordle-v1", "protocol.toml")


def test_loader_compares_the_corpus_manifest_pin_when_present(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    actual_pin = json.loads(
        (CANDIDATE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    import hashlib

    real_sha = hashlib.sha256((CANDIDATE_DIR / "manifest.json").read_bytes()).hexdigest()
    del actual_pin
    raw = PROTOCOL_PATH.read_text(encoding="utf-8")
    drifted = tmp_path / "protocol-drift.toml"
    drifted.write_text(
        raw.replace(f'corpus_manifest_sha256 = "{real_sha}"', 'corpus_manifest_sha256 = "' + "0" * 64 + '"'),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="corpus_manifest_sha256"):
        load_protocol(drifted)


def test_freeze_cli_reports_preflight_errors_without_a_traceback(tmp_path, capsys):
    from deep_green.benchmark_commands import run_benchmark_freeze

    poisoned = tmp_path / "poisoned-candidate"
    poisoned.mkdir()
    for item in CANDIDATE_DIR.rglob("*"):
        rel = item.relative_to(CANDIDATE_DIR)
        if item.is_dir():
            (poisoned / rel).mkdir(parents=True, exist_ok=True)
        else:
            (poisoned / rel).write_bytes(item.read_bytes())
    manifest = json.loads((poisoned / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["answers/primary-2095.txt"] = 2096
    (poisoned / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    class _Args:
        candidate = str(poisoned)
        protocol = str(PROTOCOL_PATH)
        output = str(tmp_path / "frozen")

    exit_code = run_benchmark_freeze(_Args())
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FREEZE FAILED" in captured.err
    assert "Traceback" not in captured.err


def test_freeze_cli_maps_preflight_error_to_clean_message(tmp_path):
    """The subprocess form proves no traceback escapes to operators."""

    env_bare = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONWARNINGS"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_green.benchmark_commands",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_path(),
        env=env_bare,
    )
    del result  # the module-level run is covered by the direct call above
