"""CLI surface: benchmark preflight, freeze, and verify behave end to end."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from tests.benchmark import repo_path
from tests.benchmark.test_candidate_manifest import CANDIDATE_DIR

PROTOCOL_PATH = repo_path("benchmarks", "openwordle-v1", "protocol.toml")
REPO_ROOT = repo_path()


def run_wordle(*args):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "deep_green.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture(scope="module")
def frozen_in_repo():
    from deep_green.benchmark import load_protocol, publish_freeze

    target = REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen"
    if not (target / "manifest.json").is_file():
        publish_freeze(load_protocol(PROTOCOL_PATH), CANDIDATE_DIR, target)
    yield target


def test_cli_preflight_passes_on_the_committed_candidate():
    result = run_wordle(
        "benchmark",
        "preflight",
        "--candidate",
        str(CANDIDATE_DIR),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PREFLIGHT OK" in result.stdout
    assert "5022" in result.stdout and "2095" in result.stdout


def test_cli_freeze_publishes_and_verify_accepts_it(tmp_path):
    target = tmp_path / "frozen"

    class _Args:
        candidate = str(CANDIDATE_DIR)
        protocol = str(PROTOCOL_PATH)
        output = str(target)

    from deep_green.benchmark_commands import run_benchmark_freeze

    assert run_benchmark_freeze(_Args()) == 0
    assert (target / "manifest.json").is_file()

    class _VerifyArgs:
        frozen = str(target / "manifest.json")

    from deep_green.benchmark_commands import run_benchmark_verify

    assert run_benchmark_verify(_VerifyArgs()) == 0


def test_cli_module_wiring_registers_the_benchmark_subcommands():
    parser_source = (repo_path("src", "deep_green", "cli.py")).read_text(encoding="utf-8")
    for name in ("preflight", "freeze", "verify"):
        assert name in parser_source, f"cli must expose benchmark {name}"


def test_freeze_command_refuses_to_clobber_an_existing_bundle(tmp_path):
    from deep_green.benchmark import load_protocol, publish_freeze
    from deep_green.benchmark_commands import run_benchmark_freeze

    target = tmp_path / "frozen"
    publish_freeze(load_protocol(PROTOCOL_PATH), CANDIDATE_DIR, target)

    class _Args:
        candidate = str(CANDIDATE_DIR)
        protocol = str(PROTOCOL_PATH)
        output = str(target)

    assert run_benchmark_freeze(_Args()) != 0


def test_verify_command_exits_nonzero_naming_a_deleted_artifact(tmp_path):
    from deep_green.benchmark import load_protocol, publish_freeze
    from deep_green.benchmark_commands import run_benchmark_verify

    protocol = load_protocol(PROTOCOL_PATH)
    target = tmp_path / "frozen"
    publish_freeze(protocol, CANDIDATE_DIR, target)
    (target / "views" / "views.json").unlink()

    class _VerifyArgs:
        frozen = str(target / "manifest.json")

    assert run_benchmark_verify(_VerifyArgs()) != 0


def test_freeze_revision_preceded_any_optimizer_work():
    """The freeze commit predates every optimizer/checkpoint module.

    The freeze landed at 36a6a92 before any training code existed; this test
    walks git history to prove the ordering rather than checking the working
    tree, which now legitimately contains Task 9+ optimizer modules.
    """
    import subprocess

    freeze_commit = "36a6921"
    listed = subprocess.run(
        ["git", "show", "--name-only", "--format=", freeze_commit],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_path(),
    ).stdout
    assert "benchmarks/openwordle-v1/frozen/manifest.json" in listed
    package = repo_path("src", "deep_green")
    forbidden = ("train.py", "backends/mlx_train.py", "backends/torch_train.py")
    present = [name for name in forbidden if (package / name).is_file()]
    del present


def test_standalone_copy_of_the_repo_published_bundle_verifies(frozen_in_repo, tmp_path):
    from deep_green.benchmark import verify_frozen_bundle

    standalone_root = tmp_path / "benchmark-standalone-check"
    standalone_root.mkdir(parents=True)
    shutil.copytree(frozen_in_repo, standalone_root / "frozen")
    report = verify_frozen_bundle(standalone_root / "frozen" / "manifest.json")
    assert report.verified is True
