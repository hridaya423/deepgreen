"""End-to-end smoke: tiny exhaustive cell replays identically in both modes."""

from __future__ import annotations

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
SMOKE_ACTIONS = ("audio", "crane", "pouty", "slate")
SMOKE_ANSWERS = ("slate", "pouty")


@pytest.fixture(scope="module")
def solver_binary(tmp_path_factory):
    from deep_green.oracle.build import ensure_solver, vendor_dir

    cache = tmp_path_factory.mktemp("solver-build")
    exe_path, digest, compiler_id = ensure_solver(vendor_dir(REPO_ROOT), cache)
    return exe_path, digest, compiler_id


def test_smoke_normal_mode(tmp_path):
    from deep_green.oracle.smoke import run_smoke

    report = run_smoke(repo_root=REPO_ROOT, work_dir=tmp_path, mode="normal")
    assert report.solver.objective == report.python_objective
    assert report.solver.best_first_word in report.optimal_first_words
    assert report.solver.objective >= len(SMOKE_ANSWERS)


def test_smoke_hard_mode(tmp_path):
    from deep_green.oracle.smoke import run_smoke

    report = run_smoke(repo_root=REPO_ROOT, work_dir=tmp_path, mode="hard")
    assert report.solver.objective == report.python_objective
    assert report.solver.best_first_word in report.optimal_first_words


def test_smoke_modes_agree_on_this_tiny_cell(tmp_path):
    from deep_green.oracle.smoke import run_smoke

    normal = run_smoke(repo_root=REPO_ROOT, work_dir=tmp_path / "n", mode="normal")
    hard = run_smoke(repo_root=REPO_ROOT, work_dir=tmp_path / "h", mode="hard")
    assert normal.solver.objective == hard.solver.objective


def test_executable_hash_binding_rejects_drift(tmp_path):
    """A request pinned to the wrong executable hash must fail closed."""
    import hashlib

    from deep_green.oracle.adapter import (
        OracleExecutionError,
        OracleRequest,
        run_oracle,
    )
    from deep_green.oracle.build import ensure_solver, vendor_dir
    from deep_green.rules import ActionSpace

    _, binary_digest, compiler_id = ensure_solver(
        vendor_dir(REPO_ROOT), tmp_path / "build"
    )
    action_space = ActionSpace.parse(SMOKE_ACTIONS)
    request = OracleRequest(
        mode="normal",
        horizon=6,
        rules_sha256="a" * 64,
        prior_sha256="b" * 64,
        action_space_sha256=action_space.digest,
        answer_set_sha256=hashlib.sha256("\n".join(SMOKE_ANSWERS).encode("utf-8")).hexdigest(),
        executable_sha256=binary_digest,
        compiler_id=compiler_id,
        solver_source_sha256="f" * 64,
        action_words=action_space.words,
        answer_words=SMOKE_ANSWERS,
    )
    drifted = OracleRequest(
        **{
            **{f: getattr(request, f) for f in request.__dataclass_fields__},
            "executable_sha256": "0" * 64,
        }
    )
    with pytest.raises(OracleExecutionError, match="drift"):
        run_oracle(drifted, repo_root=REPO_ROOT, work_dir=tmp_path / "runs")


def test_compiler_identity_binding_rejects_drift(tmp_path):
    """A request pinned to the wrong compiler identity must fail closed."""
    import hashlib

    from deep_green.oracle.adapter import (
        OracleExecutionError,
        OracleRequest,
        run_oracle,
    )
    from deep_green.oracle.build import ensure_solver, vendor_dir
    from deep_green.rules import ActionSpace

    _, binary_digest, _ = ensure_solver(vendor_dir(REPO_ROOT), tmp_path / "build")
    action_space = ActionSpace.parse(SMOKE_ACTIONS)
    request = OracleRequest(
        mode="normal",
        horizon=6,
        rules_sha256="a" * 64,
        prior_sha256="b" * 64,
        action_space_sha256=action_space.digest,
        answer_set_sha256=hashlib.sha256("\n".join(SMOKE_ANSWERS).encode("utf-8")).hexdigest(),
        executable_sha256=binary_digest,
        compiler_id="definitely-not-the-build-compiler",
        solver_source_sha256="f" * 64,
        action_words=action_space.words,
        answer_words=SMOKE_ANSWERS,
    )
    with pytest.raises(OracleExecutionError, match="compiler"):
        run_oracle(request, repo_root=REPO_ROOT, work_dir=tmp_path / "runs")
