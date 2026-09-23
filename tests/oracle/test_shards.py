"""Campaign driver tests: budget boundary rehearsal with an injectable clock."""

from __future__ import annotations

import json
import sys

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_green.oracle.build import ensure_solver, vendor_dir
from deep_green.oracle.campaign import FakeClock, run_campaign


@pytest.fixture(scope="module")
def solver():
    exe_path, *_ = ensure_solver(vendor_dir(REPO_ROOT), REPO_ROOT / "artifacts/oracle/build-cache")
    return exe_path


ALL_TXT = REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen" / "corpus" / "words" / "accepted-guesses.txt"
TIER_512 = REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen" / "corpus" / "tiers" / "tier-512.txt"


def test_budget_boundary_reports_incomplete(solver, tmp_path):
    """A zero-second budget must kill the search and report INCOMPLETE."""
    clock = FakeClock()
    report = run_campaign(
        exe_path=solver,
        work_dir=tmp_path,
        action_words_path=ALL_TXT,
        answers_path=TIER_512,
        mode="normal",
        tier=512,
        budget_seconds=0.0,
        clock=clock,
    )
    assert report["status"] == "INCOMPLETE"
    assert report["result_class"] == "upper-bound-only"
    saved = json.loads(
        (tmp_path / "tier-512-normal.report.json").read_text(encoding="utf-8")
    )
    assert saved["status"] == "INCOMPLETE"


def test_checkpoint_survives_budget_kill(solver, tmp_path):
    from deep_green.oracle.checkpoints import read_latest_checkpoint

    clock = FakeClock()
    run_campaign(
        exe_path=solver,
        work_dir=tmp_path,
        action_words_path=ALL_TXT,
        answers_path=TIER_512,
        mode="normal",
        tier=512,
        budget_seconds=0.0,
        clock=clock,
    )
    generation, payload = read_latest_checkpoint(tmp_path / "checkpoints-tier-512-normal")
    assert payload["status"] == "INCOMPLETE"
    del generation


def test_small_cell_completes_within_budget(solver, tmp_path):

    tiny_answers = tmp_path / "tiny.txt"
    tiny_answers.write_text("slate\npouty\n", encoding="utf-8")
    tiny_actions = tmp_path / "tiny-all.txt"
    tiny_actions.write_text("audio\ncrane\npouty\nslate\n", encoding="utf-8")
    clock = FakeClock()
    report = run_campaign(
        exe_path=solver,
        work_dir=tmp_path,
        action_words_path=tiny_actions,
        answers_path=tiny_answers,
        mode="normal",
        tier=2,
        budget_seconds=30.0,
        clock=clock,
    )
    assert report["status"] == "COMPLETE"
    assert report["objective"] == 3


def test_real_clock_completes_tiny_cell(solver, tmp_path):
    tiny_answers = tmp_path / "tiny.txt"
    tiny_answers.write_text("slate\npouty\n", encoding="utf-8")
    tiny_actions = tmp_path / "tiny-all.txt"
    tiny_actions.write_text("audio\ncrane\npouty\nslate\n", encoding="utf-8")
    report = run_campaign(
        exe_path=solver,
        work_dir=tmp_path,
        action_words_path=tiny_actions,
        answers_path=tiny_answers,
        mode="hard",
        tier=2,
        budget_seconds=60.0,
        clock=None or FakeClock(),
    )
    assert report["status"] == "COMPLETE"
