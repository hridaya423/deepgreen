"""Long-run campaign driver: checkpointed tier searches with budget reporting."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from deep_green.oracle.checkpoints import CheckpointWriter


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Clock:
    """Injectable clock so the 72 h boundary can be rehearsed in tests."""

    start_monotonic: float | None = None

    def elapsed(self) -> float:
        if self.start_monotonic is None:
            return time.monotonic()
        return time.monotonic() - self.start_monotonic


class FakeClock(Clock):
    def __init__(self) -> None:
        super().__init__(start_monotonic=0.0)
        self._now = 0.0

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def elapsed(self) -> float:
        return self._now


def run_campaign(
    *,
    exe_path: Path,
    work_dir: Path,
    action_words_path: Path,
    answers_path: Path,
    mode: str,
    tier: int,
    budget_seconds: float,
    clock: Clock,
) -> dict:
    """Run one tier search under a hard local budget.

    At the budget the campaign checkpoints its state and reports INCOMPLETE;
    a partial result is always labeled an upper bound and never renamed to a
    complete tier.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    events_jsonl = work_dir / f"tier-{tier}-{mode}.jsonl"
    log_path = work_dir / f"tier-{tier}-{mode}.log"
    command = [
        str(exe_path),
        "-a",
        str(action_words_path),
        "-h",
        str(answers_path),
        "-E",
        str(events_jsonl),
    ]
    if mode == "hard":
        command.append("-H")
    started = clock.elapsed()
    log_handle = open(log_path, "w")  # noqa: SIM115 - lifetime spans the process poll loop
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        while True:
            if process.poll() is not None:
                break
            if clock.elapsed() - started >= budget_seconds:
                process.kill()
                process.wait()
                report = {
                    "schema": "wordle.oracle-campaign/v1",
                    "tier": tier,
                    "mode": mode,
                    "status": "INCOMPLETE",
                    "reason": "local-time-budget-exhausted",
                    "budget_seconds": budget_seconds,
                    "elapsed_seconds": clock.elapsed() - started,
                    "events_file": str(events_jsonl),
                    "result_class": "upper-bound-only",
                }
                writer = CheckpointWriter(work_dir / f"checkpoints-tier-{tier}-{mode}")
                writer.publish(report)
                (work_dir / f"tier-{tier}-{mode}.report.json").write_text(
                    json.dumps(report, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return report
            time.sleep(0.05)
    except BaseException:
        process.kill()
        raise
    finally:
        log_handle.close()
    if process.returncode != 0:
        raise CampaignError(f"solver exited {process.returncode}; see {log_path}")
    events = [
        json.loads(line)
        for line in events_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity = next(e for e in events if e["type"] == "run_identity")
    result = next(e for e in events if e["type"] == "result")
    completion = next(e for e in events if e["type"] == "completion")
    if completion["status"] != "completed" or not identity["exhaustive"]:
        raise CampaignError("solver did not complete exhaustively")
    if identity["schema"] != "alex1770.wordle-events/v1":
        raise CampaignError("unexpected event schema")
    report = {
        "schema": "wordle.oracle-campaign/v1",
        "tier": tier,
        "mode": mode,
        "status": "COMPLETE",
        "objective": int(result["objective"]),
        "average": float(result["average"]),
        "nodes": int(result["nodes"]),
        "time_seconds": float(result["time_seconds"]),
        "elapsed_seconds": clock.elapsed() - started,
        "exhaustive": True,
    }
    (work_dir / f"tier-{tier}-{mode}.report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
