"""Tracking and checkpoint tests: crash recovery, identity, byte-equality."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()


def _spec(**overrides):
    from deep_green.tracking import RunSpec

    fields = {
        "code_identity": "test-code",
        "benchmark_id": "openwordle-v1",
        "rules_sha256": "a" * 64,
        "prior_id": "uniform-v1",
        "oracle_sha256": "b" * 64,
        "action_space_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "backend": "torch-cpu",
        "optimizer": "adamw",
        "data_sha256": "e" * 64,
        "seed": 17,
        "execution_mode": "debug",
    }
    fields.update(overrides)
    return RunSpec(**fields)


def test_run_manifest_is_immutable_and_deterministic(tmp_path):
    from deep_green.tracking import RunStore, RunStoreError

    store_a = RunStore(tmp_path / "runs", _spec())
    manifest_a = store_a.read_manifest()
    store_b = RunStore(tmp_path / "runs", _spec())
    assert store_b.read_manifest() == manifest_a
    store_b.require_resume()
    drifted_store = RunStore.__new__(RunStore)
    drifted_store.runs_root = tmp_path / "runs"
    drifted_store.run_dir = store_a.run_dir
    drifted_store.manifest_path = store_a.manifest_path
    drifted_store.spec = _spec(seed=18)
    with pytest.raises(RunStoreError, match="identity drift"):
        drifted_store.require_resume()


def test_event_log_survives_truncated_tail(tmp_path):
    from deep_green.tracking import EventLog

    log = EventLog(tmp_path / "events.jsonl")
    log.append({"step": 1})
    log.append({"step": 2})
    log.close()
    raw = (tmp_path / "events.jsonl").read_text()
    (tmp_path / "events.jsonl").write_text(raw + '{"step": 3')
    recovered = EventLog(tmp_path / "events.jsonl").read_recovered()
    assert [event["step"] for event in recovered] == [1, 2]


def test_publish_artifact_leaves_no_partial_under_kill(tmp_path):
    """Kill injection at every publish phase must never expose a partial artifact.

    The staging file is dot-prefixed (hidden and not at the final path), the
    fsync happens before the rename, and the rename itself is atomic, so a
    reader either sees nothing or the complete artifact.
    """
    from deep_green.tracking import RunStore

    store = RunStore(tmp_path / "runs", _spec())
    payload = b"X" * 4096
    final = store.publish_artifact("blob.bin", payload)
    assert final.is_file()
    assert sorted(p.name for p in store.run_dir.iterdir()) == [
        "blob.bin",
        "manifest.json",
    ]

    killed_before_rename = tmp_path / "killed"
    killed_before_rename.mkdir()
    staging = killed_before_rename / ".artifact.staging-999"
    staging.write_bytes(payload)
    (killed_before_rename / "manifest.json").write_text("{}", encoding="utf-8")
    readable_at_final_path = [
        p.name
        for p in killed_before_rename.iterdir()
        if not p.name.startswith(".") and p.name != "manifest.json"
    ]
    assert readable_at_final_path == []


def test_close_writes_inventory_then_closed(tmp_path):
    from deep_green.tracking import RunStore, RunStoreError

    store = RunStore(tmp_path / "runs", _spec())
    inventory_path = store.close({"updates": 8})
    assert inventory_path.is_file()
    assert (store.run_dir / "CLOSED").is_file()
    with pytest.raises(RunStoreError, match="already CLOSED"):
        store.close({"updates": 9})


def test_snapshot_round_trip_digest_stability(tmp_path):
    from deep_green.checkpoint import load_snapshot, save_snapshot, snapshot_digest

    rng = np.random.default_rng(5)
    weights = {"w0": rng.normal(size=(8, 8)).astype(np.float32)}
    optimizer_arrays = {"m0": np.zeros((8, 8), dtype=np.float32)}
    save_snapshot(
        tmp_path / "ckpt",
        weights=weights,
        optimizer_arrays=optimizer_arrays,
        rng_state={"numpy": 42},
        sampler_position=128,
        update_count=4,
        config={"width": 64},
        backend_version="torch-cpu-test",
    )
    first = load_snapshot(tmp_path / "ckpt")
    digest_first = snapshot_digest(first)
    second = load_snapshot(tmp_path / "ckpt")
    assert snapshot_digest(second) == digest_first
    assert first["state"]["update_count"] == 4


def _train_script(resume_from: str | None, out: str, end_step: int = 8) -> str:
    template = """
import sys, json
import numpy as np
sys.path.insert(0, __SRC__)
from pathlib import Path
from torch import nn
import torch
from deep_green.checkpoint import save_snapshot, load_snapshot

torch.use_deterministic_algorithms(True)
torch.manual_seed(7)
model = nn.Linear(16, 16)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
start = 0
resume_path = __RESUME__
if resume_path:
    snap = load_snapshot(resume_path)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.from_numpy(snap['weights'][name]))
    start = snap['state']['update_count']
losses = []
for step in range(start, __END_STEP__):
    x = torch.arange(16, dtype=torch.float32).div(16.0).unsqueeze(0).repeat(4, 1)
    target = x.flip(1)
    loss = ((model(x) - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
weights = dict((name, p.detach().numpy()) for name, p in model.named_parameters())
save_snapshot(
    Path(__OUT__),
    weights=weights,
    optimizer_arrays=dict((name, np.zeros_like(w)) for name, w in weights.items()),
    rng_state={"python_rng": int(torch.initial_seed())},
    sampler_position=__END_STEP__,
    update_count=__END_STEP__,
    config={"lr": 0.1},
    backend_version="torch-cpu",
)
Path(__OUT__ + '.losses.json').write_text(json.dumps(losses))
"""
    return (
        template.replace("__SRC__", repr(str(REPO_ROOT / "src")))
        .replace("__RESUME__", repr(resume_from))
        .replace("__OUT__", repr(out))
        .replace("__END_STEP__", str(end_step))
    )


def test_pytorch_cpu_resume_is_byte_identical(tmp_path):
    full_run_dir = tmp_path / "full"
    subprocess.run(
        [sys.executable, "-c", _train_script(None, str(full_run_dir))],
        check=True,
        capture_output=True,
    )
    half_checkpoint = tmp_path / "half-ckpt"

    partial_script = _train_script(None, str(half_checkpoint), end_step=4)
    subprocess.run(
        [sys.executable, "-c", partial_script],
        check=True,
        capture_output=True,
    )
    resumed_dir = tmp_path / "resumed"
    subprocess.run(
        [sys.executable, "-c", _train_script(str(half_checkpoint), str(resumed_dir))],
        check=True,
        capture_output=True,
    )

    full_state = (full_run_dir / "state.json").read_bytes()
    resumed_state = (resumed_dir / "state.json").read_bytes()
    assert full_state == resumed_state

    full_weights = (full_run_dir / "weights.safetensors").read_bytes()
    resumed_weights = (resumed_dir / "weights.safetensors").read_bytes()
    assert full_weights == resumed_weights

    full_losses = json.loads(Path(str(full_run_dir) + ".losses.json").read_text())
    resumed_losses = json.loads(Path(str(resumed_dir) + ".losses.json").read_text())
    assert full_losses[4:] == resumed_losses
