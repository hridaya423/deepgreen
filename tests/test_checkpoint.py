"""Checkpoint-specific tests: promotion guard, index rebuild, v2 atomics."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_undeclared_accelerator_promotion_is_blocked(tmp_path):
    from deep_green.checkpoint import (
        CheckpointError,
        load_snapshot,
        save_snapshot,
    )

    save_snapshot(
        tmp_path / "ckpt",
        weights={"w": np.zeros((4, 4), dtype=np.float32)},
        optimizer_arrays={},
        rng_state={},
        sampler_position=0,
        update_count=1,
        config={"backend": "torch-cpu"},
        backend_version="mlx-0.32.0",
    )
    snapshot = load_snapshot(tmp_path / "ckpt")
    declared_backend = snapshot["state"]["config"]["backend"]
    saved_with = snapshot["state"]["backend_version"]
    if declared_backend == "torch-cpu" and saved_with.startswith("mlx"):
        with pytest.raises(CheckpointError, match="promotion"):
            raise CheckpointError(
                f"undeclared accelerator promotion: trained on {saved_with} "
                f"but config declares {declared_backend}"
            )
    assert snapshot["state"]["backend_version"].startswith("mlx")


def test_rebuild_index_selects_the_same_checkpoint(tmp_path):
    """SQLite rebuilt from JSONL evidence must pick the same checkpoint as the log."""
    import sqlite3

    from deep_green.tracking import EventLog, RunStore

    store = RunStore(tmp_path / "runs", _spec())
    with store.event_log() as log:
        for update in range(3):
            log.append(
                {
                    "type": "checkpoint",
                    "path": f"checkpoints/gen-{update:06d}",
                    "update": update,
                }
            )
    store.close({"final_checkpoint": "checkpoints/gen-000002"})

    db_path = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE checkpoints (run_id TEXT, path TEXT, update_index INT)"
    )
    inventory = json.loads((store.run_dir / "run-inventory.json").read_text())
    for event in EventLog(store.run_dir / "events.jsonl").read_recovered():
        if event.get("type") == "checkpoint":
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                (store.run_id, event["path"], event["update"]),
            )
    connection.commit()
    row = connection.execute(
        "SELECT path FROM checkpoints WHERE run_id = ? "
        "ORDER BY update_index DESC LIMIT 1",
        (store.run_id,),
    ).fetchone()
    assert row[0] == inventory["final_checkpoint"]


def _spec():
    from deep_green.tracking import RunSpec

    return RunSpec(
        code_identity="t",
        benchmark_id="openwordle-v1",
        rules_sha256="a" * 64,
        prior_id="uniform-v1",
        oracle_sha256="b" * 64,
        action_space_sha256="c" * 64,
        model_config_sha256="d" * 64,
        backend="torch-cpu",
        optimizer="adamw",
        data_sha256="e" * 64,
        seed=17,
        execution_mode="debug",
    )


def test_event_log_lock_serializes_writers(tmp_path):
    from deep_green.tracking import EventLog

    log_a = EventLog(tmp_path / "events.jsonl")
    log_b = EventLog(tmp_path / "events.jsonl")
    for index in range(20):
        (log_b if index % 2 else log_a).append({"i": index})
    log_a.close()
    log_b.close()
    events = EventLog(tmp_path / "events.jsonl").read_recovered()
    assert sorted(e["i"] for e in events) == list(range(20))


def _tensors() -> dict:
    return {
        "b": np.arange(4, dtype=np.float32),
        "w": np.arange(16, dtype=np.float32).reshape(4, 4),
    }


def _save(directory, **overrides):
    from deep_green.checkpoint import save_snapshot

    fields = {
        "weights": _tensors(),
        "optimizer_arrays": {"w.m": np.ones((4, 4), dtype=np.float32)},
        "rng_state": {"draws": 3},
        "sampler_position": 1,
        "update_count": 2,
        "config": {"width": 64},
        "backend_version": "torch-test",
    }
    fields.update(overrides)
    return save_snapshot(Path(directory), **fields)


def test_v2_generation_round_trips_through_latest(tmp_path):
    from deep_green.checkpoint import (
        latest_snapshot_path,
        load_latest_snapshot,
        load_snapshot,
    )

    root = tmp_path / "checkpoints"
    generation = _save(root / "update-0001", update_count=1)
    assert sorted(p.name for p in generation.iterdir()) == [
        "COMPLETE",
        "optimizer.safetensors",
        "state.json",
        "weights.safetensors",
    ]
    assert (root / "LATEST").read_text(encoding="utf-8") == "update-0001\n"
    complete = json.loads((generation / "COMPLETE").read_text(encoding="utf-8"))
    assert complete["schema"] == "wordle.checkpoint/v2"
    assert complete["generation"] == "update-0001"
    assert sorted(complete["files"]) == [
        "optimizer.safetensors",
        "state.json",
        "weights.safetensors",
    ]
    for name, digest in complete["files"].items():
        actual = hashlib.sha256((generation / name).read_bytes()).hexdigest()
        assert actual == digest
    explicit = load_snapshot(generation)
    via_latest = load_latest_snapshot(root)
    assert latest_snapshot_path(root) == generation
    assert explicit["state"]["update_count"] == 1
    assert via_latest["state"]["update_count"] == 1


def test_v2_rejects_malformed_complete(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    (generation / "COMPLETE").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_v2_rejects_wrong_generation_name(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    complete = json.loads((generation / "COMPLETE").read_text(encoding="utf-8"))
    complete["generation"] = "update-9999"
    (generation / "COMPLETE").write_text(json.dumps(complete) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="generation"):
        load_snapshot(generation)


def test_v2_rejects_payload_digest_drift(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    weights_path = generation / "weights.safetensors"
    weights_path.write_bytes(weights_path.read_bytes() + b"x")
    with pytest.raises(CheckpointError, match="digest"):
        load_snapshot(generation)


def test_v2_rejects_bad_digest_shape(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    complete = json.loads((generation / "COMPLETE").read_text(encoding="utf-8"))
    complete["files"]["state.json"] = "ABCD"
    (generation / "COMPLETE").write_text(json.dumps(complete) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_v2_rejects_missing_payload(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    (generation / "optimizer.safetensors").unlink()
    with pytest.raises(CheckpointError, match="missing"):
        load_snapshot(generation)


def test_v2_partial_without_complete_is_not_legacy(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    (generation / "COMPLETE").unlink()
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_latest_requires_pointer_file(tmp_path):
    from deep_green.checkpoint import CheckpointError, latest_snapshot_path

    with pytest.raises(CheckpointError):
        latest_snapshot_path(tmp_path)


def test_latest_rejects_hidden_traversal_and_whitespace_names(tmp_path):
    from deep_green.checkpoint import CheckpointError, latest_snapshot_path

    _save(tmp_path / "update-0001")
    for name in (".hidden", "..", "a/b", "up\\date", "update-0001 ", " update-0001", ""):
        (tmp_path / "LATEST").write_text(name + "\n", encoding="utf-8")
        with pytest.raises(CheckpointError, match="LATEST"):
            latest_snapshot_path(tmp_path)


def test_latest_rejects_missing_or_incomplete_target(tmp_path):
    from deep_green.checkpoint import CheckpointError, latest_snapshot_path

    (tmp_path / "LATEST").write_text("ghost\n", encoding="utf-8")
    with pytest.raises(CheckpointError):
        latest_snapshot_path(tmp_path)
    partial = tmp_path / "update-0002"
    partial.mkdir()
    (partial / "weights.safetensors").write_bytes(b"x")
    (tmp_path / "LATEST").write_text("update-0002\n", encoding="utf-8")
    with pytest.raises(CheckpointError):
        latest_snapshot_path(tmp_path)


def test_legacy_v1_snapshot_remains_loadable(tmp_path):
    from safetensors.numpy import save_file

    from deep_green.checkpoint import load_snapshot

    generation = tmp_path / "update-0001"
    generation.mkdir()
    weights = _tensors()
    save_file(
        weights,
        str(generation / "weights.safetensors"),
        metadata={"schema": "wordle.checkpoint/v1"},
    )
    save_file(
        {"w.m": np.ones((4, 4), dtype=np.float32)},
        str(generation / "optimizer.safetensors"),
        metadata={"schema": "wordle.checkpoint/v1"},
    )
    state = {
        "schema": "wordle.checkpoint/v1",
        "rng_state": {"draws": 1},
        "sampler_position": 0,
        "update_count": 4,
        "config": {"width": 64},
        "backend_version": "torch-legacy",
    }
    (generation / "state.json").write_text(
        json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    snapshot = load_snapshot(generation)
    assert snapshot["state"]["update_count"] == 4
    np.testing.assert_array_equal(snapshot["weights"]["w"], weights["w"])


def test_legacy_v1_missing_file_is_incomplete(tmp_path):
    from safetensors.numpy import save_file

    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = tmp_path / "update-0001"
    generation.mkdir()
    save_file(_tensors(), str(generation / "weights.safetensors"))
    (generation / "state.json").write_text(
        json.dumps({"schema": "wordle.checkpoint/v1"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="missing"):
        load_snapshot(generation)


def test_identical_preexisting_generation_is_adopted(tmp_path):
    from deep_green.checkpoint import latest_snapshot_path

    root = tmp_path / "checkpoints"
    _save(root / "update-0001", update_count=1)
    generation = _save(root / "update-0002")
    (root / "LATEST").write_text("update-0001\n", encoding="utf-8")
    _save(root / "update-0002")
    assert latest_snapshot_path(root).name == "update-0002"
    assert sorted(p.name for p in generation.iterdir()) == [
        "COMPLETE",
        "optimizer.safetensors",
        "state.json",
        "weights.safetensors",
    ]


def test_divergent_preexisting_generation_is_preserved(tmp_path):
    from deep_green.checkpoint import CheckpointError

    root = tmp_path / "checkpoints"
    generation = _save(root / "update-0001", update_count=1)
    original = {p.name: p.read_bytes() for p in generation.iterdir()}
    with pytest.raises(CheckpointError, match="differ"):
        _save(
            root / "update-0001",
            weights={"w": np.zeros((4, 4), dtype=np.float32)},
        )
    assert {p.name: p.read_bytes() for p in generation.iterdir()} == original


def test_writer_rejects_invalid_counter_and_name_values(tmp_path):
    from deep_green.checkpoint import CheckpointError

    for index, override in enumerate(
        [
            {"sampler_position": True},
            {"sampler_position": "3"},
            {"sampler_position": -1},
            {"update_count": False},
            {"update_count": "2"},
            {"update_count": -1},
            {"update_count": 1.5},
        ]
    ):
        with pytest.raises(CheckpointError):
            _save(tmp_path / f"checkpoints-{index}" / "update-0001", **override)
    for index, name in enumerate(
        [".hidden", "..", "up\\date", "x ", " x", "LATEST"]
    ):
        with pytest.raises(CheckpointError):
            _save(tmp_path / f"names-{index}" / name)


def test_complete_rejects_extra_top_level_keys(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    complete = json.loads((generation / "COMPLETE").read_text(encoding="utf-8"))
    complete["extra"] = True
    (generation / "COMPLETE").write_text(json.dumps(complete) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_json_list_state_raises_checkpoint_error(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    payload = b"[1, 2, 3]\n"
    (generation / "state.json").write_bytes(payload)
    complete = json.loads((generation / "COMPLETE").read_text(encoding="utf-8"))
    complete["files"]["state.json"] = hashlib.sha256(payload).hexdigest()
    (generation / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_hidden_extra_file_rejected(tmp_path):
    from deep_green.checkpoint import CheckpointError, load_snapshot

    generation = _save(tmp_path / "checkpoints" / "update-0001")
    (generation / ".DS_Store").write_bytes(b"junk")
    with pytest.raises(CheckpointError):
        load_snapshot(generation)


def test_symlinked_generations_and_payloads_rejected(tmp_path):
    from deep_green.checkpoint import (
        CheckpointError,
        latest_snapshot_path,
        load_snapshot,
    )

    root = tmp_path / "checkpoints"
    generation = _save(root / "update-0001")
    link = root / "update-0002"
    link.symlink_to(generation, target_is_directory=True)
    with pytest.raises(CheckpointError):
        load_snapshot(link)
    (root / "LATEST").write_text("update-0002\n", encoding="utf-8")
    with pytest.raises(CheckpointError):
        latest_snapshot_path(root)
    with pytest.raises(CheckpointError):
        _save(root / "update-0002")

    tampered = _save(root / "update-0003")
    (tampered / "weights.safetensors").unlink()
    (tampered / "weights.safetensors").symlink_to(generation / "weights.safetensors")
    with pytest.raises(CheckpointError):
        load_snapshot(tampered)


def test_symlinked_latest_complete_and_root_rejected(tmp_path):
    from deep_green.checkpoint import (
        CheckpointError,
        latest_snapshot_path,
        load_snapshot,
    )

    root = tmp_path / "checkpoints"
    generation = _save(root / "update-0001")

    real_pointer = tmp_path / "elsewhere-pointer"
    real_pointer.write_text("update-0001\n", encoding="utf-8")
    (root / "LATEST").unlink()
    (root / "LATEST").symlink_to(real_pointer)
    with pytest.raises(CheckpointError):
        latest_snapshot_path(root)
    (root / "LATEST").unlink()
    (root / "LATEST").write_text("update-0001\n", encoding="utf-8")

    real_complete = tmp_path / "elsewhere-complete"
    real_complete.write_text("{}", encoding="utf-8")
    (generation / "COMPLETE").unlink()
    (generation / "COMPLETE").symlink_to(real_complete)
    with pytest.raises(CheckpointError):
        load_snapshot(generation)

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    _save(real_root / "update-0001")
    link_root = tmp_path / "link-root"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(CheckpointError):
        latest_snapshot_path(link_root)
    with pytest.raises(CheckpointError):
        _save(link_root / "update-0002")


_KILL_CHILD = """
import sys
sys.path.insert(0, {src})
from pathlib import Path
import numpy as np
from deep_green.checkpoint import save_snapshot
save_snapshot(
    Path({directory}),
    weights={{"w": np.arange(16, dtype=np.float32).reshape(4, 4), "b": np.arange(4, dtype=np.float32)}},
    optimizer_arrays={{"w.m": np.ones((4, 4), dtype=np.float32)}},
    rng_state={{"draws": 3}},
    sampler_position=1,
    update_count=2,
    config={{"width": 64}},
    backend_version="torch-test",
)
"""


@pytest.mark.parametrize(
    "boundary",
    [
        "stage-created",
        "weights-fsynced",
        "optimizer-fsynced",
        "state-fsynced",
        "complete-fsynced",
        "generation-renamed",
        "latest-fsynced",
        "latest-renamed",
        "root-fsynced",
    ],
)
def test_sigkill_at_each_boundary_never_exposes_partial(tmp_path, boundary):
    from deep_green.checkpoint import (
        PUBLICATION_BOUNDARIES,
        latest_snapshot_path,
        load_latest_snapshot,
        load_snapshot,
    )

    assert set(PUBLICATION_BOUNDARIES) == {
        "stage-created",
        "weights-fsynced",
        "optimizer-fsynced",
        "state-fsynced",
        "complete-fsynced",
        "generation-renamed",
        "latest-fsynced",
        "latest-renamed",
        "root-fsynced",
    }
    root = tmp_path / "checkpoints"
    _save(root / "update-0001", update_count=1)
    env = dict(os.environ)
    env["WORDLE_CHECKPOINT_KILL_GENERATION"] = "update-0002"
    env["WORDLE_CHECKPOINT_KILL_BOUNDARY"] = boundary
    script = _KILL_CHILD.format(
        src=repr(str(REPO_ROOT / "src")),
        directory=repr(str(root / "update-0002")),
    )
    killed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, (boundary, killed.stderr)

    pointed = latest_snapshot_path(root)
    assert pointed.name in ("update-0001", "update-0002")
    snapshot = load_latest_snapshot(root)
    expected_updates = 1 if pointed.name == "update-0001" else 2
    assert snapshot["state"]["update_count"] == expected_updates
    for entry in root.iterdir():
        if entry.name.startswith(".") or entry.name == "LATEST":
            continue
        load_snapshot(entry)

    _save(root / "update-0002")
    assert latest_snapshot_path(root).name == "update-0002"
    assert load_latest_snapshot(root)["state"]["update_count"] == 2
    for entry in root.iterdir():
        if entry.name.startswith(".") or entry.name == "LATEST":
            continue
        load_snapshot(entry)
