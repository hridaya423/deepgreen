"""Checkpoint and shard tests: crash safety and strict merges."""

from __future__ import annotations

import pytest

from deep_green.oracle.checkpoints import (
    CheckpointError,
    CheckpointWriter,
    read_latest_checkpoint,
)
from deep_green.oracle.shards import (
    ShardError,
    ShardManifest,
    assign_shards,
    merge_shards,
)


def test_publish_and_read_back(tmp_path):
    writer = CheckpointWriter(tmp_path)
    handle = writer.publish({"state": "one"})
    assert handle.generation == 0
    generation, payload = read_latest_checkpoint(tmp_path)
    assert generation == 0
    assert payload["state"] == "one"
    handle2 = writer.publish({"state": "two"})
    generation, payload = read_latest_checkpoint(tmp_path)
    assert generation == handle2.generation == 1
    assert payload["state"] == "two"


def test_truncated_newest_generation_falls_back(tmp_path):
    writer = CheckpointWriter(tmp_path)
    writer.publish({"state": "one"})
    writer.publish({"state": "two"})
    newest = tmp_path / "gen-000001"
    for child in newest.iterdir():
        if child.is_file():
            child.write_text("{ truncated", encoding="utf-8")
    generation, payload = read_latest_checkpoint(tmp_path)
    assert generation == 0
    assert payload["state"] == "one"


def test_failure_before_rename_leaves_previous_complete(tmp_path):
    """A killed writer mid-publish must not corrupt the previous generation."""
    writer = CheckpointWriter(tmp_path)
    writer.publish({"state": "good"})
    staging = tmp_path / ".staging-1"
    staging.mkdir()
    (staging / "payload.json").write_text('{"state": "half"}', encoding="utf-8")
    generation, payload = read_latest_checkpoint(tmp_path)
    assert generation == 0
    assert payload["state"] == "good"


def test_pointer_tampering_still_recovers(tmp_path):
    writer = CheckpointWriter(tmp_path)
    writer.publish({"state": "one"})
    writer.publish({"state": "two"})
    (tmp_path / "CURRENT").write_text("gen-999999\n", encoding="utf-8")
    generation, payload = read_latest_checkpoint(tmp_path)
    assert generation in (0, 1)
    assert payload["state"] in ("one", "two")


def test_empty_checkpoint_root_raises(tmp_path):
    with pytest.raises(CheckpointError):
        read_latest_checkpoint(tmp_path)


def _shard(index, count, actions, exact=None, bounds=None):
    return ShardManifest(
        shard_index=index,
        shard_count=count,
        identity_sha256="a" * 64,
        action_words=tuple(actions),
        exact=dict(exact or {}),
        bounds=dict(bounds or {}),
    )


def test_assign_shards_is_index_modulo():
    shards = assign_shards(("a", "b", "c", "d", "e"), 2)
    assert shards[0] == ("a", "c", "e")
    assert shards[1] == ("b", "d")


def test_merge_disjoint_complete_shards():
    shards = assign_shards(("a", "b", "c", "d"), 2)
    left = _shard(0, 2, shards[0], exact={"a": 10}, bounds={"c": 12})
    right = _shard(1, 2, shards[1], exact={"b": 11, "d": 9})
    merged = merge_shards([left, right])
    assert merged["complete"] is False
    assert merged["exact"] == {"a": 10, "b": 11, "d": 9}
    assert merged["bound_only"] == ["c"]


def test_merge_rejects_identity_mismatch():
    left = _shard(0, 2, ("a",), exact={"a": 1})
    right = ShardManifest(
        shard_index=1,
        shard_count=2,
        identity_sha256="b" * 64,
        action_words=("c",),
        exact={"c": 2},
        bounds={},
    )
    with pytest.raises(ShardError, match="identity"):
        merge_shards([left, right])


def test_merge_rejects_incomplete_set():
    left = _shard(0, 3, ("a",), exact={"a": 1})
    middle = _shard(1, 3, ("b",), exact={"b": 2})
    with pytest.raises(ShardError, match="incomplete"):
        merge_shards([left, middle])


def test_merge_rejects_overlapping_actions():
    left = _shard(0, 2, ("a", "b"), exact={"a": 1})
    right = _shard(1, 2, ("a",), exact={"a": 2})
    with pytest.raises(ShardError, match="claimed by multiple shards"):
        merge_shards([left, right])


def test_merge_rejects_uncovered_action():
    left = _shard(0, 2, ("a", "ghost"), exact={"a": 1})
    right = _shard(1, 2, ("c",), exact={"c": 2})
    with pytest.raises(ShardError, match="lack both"):
        merge_shards([left, right])


def test_merge_rejects_out_of_shard_records():
    left = _shard(0, 2, ("a",), exact={"a": 1, "c": 5})
    right = _shard(1, 2, ("c",), exact={"c": 2})
    with pytest.raises(ShardError, match="outside its assignment"):
        merge_shards([left, right])


def test_manifest_round_trip():
    manifest = _shard(1, 4, ("x", "y"), exact={"x": 3}, bounds={"y": 5})
    restored = ShardManifest.from_json(manifest.to_json())
    assert restored == manifest
