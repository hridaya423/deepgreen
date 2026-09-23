"""Deterministic index-modulo shards with strict merge validation."""

from __future__ import annotations

import json
from dataclasses import dataclass


class ShardError(ValueError):
    pass


@dataclass(frozen=True)
class ShardManifest:
    shard_index: int
    shard_count: int
    identity_sha256: str
    action_words: tuple[str, ...]
    exact: dict[str, int]
    bounds: dict[str, int]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "shard_index": self.shard_index,
                    "shard_count": self.shard_count,
                    "identity_sha256": self.identity_sha256,
                    "action_words": list(self.action_words),
                    "exact": dict(self.exact),
                    "bounds": dict(self.bounds),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )

    @staticmethod
    def from_json(text: str) -> ShardManifest:
        raw = json.loads(text)
        required = {
            "shard_index",
            "shard_count",
            "identity_sha256",
            "action_words",
            "exact",
            "bounds",
        }
        if set(raw) != required:
            raise ShardError(f"shard manifest fields malformed: {sorted(raw)}")
        return ShardManifest(
            shard_index=int(raw["shard_index"]),
            shard_count=int(raw["shard_count"]),
            identity_sha256=str(raw["identity_sha256"]),
            action_words=tuple(raw["action_words"]),
            exact={k: int(v) for k, v in raw["exact"].items()},
            bounds={k: int(v) for k, v in raw["bounds"].items()},
        )


def assign_shards(action_words: tuple[str, ...], shard_count: int) -> tuple[tuple[str, ...], ...]:
    """Action at canonical index i belongs to shard i mod shard_count."""
    if shard_count < 1:
        raise ShardError("shard_count must be positive")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, word in enumerate(action_words):
        shards[index % shard_count].append(word)
    return tuple(tuple(shard) for shard in shards)


def merge_shards(manifests: list[ShardManifest]) -> dict:
    """Merge only when shards are disjoint, complete, and identity-matched.

    Every top-level action must carry an exact value or an independently
    recorded lower bound; a bound alone never masquerades as an exact value.
    """
    if not manifests:
        raise ShardError("no shards to merge")
    first = manifests[0]
    for manifest in manifests[1:]:
        if manifest.identity_sha256 != first.identity_sha256:
            raise ShardError("shard identity mismatch across the merge set")
        if manifest.shard_count != first.shard_count:
            raise ShardError("shard count mismatch across the merge set")
    indexes = sorted(m.shard_index for m in manifests)
    if indexes != list(range(first.shard_count)):
        raise ShardError(f"shard set incomplete: present indexes {indexes}")
    all_actions_by_shard: dict[int, tuple[str, ...]] = {
        m.shard_index: m.action_words for m in manifests
    }
    seen: dict[str, int] = {}
    for shard_index, words in all_actions_by_shard.items():
        for word in words:
            if word in seen:
                raise ShardError(f"action {word!r} claimed by multiple shards")
            seen[word] = shard_index
    expected = {m.shard_index: set(m.action_words) for m in manifests}
    exact: dict[str, int] = {}
    bounds: dict[str, int] = {}
    for manifest in manifests:
        allowed = expected[manifest.shard_index]
        outside_exact = set(manifest.exact) - allowed
        outside_bounds = set(manifest.bounds) - allowed
        if outside_exact or outside_bounds:
            raise ShardError(
                f"shard {manifest.shard_index} records actions outside its assignment"
            )
        overlap = set(manifest.exact) & set(manifest.bounds)
        if overlap:
            raise ShardError(
                f"shard {manifest.shard_index} records both bound and exact for {sorted(overlap)[:4]}"
            )
        exact.update(manifest.exact)
        bounds.update(manifest.bounds)
    covered = set(exact) | set(bounds)
    all_actions: set[str] = set(seen)
    uncovered = all_actions - covered
    if uncovered:
        raise ShardError(
            f"{len(uncovered)} top-level actions lack both exact values and bounds"
        )
    return {
        "identity_sha256": first.identity_sha256,
        "exact": exact,
        "bounds": bounds,
        "bound_only": sorted(set(bounds)),
        "complete": not bounds,
    }
