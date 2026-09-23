"""Generation-numbered crash-safe checkpoints with atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointHandle:
    root: Path
    generation: int
    directory: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_path(path: Path, *, is_dir: bool = False) -> None:
    flags = os.O_RDONLY
    if is_dir:
        flags |= getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointWriter:
    """Publish immutable generation directories plus an atomic CURRENT pointer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_generation(self) -> int:
        existing = [
            int(name.split("-")[1])
            for name in os.listdir(self.root)
            if name.startswith("gen-")
        ]
        return max(existing, default=-1) + 1

    def publish(self, payload: dict) -> CheckpointHandle:
        generation = self._next_generation()
        staging = self.root / f".staging-{generation}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
        (staging / "payload.json").write_text(body, encoding="utf-8")
        manifest = {
            "generation": generation,
            "payload_sha256": _sha256_bytes(body.encode("utf-8")),
        }
        (staging / "_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        _fsync_path(staging / "payload.json")
        _fsync_path(staging / "_manifest.json")
        _fsync_path(staging, is_dir=True)
        final = self.root / f"gen-{generation:06d}"
        os.rename(staging, final)
        _fsync_path(self.root, is_dir=True)
        pointer_tmp = self.root / ".CURRENT.tmp"
        pointer_tmp.write_text(f"{final.name}\n", encoding="utf-8")
        _fsync_path(pointer_tmp)
        os.replace(pointer_tmp, self.root / "CURRENT")
        _fsync_path(self.root, is_dir=True)
        return CheckpointHandle(self.root, generation, final)


def read_latest_checkpoint(root: Path) -> tuple[int, dict]:
    """Read the newest COMPLETE generation, preferring the CURRENT pointer."""
    root = Path(root)
    generations = sorted(
        int(name.split("-")[1]) for name in os.listdir(root) if name.startswith("gen-")
    )
    if not generations:
        raise CheckpointError(f"no checkpoint generations under {root}")
    pointer_name = None
    pointer_path = root / "CURRENT"
    if pointer_path.is_file():
        pointer_name = pointer_path.read_text(encoding="utf-8").strip()
    ordered = sorted(generations, reverse=True)
    if pointer_name is not None and pointer_name.startswith("gen-"):
        pointed = int(pointer_name.split("-")[1])
        ordered = [pointed] + [g for g in ordered if g != pointed]
    last_error: Exception | None = None
    for generation in ordered:
        try:
            return generation, _load_generation(root / f"gen-{generation:06d}", generation)
        except CheckpointError as exc:
            last_error = exc
            continue
    raise CheckpointError(f"no complete checkpoint generation under {root}: {last_error}")


def _load_generation(directory: Path, expected_generation: int) -> dict:
    manifest_path = directory / "_manifest.json"
    payload_path = directory / "payload.json"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise CheckpointError(f"incomplete generation directory: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"unreadable manifest in {directory}: {exc}") from exc
    if manifest.get("generation") != expected_generation:
        raise CheckpointError(f"generation number drift in {directory}")
    body = payload_path.read_text(encoding="utf-8")
    if _sha256_bytes(body.encode("utf-8")) != manifest.get("payload_sha256"):
        raise CheckpointError(f"payload digest mismatch in {directory}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"unreadable payload in {directory}: {exc}") from exc
