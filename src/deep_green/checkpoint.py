"""Crash-safe checkpoint generations with byte-identical CPU restore.

Every generation is staged in a hidden sibling directory, fsynced file by
file, sealed by a digest-binding COMPLETE manifest, then atomically renamed
into place. A root LATEST pointer is republished atomically after each
generation. SIGKILL at any publication boundary leaves either the previous
or the next complete generation reachable, never a partial.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import uuid
from pathlib import Path

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load_file, save

LEGACY_SNAPSHOT_SCHEMA = "wordle.checkpoint/v1"
SNAPSHOT_SCHEMA = "wordle.checkpoint/v2"
COMPLETE_NAME = "COMPLETE"
LATEST_NAME = "LATEST"
PUBLICATION_BOUNDARIES = (
    "stage-created",
    "weights-fsynced",
    "optimizer-fsynced",
    "state-fsynced",
    "complete-fsynced",
    "generation-renamed",
    "latest-fsynced",
    "latest-renamed",
    "root-fsynced",
)

_PAYLOAD_BOUNDARIES = (
    ("weights.safetensors", "weights-fsynced"),
    ("optimizer.safetensors", "optimizer-fsynced"),
    ("state.json", "state-fsynced"),
)
_GENERATION_FILES = tuple(name for name, _ in _PAYLOAD_BOUNDARIES) + (COMPLETE_NAME,)
_PAYLOAD_NAMES = {name for name, _ in _PAYLOAD_BOUNDARIES}
_HEX = frozenset("0123456789abcdef")


class CheckpointError(RuntimeError):
    pass


def _validate_generation_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name.startswith(".")
        or name in (".", "..", LATEST_NAME)
        or "/" in name
        or "\\" in name
    ):
        raise CheckpointError(f"invalid generation name {name!r}")


def _kill_boundary(generation: str, boundary: str) -> None:
    if (
        os.environ.get("WORDLE_CHECKPOINT_KILL_GENERATION") == generation
        and os.environ.get("WORDLE_CHECKPOINT_KILL_BOUNDARY") == boundary
    ):
        os.kill(os.getpid(), signal.SIGKILL)


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    _fsync_file(path)


def save_snapshot(
    directory: Path,
    *,
    weights: dict[str, np.ndarray],
    optimizer_arrays: dict[str, np.ndarray],
    rng_state: dict,
    sampler_position: int,
    update_count: int,
    config: dict,
    backend_version: str,
    training_state: dict | None = None,
) -> Path:
    directory = Path(directory)
    _validate_generation_name(directory.name)
    if directory.is_symlink():
        raise CheckpointError(f"generation path is a symlink: {directory}")
    state = {
        "schema": SNAPSHOT_SCHEMA,
        "rng_state": rng_state,
        "sampler_position": sampler_position,
        "update_count": update_count,
        "config": config,
        "backend_version": backend_version,
        "training_state": {} if training_state is None else training_state,
    }
    _validate_state(state, require_training_state=True)
    root = directory.parent
    if root.is_symlink():
        raise CheckpointError(f"checkpoint root is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{directory.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    _kill_boundary(directory.name, "stage-created")
    latest_temp = None
    try:
        digests = {}
        payloads = {
            "weights.safetensors": save(
                {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in weights.items()
                },
                metadata={"schema": SNAPSHOT_SCHEMA},
            ),
            "optimizer.safetensors": save(
                {name: np.asarray(value) for name, value in optimizer_arrays.items()},
                metadata={"schema": SNAPSHOT_SCHEMA},
            ),
            "state.json": (
                json.dumps(state, indent=1, sort_keys=True) + "\n"
            ).encode("utf-8"),
        }
        for name, boundary in _PAYLOAD_BOUNDARIES:
            payload = payloads[name]
            target = staging / name
            target.write_bytes(payload)
            _fsync_file(target)
            digests[name] = hashlib.sha256(payload).hexdigest()
            _kill_boundary(directory.name, boundary)
        complete = (
            json.dumps(
                {
                    "schema": SNAPSHOT_SCHEMA,
                    "generation": directory.name,
                    "files": digests,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        complete_path = staging / COMPLETE_NAME
        complete_path.write_bytes(complete)
        _fsync_file(complete_path)
        _fsync_dir(staging)
        _kill_boundary(directory.name, "complete-fsynced")
        if not directory.exists():
            os.rename(staging, directory)
        elif _generation_matches(staging, directory):
            shutil.rmtree(staging)
        else:
            shutil.rmtree(staging, ignore_errors=True)
            raise CheckpointError(
                f"existing generation {directory.name!r} has different bytes; "
                "refusing to overwrite"
            )
        _fsync_dir(root)
        _kill_boundary(directory.name, "generation-renamed")
        latest_temp = root / f".{LATEST_NAME}.staging-{os.getpid()}-{uuid.uuid4().hex}"
        fd = os.open(str(latest_temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(fd, (directory.name + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        _kill_boundary(directory.name, "latest-fsynced")
        os.replace(latest_temp, root / LATEST_NAME)
        latest_temp = None
        _kill_boundary(directory.name, "latest-renamed")
        _fsync_dir(root)
        _kill_boundary(directory.name, "root-fsynced")
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
        if latest_temp is not None and latest_temp.exists():
            latest_temp.unlink()
        raise
    return directory


def _generation_matches(staging: Path, final: Path) -> bool:
    if final.is_symlink() or not final.is_dir():
        return False
    if {p.name for p in final.iterdir()} != set(_GENERATION_FILES):
        return False
    for name in _GENERATION_FILES:
        path = final / name
        if path.is_symlink() or not path.is_file():
            return False
        if path.read_bytes() != (staging / name).read_bytes():
            return False
    return True


def latest_snapshot_path(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink():
        raise CheckpointError(f"checkpoint root is a symlink: {root}")
    pointer = root / LATEST_NAME
    if pointer.is_symlink():
        raise CheckpointError(f"LATEST is a symlink: {pointer}")
    try:
        raw = pointer.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"no LATEST pointer under {root}: {exc}") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckpointError("LATEST is not valid UTF-8") from None
    if text.count("\n") != 1 or not text.endswith("\n"):
        raise CheckpointError("LATEST must contain exactly one generation name")
    name = text[:-1]
    try:
        _validate_generation_name(name)
    except CheckpointError:
        raise CheckpointError(f"LATEST names an invalid generation {name!r}") from None
    generation = root / name
    if generation.is_symlink():
        raise CheckpointError(f"LATEST generation {name!r} is a symlink")
    load_snapshot(generation)
    return generation


def load_latest_snapshot(root: Path) -> dict:
    return load_snapshot(latest_snapshot_path(root))


def load_snapshot(directory: Path) -> dict:
    directory = Path(directory)
    if directory.is_symlink():
        raise CheckpointError(f"snapshot is a symlink: {directory}")
    if not directory.is_dir():
        raise CheckpointError(f"snapshot missing: {directory}")
    if (directory / COMPLETE_NAME).is_file():
        _verify_complete(directory)
        return _load_payloads(directory, SNAPSHOT_SCHEMA)
    return _load_legacy(directory)


def _visible_names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()}


def _verify_complete(directory: Path) -> None:
    names = _visible_names(directory)
    missing_names = sorted(set(_GENERATION_FILES) - names)
    if missing_names:
        raise CheckpointError(f"snapshot incomplete; missing {missing_names}")
    if names != set(_GENERATION_FILES):
        raise CheckpointError(f"generation {directory.name!r} has unexpected files")
    complete_path = directory / COMPLETE_NAME
    if complete_path.is_symlink() or not complete_path.is_file():
        raise CheckpointError("COMPLETE is not a regular file")
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CheckpointError(f"malformed COMPLETE: {exc}") from None
    if not isinstance(complete, dict) or set(complete) != {
        "schema",
        "generation",
        "files",
    }:
        raise CheckpointError("COMPLETE is malformed")
    if complete["schema"] != SNAPSHOT_SCHEMA:
        raise CheckpointError("COMPLETE has an unknown schema")
    if complete["generation"] != directory.name:
        raise CheckpointError(
            f"COMPLETE generation {complete['generation']!r} "
            f"does not match {directory.name!r}"
        )
    files = complete["files"]
    if not isinstance(files, dict) or set(files) != _PAYLOAD_NAMES:
        raise CheckpointError("COMPLETE files manifest is malformed")
    for name in _GENERATION_FILES[:-1]:
        expected = files[name]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in _HEX for char in expected)
        ):
            raise CheckpointError(f"COMPLETE digest for {name} is malformed")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"snapshot payload {name} is not a regular file")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise CheckpointError(f"snapshot payload digest mismatch: {name}")


def _load_legacy(directory: Path) -> dict:
    state_path = directory / "state.json"
    legacy_files = {"state.json", "weights.safetensors", "optimizer.safetensors"}
    names = _visible_names(directory)
    missing = sorted(legacy_files - names)
    if missing:
        raise CheckpointError(f"snapshot incomplete; missing {missing}")
    if names != legacy_files:
        raise CheckpointError(f"generation {directory.name!r} has unexpected files")
    for name in legacy_files:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"snapshot payload {name} is not a regular file")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CheckpointError(f"malformed state.json: {exc}") from None
    schema = state.get("schema") if isinstance(state, dict) else None
    if schema == SNAPSHOT_SCHEMA:
        raise CheckpointError(f"snapshot incomplete; missing {COMPLETE_NAME}")
    if schema != LEGACY_SNAPSHOT_SCHEMA:
        raise CheckpointError(f"unknown snapshot schema {schema!r}")
    return _load_payloads(directory, LEGACY_SNAPSHOT_SCHEMA)


def _load_payloads(directory: Path, expected_schema: str) -> dict:
    try:
        state = json.loads(
            (directory / "state.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CheckpointError(f"malformed state.json: {exc}") from None
    if not isinstance(state, dict):
        raise CheckpointError("state.json is not an object")
    if state.get("schema") != expected_schema:
        raise CheckpointError(f"unknown snapshot schema {state.get('schema')!r}")
    _validate_state(state, require_training_state=expected_schema == SNAPSHOT_SCHEMA)
    try:
        weights = load_file(str(directory / "weights.safetensors"))
        optimizer = load_file(str(directory / "optimizer.safetensors"))
    except (SafetensorError, OSError) as exc:
        raise CheckpointError(f"unreadable safetensors payload: {exc}") from None
    return {
        "state": state,
        "weights": weights,
        "optimizer_arrays": optimizer,
    }


def _validate_state(state: dict, *, require_training_state: bool) -> None:
    if not isinstance(state.get("rng_state"), dict):
        raise CheckpointError("state.rng_state must be a dict")
    for field in ("sampler_position", "update_count"):
        value = state.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointError(f"state.{field} must be a nonnegative integer")
    if not isinstance(state.get("config"), dict):
        raise CheckpointError("state.config must be a dict")
    backend = state.get("backend_version")
    if not isinstance(backend, str) or not backend:
        raise CheckpointError("state.backend_version must be a nonempty string")
    if (require_training_state or "training_state" in state) and not isinstance(
        state.get("training_state"), dict
    ):
        raise CheckpointError("state.training_state must be a dict")


def snapshot_digest(snapshot: dict) -> str:
    hasher = hashlib.sha256()
    for name in sorted(snapshot["weights"]):
        array = np.ascontiguousarray(snapshot["weights"][name])
        hasher.update(name.encode("utf-8"))
        hasher.update(array.tobytes())
    for name in sorted(snapshot["optimizer_arrays"]):
        array = np.ascontiguousarray(snapshot["optimizer_arrays"][name])
        hasher.update(("opt:" + name).encode("utf-8"))
        hasher.update(array.tobytes())
    state_bytes = json.dumps(snapshot["state"], sort_keys=True).encode("utf-8")
    hasher.update(state_bytes)
    return hasher.hexdigest()
