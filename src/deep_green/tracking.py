"""Immutable run records, append-only evidence logs, atomic artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

RUN_SPEC_FIELDS = (
    "code_identity",
    "benchmark_id",
    "rules_sha256",
    "prior_id",
    "oracle_sha256",
    "action_space_sha256",
    "model_config_sha256",
    "backend",
    "optimizer",
    "data_sha256",
    "seed",
    "execution_mode",
)


class RunStoreError(RuntimeError):
    pass


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunSpec:
    code_identity: str
    benchmark_id: str
    rules_sha256: str
    prior_id: str
    oracle_sha256: str
    action_space_sha256: str
    model_config_sha256: str
    backend: str
    optimizer: str
    data_sha256: str
    seed: int
    execution_mode: str

    def to_dict(self) -> dict:
        return {field: getattr(self, field) for field in RUN_SPEC_FIELDS}

    @staticmethod
    def from_dict(raw: dict) -> RunSpec:
        missing = [field for field in RUN_SPEC_FIELDS if field not in raw]
        if missing:
            raise RunStoreError(f"run spec missing fields: {missing}")
        extra = sorted(set(raw) - set(RUN_SPEC_FIELDS))
        if extra:
            raise RunStoreError(f"run spec has unknown fields: {extra}")
        return RunSpec(**{field: raw[field] for field in RUN_SPEC_FIELDS})


class EventLog:
    """Append-only canonical JSONL with single-writer lock and tail recovery."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def append(self, event: dict) -> None:
        with open(self._lock_path, "w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                line = canonical_json(event) + "\n"
                self._handle.write(line)
                self._handle.flush()
                os.fsync(self._handle.fileno())
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def read_recovered(self) -> list[dict]:
        """Read all complete lines; a truncated final line is dropped."""
        events = []
        if not self.path.is_file():
            return events
        text = self.path.read_text(encoding="utf-8")
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            is_last = index == len(lines) - 1
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if not is_last:
                    raise RunStoreError(f"corrupt event mid-file at line {index + 1}")
        return events

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class RunStore:
    """Owns one immutable run directory under runs/."""

    def __init__(self, runs_root: Path, spec: RunSpec) -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        manifest_body = canonical_json(spec.to_dict())
        run_id = _sha_hex(manifest_body)[:16]
        self.run_dir = self.runs_root / run_id
        self.manifest_path = self.run_dir / "manifest.json"
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("spec") != spec.to_dict():
                raise RunStoreError(
                    f"identity drift on resume for run {run_id}: "
                    "the stored manifest does not match the requested spec"
                )
            self.spec = spec
        else:
            staging = self.run_dir.with_suffix(".staging")
            staging.mkdir(parents=True, exist_ok=True)
            payload = {"spec": spec.to_dict(), "run_id": run_id}
            (staging / "manifest.json").write_text(
                canonical_json(payload) + "\n", encoding="utf-8"
            )
            os.replace(staging, self.run_dir)
            self.spec = spec

    def require_resume(self) -> None:
        """Explicit identity check for callers resuming an existing run."""
        existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if existing.get("spec") != self.spec.to_dict():
            raise RunStoreError(
                f"identity drift on resume for run {self.run_id}: "
                "the stored manifest does not match the requested spec"
            )

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def event_log(self, name: str = "events.jsonl") -> EventLog:
        return EventLog(self.run_dir / name)

    def publish_artifact(self, name: str, payload: bytes) -> Path:
        """Atomically publish an artifact; no readable partial may ever appear."""
        final = self.run_dir / name
        staging = self.run_dir / f".{name}.staging-{os.getpid()}"
        staging.write_bytes(payload)
        directory_fd = os.open(str(self.run_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, final)
        return final

    def close(self, inventory: dict) -> Path:
        closed_marker = self.run_dir / "CLOSED"
        if closed_marker.exists():
            raise RunStoreError(f"run {self.run_id} is already CLOSED")
        inventory["run_id"] = self.run_id
        inventory_path = self.run_dir / ".run-inventory.staging"
        inventory_path.write_text(
            json.dumps(inventory, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        directory_fd = os.open(str(self.run_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(inventory_path, self.run_dir / "run-inventory.json")
        closed_marker.write_text(canonical_json({"closed_at_inventory": True}) + "\n")
        return self.run_dir / "run-inventory.json"

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))
