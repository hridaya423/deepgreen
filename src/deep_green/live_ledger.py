"""Append-only public ledger of live games, anchored to a sealed release.

The ledger opens only against a valid SEALED seal.json whose archive hash
exists and matches. Each line is a canonical JSONL record binding a public
game event to the chain via sequence/previous_hash/entry_hash.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

LEDGER_SCHEMA = "wordle.live-ledger/v1"
SEAL_SCHEMA = "wordle.release-seal/v1"
_EVENT_KEYS = {"mode", "visible_rows", "outcome", "latency_ms", "override_status"}
_ROW_KEYS = {"guess", "marks"}
_MODES = {"normal", "hard"}
_OUTCOMES = {"WON", "LOST", "USER_WON", "CHAMPION_WON", "TIE"}
_OVERRIDE_STATUSES = {"CERTIFIED", "EXHIBITION", "UNVERIFIED", "PENDING"}
_MARKS = re.compile(r"^[byg]{5}$")
_WORD5 = re.compile(r"^[a-z]{5}$")


class LedgerError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"))).encode("utf-8")


def _require_regular(path: Path, label: str, *, directory: bool = False) -> None:
    if path.is_symlink() or not os.path.lexists(path):
        raise LedgerError(f"{label} missing or symlink: {path}")
    mode = path.stat().st_mode
    if directory:
        if not stat.S_ISDIR(mode):
            raise LedgerError(f"{label} is not a directory: {path}")
    elif not stat.S_ISREG(mode):
        raise LedgerError(f"{label} is not a regular file: {path}")


def _load_seal(seal_dir: Path) -> dict:
    seal_dir = Path(seal_dir)
    _require_regular(seal_dir, "seal directory", directory=True)
    _require_regular(seal_dir / "seal.json", "seal.json")
    _require_regular(seal_dir / "release.tar.gz", "sealed archive")
    try:
        seal = json.loads(
            (seal_dir / "seal.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"cannot read seal.json: {exc}") from None
    if (
        not isinstance(seal, dict)
        or seal.get("schema") != SEAL_SCHEMA
        or seal.get("status") != "SEALED"
    ):
        raise LedgerError("seal.json is not a SEALED release seal")
    archive = seal_dir / "release.tar.gz"
    expected = seal.get("archive_sha256")
    try:
        data = archive.read_bytes()
    except OSError as exc:
        raise LedgerError(f"sealed archive missing: {exc}") from None
    if _sha256(data) != expected or len(data) != seal.get("archive_bytes"):
        raise LedgerError("sealed archive hash diverges from seal.json")
    return seal


def _entry_hash(record: dict) -> str:
    body = {key: value for key, value in record.items() if key != "entry_hash"}
    return _sha256(_canonical(body))


def _validate_event(event) -> dict:
    if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
        raise LedgerError("ledger event must carry exactly the public fields")
    if event["mode"] not in _MODES:
        raise LedgerError("ledger event mode is invalid")
    rows = event["visible_rows"]
    if not isinstance(rows, list):
        raise LedgerError("visible_rows must be a list")
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise LedgerError("visible_rows entries carry exactly guess/marks")
        if not _WORD5.match(row["guess"] or ""):
            raise LedgerError("visible_rows guess is not a public Word5")
        if not _MARKS.match(row["marks"] or ""):
            raise LedgerError("visible_rows marks must be five b/y/g marks")
    if event["outcome"] not in _OUTCOMES:
        raise LedgerError("ledger event outcome is invalid")
    latency = event["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        raise LedgerError("ledger event latency_ms is invalid")
    override = event["override_status"]
    if override is not None and override not in _OVERRIDE_STATUSES:
        raise LedgerError("ledger event override_status is invalid")
    return dict(event)


def _genesis_event(seal: dict) -> dict:
    release = seal.get("release", {})
    return {
        "type": "genesis",
        "seal_sha256": _sha256(_canonical(seal)),
        "archive_sha256": seal["archive_sha256"],
        "weights_sha256": release.get("weights_sha256"),
        "manifest_sha256": release.get("manifest_sha256"),
        "proof_sha256": seal.get("proof_sha256"),
    }


class LiveLedger:
    def __init__(self, path: Path, seal_dir: Path, lock_fd: int):
        self.path = Path(path)
        self.seal_dir = Path(seal_dir)
        self._lock_fd = lock_fd
        self._entries: list[dict] = []

    @classmethod
    def open(cls, path, *, seal_dir) -> LiveLedger:
        path = Path(path)
        seal_dir = Path(seal_dir)
        if os.path.lexists(path):
            _require_regular(path, "ledger")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / f"{path.name}.lock"
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        ledger = cls(path, seal_dir, lock_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            seal = _load_seal(seal_dir)
            if path.exists():
                _require_regular(path, "ledger")
                ledger._entries = _read_chain(path)
                _verify_chain(ledger._entries, seal)
            else:
                genesis = {
                    "schema": LEDGER_SCHEMA,
                    "sequence": 0,
                    "previous_hash": None,
                    "genesis": _genesis_event(seal),
                }
                genesis["entry_hash"] = _entry_hash(genesis)
                ledger._entries = [genesis]
                ledger._write_line(genesis)
        except BaseException:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        return ledger

    def _write_line(self, record: dict) -> None:
        line = _canonical(record) + b"\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    @property
    def sequence(self) -> int:
        return len(self._entries) - 1

    def append(self, event: dict) -> dict:
        event = _validate_event(event)
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            seal = _load_seal(self.seal_dir)
            entries = _read_chain(self.path)
            _verify_chain(entries, seal)
            previous = entries[-1]
            record = {
                "schema": LEDGER_SCHEMA,
                "sequence": previous["sequence"] + 1,
                "previous_hash": previous["entry_hash"],
                "event": event,
            }
            record["entry_hash"] = _entry_hash(record)
            self._write_line(record)
            self._entries = entries + [record]
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        return record

    def verify(self) -> bool:
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            seal = _load_seal(self.seal_dir)
            entries = _read_chain(self.path)
            _verify_chain(entries, seal)
            self._entries = entries
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        return True

    def close(self) -> None:
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1


def _read_chain(path: Path) -> list[dict]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise LedgerError(f"cannot read ledger: {exc}") from None
    if not raw:
        raise LedgerError("ledger is empty")
    if not raw.endswith(b"\n"):
        raise LedgerError("ledger is truncated")
    entries = []
    for line in raw.split(b"\n")[:-1]:
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerError(f"ledger line is not valid JSON: {exc}") from None
        if not isinstance(record, dict):
            raise LedgerError("ledger line is not an object")
        entries.append(record)
    return entries


def _verify_chain(entries: list[dict], seal: dict) -> None:
    if not entries:
        raise LedgerError("ledger has no entries")
    genesis = entries[0]
    if set(genesis) != {
        "schema",
        "sequence",
        "previous_hash",
        "genesis",
        "entry_hash",
    }:
        raise LedgerError("genesis record is malformed")
    if (
        genesis["schema"] != LEDGER_SCHEMA
        or genesis["sequence"] != 0
        or genesis["previous_hash"] is not None
    ):
        raise LedgerError("genesis record anchor is malformed")
    if genesis["genesis"] != _genesis_event(seal):
        raise LedgerError("genesis does not anchor this seal")
    if _entry_hash(genesis) != genesis["entry_hash"]:
        raise LedgerError("genesis entry_hash diverges")
    previous_hash = genesis["entry_hash"]
    for index, record in enumerate(entries[1:], start=1):
        if set(record) != {
            "schema",
            "sequence",
            "previous_hash",
            "event",
            "entry_hash",
        }:
            raise LedgerError(f"ledger entry {index} is malformed")
        if record["schema"] != LEDGER_SCHEMA or record["sequence"] != index:
            raise LedgerError(f"ledger entry {index} sequence is broken")
        if record["previous_hash"] != previous_hash:
            raise LedgerError(f"ledger entry {index} chain is broken")
        _validate_event(record["event"])
        if _entry_hash(record) != record["entry_hash"]:
            raise LedgerError(f"ledger entry {index} hash diverges")
        previous_hash = record["entry_hash"]
