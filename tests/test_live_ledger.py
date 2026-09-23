"""Live ledger: seal-anchored append-only public game event chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_green.live_ledger import (
    LEDGER_SCHEMA,
    LedgerError,
    LiveLedger,
    _canonical,
)


def _seal_dir(tmp_path: Path, archive: bytes = b"deterministic-archive-bytes",
              **overrides) -> Path:
    seal_dir = tmp_path / "sealed"
    seal_dir.mkdir(parents=True)
    (seal_dir / "release.tar.gz").write_bytes(archive)
    seal = {
        "schema": "wordle.release-seal/v1",
        "status": "SEALED",
        "candidate_manifest_sha256": "a" * 64,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_bytes": len(archive),
        "verification_sha256": "b" * 64,
        "clean_reproduction_sha256": "c" * 64,
        "release": {"weights_sha256": "d" * 64, "manifest_sha256": "e" * 64},
        "proof_sha256": "f" * 64,
    }
    seal.update(overrides)
    (seal_dir / "seal.json").write_text(json.dumps(seal, sort_keys=True))
    return seal_dir


def _event(**overrides) -> dict:
    event = {
        "mode": "normal",
        "visible_rows": [{"guess": "stare", "marks": "bbygb"}],
        "outcome": "WON",
        "latency_ms": 12.5,
        "override_status": "EXHIBITION",
    }
    event.update(overrides)
    return event


def test_open_requires_valid_seal(tmp_path):
    seal_dir = tmp_path / "sealed"
    seal_dir.mkdir()
    (seal_dir / "release.tar.gz").write_bytes(b"x")
    with pytest.raises(LedgerError):
        LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)


def test_open_requires_matching_archive(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    (seal_dir / "release.tar.gz").write_bytes(b"different")
    with pytest.raises(LedgerError):
        LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)


def test_genesis_and_append_chain(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    ledger = LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    assert ledger.sequence == 0
    first = ledger.append(_event())
    second = ledger.append(_event(mode="hard", outcome="CHAMPION_WON"))
    assert first["sequence"] == 1 and second["sequence"] == 2
    assert second["previous_hash"] == first["entry_hash"]
    assert first["schema"] == LEDGER_SCHEMA
    assert ledger.verify() is True


def test_reopen_verifies_and_continues(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    path = tmp_path / "ledger.jsonl"
    ledger = LiveLedger.open(path, seal_dir=seal_dir)
    ledger.append(_event())
    reopened = LiveLedger.open(path, seal_dir=seal_dir)
    assert reopened.sequence == 1
    assert reopened.verify() is True


def test_rejects_hidden_fields(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    ledger = LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    for field in ("answer", "candidate_count", "posterior", "oracle", "entropy"):
        with pytest.raises(LedgerError):
            ledger.append(_event(**{field: "secret"}))


def test_rejects_bad_event_shapes(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    ledger = LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    with pytest.raises(LedgerError):
        ledger.append(_event(mode="expert"))
    with pytest.raises(LedgerError):
        ledger.append(_event(outcome="UNKNOWN"))
    with pytest.raises(LedgerError):
        ledger.append(_event(latency_ms=-1))
    with pytest.raises(LedgerError):
        ledger.append(_event(visible_rows=[{"guess": "stare"}]))


def test_truncated_and_tampered_fail(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    path = tmp_path / "ledger.jsonl"
    ledger = LiveLedger.open(path, seal_dir=seal_dir)
    ledger.append(_event())
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])
    with pytest.raises(LedgerError):
        LiveLedger.open(path, seal_dir=seal_dir)


def test_tampered_middle_entry_fails(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    path = tmp_path / "ledger.jsonl"
    ledger = LiveLedger.open(path, seal_dir=seal_dir)
    ledger.append(_event())
    ledger.append(_event())
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["event"]["outcome"] = "LOST"
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        ledger.verify()


def test_reanchored_genesis_fails(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    path = tmp_path / "ledger.jsonl"
    LiveLedger.open(path, seal_dir=seal_dir)
    other_dir = _seal_dir(tmp_path / "other", archive=b"other-archive")
    with pytest.raises(LedgerError):
        LiveLedger.open(path, seal_dir=other_dir)


def test_rejects_nonfinite_latency(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    ledger = LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    with pytest.raises(LedgerError):
        ledger.append(_event(latency_ms=float("nan")))
    with pytest.raises(LedgerError):
        ledger.append(_event(latency_ms=float("inf")))
    ledger.append(_event(latency_ms=0))
    assert ledger.verify() is True


def test_rejects_symlinked_ledger_path(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    real = tmp_path / "real.jsonl"
    real.write_text("")
    link = tmp_path / "ledger.jsonl"
    link.symlink_to(real)
    with pytest.raises(LedgerError):
        LiveLedger.open(link, seal_dir=seal_dir)


def test_rejects_symlinked_seal_members(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    link = tmp_path / "seal-link"
    link.symlink_to(seal_dir, target_is_directory=True)
    with pytest.raises(LedgerError):
        LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=link)
    seal_json = seal_dir / "seal.json"
    original = seal_json.read_bytes()
    seal_json.unlink()
    seal_json.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(LedgerError):
        LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    seal_json.unlink()
    seal_json.write_bytes(original)


def test_concurrent_instances_append_contiguous(tmp_path):
    import threading

    seal_dir = _seal_dir(tmp_path)
    path = tmp_path / "ledger.jsonl"
    first = LiveLedger.open(path, seal_dir=seal_dir)
    second = LiveLedger.open(path, seal_dir=seal_dir)
    errors = []

    def write(ledger, count):
        for index in range(count):
            try:
                ledger.append(_event(latency_ms=float(index)))
            except LedgerError as exc:
                errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(first, 5)),
        threading.Thread(target=write, args=(second, 5)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    reopened = LiveLedger.open(path, seal_dir=seal_dir)
    assert reopened.sequence == 10
    assert reopened.verify() is True


def test_canonical_hash_covers_body(tmp_path):
    seal_dir = _seal_dir(tmp_path)
    ledger = LiveLedger.open(tmp_path / "ledger.jsonl", seal_dir=seal_dir)
    record = ledger.append(_event())
    body = {k: v for k, v in record.items() if k != "entry_hash"}
    assert record["entry_hash"] == hashlib.sha256(_canonical(body)).hexdigest()
