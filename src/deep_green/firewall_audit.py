"""Firewall audit: prove noninterference and emit release evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

AUDIT_SCHEMA = "wordle.firewall/v1"


class FirewallAuditError(RuntimeError):
    pass


def _sha_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_release(release_dir: Path, *, output_path: Path | None = None) -> dict:
    """Run the full attack battery; emit PASS evidence bound to payload hashes."""
    from deep_green.inference import (
        PolicyRuntime,
        PolicyRuntimeError,
        seatbelt_profile,
    )
    from deep_green.protocol import (
        PolicyRequest,
        ProtocolError,
        parse_request,
        serialize_request,
        validate_parsed_request,
    )
    from deep_green.release import ReleaseError, load_release
    from deep_green.rules import ActionSpace

    release_dir = Path(release_dir)
    findings: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        findings.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    try:
        payload = load_release(release_dir)
        check("strict-release-load", True, "release payload accepted")
    except (ReleaseError, OSError) as exc:
        check("strict-release-load", False, str(exc)[:200])
        return _evidence(findings, None, None, None, output_path)

    words = payload.action_words
    action_digest = ActionSpace.parse(words).digest
    profile = seatbelt_profile(release_dir)
    profile_sha256 = _sha_hex(profile.encode("utf-8"))

    try:
        runtime = PolicyRuntime(release_dir=release_dir, action_words=words)
    except (PolicyRuntimeError, OSError) as exc:
        runtime = None
        check(
            "actual-seatbelt-launch",
            False,
            f"{type(exc).__name__}: {exc}"[:200],
        )

    if runtime is not None:
        try:
            check(
                "actual-seatbelt-launch",
                runtime.sandboxed is True,
                f"sandboxed={runtime.sandboxed} pid={runtime.pid}",
            )
            request = PolicyRequest(
                mode="normal",
                turn=2,
                played_rows=(("crane", "bbybb"),),
                action_space_sha256=action_digest,
                legality_mask=tuple(i % 7 != 3 for i in range(len(words))),
            )
            first = runtime.guess(request)
            second = runtime.guess(request)
            check(
                "request-determinism", first == second, f"argmax stable at {first}"
            )
            runtime.reset()
            reset_choice = runtime.guess(request)
            check(
                "reset-determinism",
                reset_choice == first,
                f"fresh worker argmax {reset_choice}",
            )
            check(
                "full-vocabulary-count",
                len(words) == 5022,
                f"{len(words)} actions scored",
            )
            empty_mask = PolicyRequest(
                mode="normal",
                turn=1,
                played_rows=(),
                action_space_sha256=action_digest,
                legality_mask=(False,) * len(words),
            )
            try:
                line = serialize_request(empty_mask, action_words=words)
                raw = parse_request(line)
                validate_parsed_request(raw, action_words=words)
                check("fail-closed-empty-mask", False, "empty mask was accepted")
            except (ProtocolError, PolicyRuntimeError):
                check("fail-closed-empty-mask", True, "empty mask fails closed")
        except (PolicyRuntimeError, ProtocolError, OSError) as exc:
            check(
                "runtime-battery",
                False,
                f"{type(exc).__name__}: {exc}"[:200],
            )
        finally:
            runtime.shutdown()

    battery_fields = (
        "answer_id",
        "candidate_count",
        "posterior",
        "entropy",
        "diagnostics",
        "oracle_value",
        "secret_word",
        "target_word",
        "daily_id",
        "prior_weight",
        "extras",
        "fold_id",
        "exposure_track",
    )
    rejected = all(
        _rejects_field(marker) for marker in battery_fields
    )
    check(
        "forbidden-field-parser-battery",
        rejected,
        f"{len(battery_fields)} forbidden field classes rejected",
    )

    network_probe = _probe(
        profile,
        release_dir,
        (
            "import socket,sys\n"
            "try:\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.bind(('127.0.0.1', 0))\n"
            "    print('NETWORK-ALLOWED')\n"
            "except PermissionError:\n"
            "    print('NETWORK-BLOCKED PermissionError')\n"
            "except Exception as exc:\n"
            "    print('NETWORK-OTHER', type(exc).__name__)\n"
        ),
    )
    check(
        "network-denied",
        "NETWORK-BLOCKED PermissionError" in network_probe.stdout,
        network_probe.stdout.strip()[:120],
    )
    target = (release_dir / "audit-probe.tmp").resolve()
    write_probe = _probe(
        profile,
        release_dir,
        (
            "import sys\n"
            f"path = {str(target)!r}\n"
            "try:\n"
            "    open(path, 'w').write('x')\n"
            "    print('WRITE-ALLOWED')\n"
            "except PermissionError:\n"
            "    print('WRITE-BLOCKED PermissionError')\n"
            "except Exception as exc:\n"
            "    print('WRITE-OTHER', type(exc).__name__)\n"
        ),
    )
    write_denied = (
        "WRITE-BLOCKED PermissionError" in write_probe.stdout
        and not target.exists()
    )
    target.unlink(missing_ok=True)
    check(
        "file-write-denied",
        write_denied,
        write_probe.stdout.strip()[:120],
    )

    return _evidence(
        findings,
        payload.weights_sha256,
        payload.manifest_sha256,
        profile_sha256,
        output_path,
    )


def _probe(profile: str, cwd: Path, script: str):
    return subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", profile, sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=cwd,
    )


def _rejects_field(field: str) -> bool:
    from deep_green.protocol import ProtocolError, parse_request

    line = (
        '{"schema":"wordle.policy-ipc/v1","mode":"normal","turn":1,'
        '"played_rows":[],"action_space_sha256":"' + "0" * 64 + '",'
        '"action_count":5022,"legality_mask":[1],"request_hash":"'
        + "0" * 64 + '","' + field + '":"x"}'
    )
    try:
        parse_request(line)
    except ProtocolError as exc:
        return "forbidden field" in str(exc)
    return False


def _evidence(
    findings: list[dict],
    weights_sha256: str | None,
    manifest_sha256: str | None,
    profile_sha256: str | None,
    output_path: Path | None,
) -> dict:
    evidence = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if all(f["status"] == "PASS" for f in findings) else "FAIL",
        "weights_sha256": weights_sha256,
        "manifest_sha256": manifest_sha256,
        "sandbox_profile_sha256": profile_sha256,
        "checks": findings,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
        temp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, output_path)
    return evidence
