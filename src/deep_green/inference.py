"""Fail-closed policy runtime: real Seatbelt sandbox, bounded pipes, poisoned workers die."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Self

from deep_green.protocol import (
    MAX_REQUEST_BYTES,
    PolicyRequest,
    serialize_request,
)

RESPONSE_SCHEMA = "wordle.policy-response/v1"
MAX_RESPONSE_BYTES = 4096
MAX_STDERR_BYTES = 8192
DEFAULT_TIMEOUT_SECONDS = 5.0
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_RESPONSE_KEYS = {
    "schema",
    "status",
    "request_hash",
    "scored_word_count",
    "argmax",
    "artifact_sha256",
}


class PolicyRuntimeError(RuntimeError):
    pass


class PolicyIllegalActionError(PolicyRuntimeError):
    pass


class _NonFiniteConstant(ValueError):
    pass


def _reject_response_constant(value: str):
    raise _NonFiniteConstant(f"non-finite JSON constant {value!r} is not allowed")


def _scheme_path(path: str) -> str:
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def seatbelt_profile(release_dir: Path) -> str:
    """Deny-default Seatbelt profile: read-only allowlist, no network."""
    resolved_release = Path(release_dir).resolve()
    src_dir = Path(__file__).resolve().parent.parent
    read_paths = [
        resolved_release,
        src_dir,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path("/System/Library"),
        Path("/System/Cryptexes"),
        Path("/System/Volumes/Preboot"),
        Path("/usr/lib"),
        Path("/usr/bin"),
        Path("/private/var/folders"),
    ]
    read_literals = ["/dev/null", "/dev/urandom", "/dev/dtracehelper"]
    rules = [
        "(version 1)",
        "(deny default)",
        '(allow file-read-data (literal "/"))',
        "(allow file-read-metadata)",
    ]
    for path in read_paths:
        rules.append(f"(allow file-read* (subpath {_scheme_path(str(path))}))")
    for path in read_literals:
        rules.append(f"(allow file-read* (literal {_scheme_path(path)}))")
    rules += [
        "(allow file-map-executable)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(deny network*)",
        "(deny system-socket)",
    ]
    return "\n".join(rules) + "\n"


class PolicyRuntime:
    """Owns one sandboxed worker process; any protocol fault terminates it."""

    def __init__(
        self,
        release_dir: Path,
        action_words: tuple[str, ...],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        _worker_command: list[str] | None = None,
    ) -> None:
        from deep_green.release import ReleaseError, load_release

        release_dir = Path(release_dir).resolve()
        try:
            payload = load_release(release_dir)
        except (ReleaseError, OSError) as exc:
            raise PolicyRuntimeError(f"release load failed: {exc}") from exc
        if payload.action_words != tuple(action_words):
            raise PolicyRuntimeError(
                "action vocabulary mismatch with the payload lexicon"
            )
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise PolicyRuntimeError(
                "timeout_seconds must be a positive finite number"
            )
        if sys.platform == "darwin":
            if not Path(SANDBOX_EXEC).is_file():
                raise PolicyRuntimeError(f"missing {SANDBOX_EXEC}")
        else:
            raise PolicyRuntimeError("sandboxed runtime requires macOS Seatbelt")

        self._release_dir = release_dir
        self._action_words = tuple(action_words)
        self._artifact_sha256 = payload.weights_sha256
        self._manifest_sha256 = payload.manifest_sha256
        self._timeout = float(timeout_seconds)
        self._profile = seatbelt_profile(release_dir)
        self._profile_sha256 = hashlib.sha256(
            self._profile.encode("utf-8")
        ).hexdigest()
        self._worker_command = (
            list(_worker_command)
            if _worker_command is not None
            else [
                sys.executable,
                "-m",
                "deep_green.policy_worker",
                str(release_dir),
            ]
        )
        self._command = [SANDBOX_EXEC, "-p", self._profile, *self._worker_command]
        self._env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "HOME": os.environ.get("HOME", "/tmp"),
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self._poisoned = False
        self._shutdown = False
        self._sandboxed = False
        self._peak_rss_bytes = 0
        self._stdout_buffer = b""
        self._stderr_tail = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=self._release_dir,
            env=self._env,
            start_new_session=True,
        )
        self._process = process
        self._stdout_buffer = b""
        with self._stderr_lock:
            self._stderr_tail = bytearray()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(process,), daemon=True
        )
        self._stderr_thread.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                self._stderr_thread.join(timeout=2)
                detail = self._stderr_text()[-400:]
                raise PolicyRuntimeError(
                    f"sandboxed worker exited at launch (rc={code}): {detail}"
                )
            time.sleep(0.02)
        self._sandboxed = True

    def _drain_stderr(self, process: subprocess.Popen) -> None:
        assert process.stderr is not None
        fd = process.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > MAX_STDERR_BYTES:
                    del self._stderr_tail[: len(self._stderr_tail) - MAX_STDERR_BYTES]

    def _stderr_text(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_tail).decode("utf-8", errors="replace")

    @property
    def sandboxed(self) -> bool:
        return self._sandboxed

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def profile_sha256(self) -> str:
        return self._profile_sha256

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    def _write_all(self, data: bytes, deadline: float) -> None:
        fd = self._process.stdin.fileno()
        offset = 0
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._poison("timed out writing the request to the worker")
            _, writable, _ = select.select([], [fd], [], min(remaining, 0.25))
            if not writable:
                continue
            try:
                offset += os.write(fd, data[offset:])
            except (BrokenPipeError, OSError) as exc:
                self._poison("policy worker died mid-episode", exc)

    def _read_line(self, deadline: float) -> bytes:
        fd = self._process.stdout.fileno()
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                if newline > MAX_RESPONSE_BYTES:
                    self._poison("worker response exceeded the bounded line length")
                line = self._stdout_buffer[:newline]
                self._stdout_buffer = self._stdout_buffer[newline + 1 :]
                return line
            if len(self._stdout_buffer) > MAX_RESPONSE_BYTES:
                self._poison("worker response exceeded the bounded line length")
            if self._process.poll() is not None and not self._readable(fd):
                self._poison("policy worker produced no response")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._poison("timed out waiting for the worker response")
            try:
                ready, _, _ = select.select([fd], [], [], min(remaining, 0.25))
            except OSError as exc:
                self._poison("worker stdout failed", exc)
            if not ready:
                continue
            limit = max(1, MAX_RESPONSE_BYTES + 2 - len(self._stdout_buffer))
            try:
                chunk = os.read(fd, limit)
            except OSError as exc:
                self._poison("worker stdout failed", exc)
            if not chunk:
                self._poison("policy worker produced no response")
            self._stdout_buffer += chunk
            if len(self._stdout_buffer) > MAX_RESPONSE_BYTES + 1:
                self._poison("worker response exceeded the bounded line length")

    def _readable(self, fd: int) -> bool:
        try:
            ready, _, _ = select.select([fd], [], [], 0)
        except OSError:
            return False
        return bool(ready)

    def guess(self, request: PolicyRequest) -> int:
        if self._shutdown:
            raise PolicyRuntimeError("runtime is shut down")
        if self._poisoned:
            raise PolicyRuntimeError("runtime is poisoned; the episode is over")
        process = self._process
        if process is None or process.poll() is not None:
            self._poison("policy worker is not running")
        line = serialize_request(request, action_words=self._action_words)
        data = line.encode("utf-8") + b"\n"
        if len(data) > MAX_REQUEST_BYTES:
            self._poison("request exceeds the bounded line length")
        deadline = time.monotonic() + self._timeout
        self._write_all(data, deadline)
        response_line = self._read_line(deadline)
        try:
            response = json.loads(
                response_line.decode("utf-8"),
                parse_constant=_reject_response_constant,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            self._poison("policy worker emitted malformed JSON", exc)
        if not isinstance(response, dict) or set(response) != _RESPONSE_KEYS:
            self._poison("worker response must carry exactly the six public fields")
        if response["schema"] != RESPONSE_SCHEMA:
            self._poison(f"unknown response schema {response['schema']!r}")
        if response["status"] != "OK":
            self._poison(
                f"fail-closed: worker refused the episode ({response.get('reason', 'no reason')})"
            )
        expected_hash = request.request_hash(action_words=self._action_words)
        if not isinstance(response["request_hash"], str) or not hmac.compare_digest(
            response["request_hash"], expected_hash
        ):
            self._poison("worker response did not echo the request hash")
        if response["artifact_sha256"] != self._artifact_sha256:
            self._poison("worker did not bind the expected artifact hash")
        if response["scored_word_count"] != len(self._action_words):
            self._poison("worker scored a different vocabulary count")
        index = response["argmax"]
        if not isinstance(index, int) or isinstance(index, bool):
            self._poison(f"worker argmax must be an integer, got {index!r}")
        if not 0 <= index < len(self._action_words):
            self._poison(f"worker argmax {index} is out of range")
        if not request.legality_mask[index]:
            self._poison_illegal(f"worker selected illegal action index {index}")
        self._sample_rss(process)
        return index

    def _sample_rss(self, process: subprocess.Popen) -> None:
        try:
            out = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        try:
            kib = int(out.stdout.strip())
        except ValueError:
            return
        self._peak_rss_bytes = max(self._peak_rss_bytes, kib * 1024)

    def _poison(self, message: str, cause: BaseException | None = None):
        self._poisoned = True
        self._kill_process()
        raise PolicyRuntimeError(message) from cause

    def _poison_illegal(self, message: str, cause: BaseException | None = None):
        self._poisoned = True
        self._kill_process()
        raise PolicyIllegalActionError(message) from cause

    def _kill_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._join_stderr_thread()

    def _join_stderr_thread(self) -> None:
        thread = self._stderr_thread
        if thread is not None:
            thread.join(timeout=5)
            self._stderr_thread = None

    def reset(self) -> None:
        """Stop the worker and spawn a fresh sandboxed worker process."""
        if self._shutdown:
            raise PolicyRuntimeError("runtime is shut down")
        self._stop_process()
        self._poisoned = False
        self._sandboxed = False
        self._spawn()

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self._join_stderr_thread()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._stop_process()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()
