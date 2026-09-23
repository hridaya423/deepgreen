"""PolicyRuntime: real Seatbelt launch plus the fail-closed fault battery."""

from __future__ import annotations

import hashlib
import json
import sys
import time

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
RELEASE = REPO_ROOT / "tests/fixtures/release"
WEIGHTS_SHA256 = hashlib.sha256(
    (RELEASE / "weights.safetensors").read_bytes()
).hexdigest()

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Seatbelt sandbox is macOS-only"
)


@pytest.fixture(scope="module")
def action_words():
    return tuple((RELEASE / "accepted-guesses.txt").read_text().split())


def _request(action_words, *, mask=None):
    from deep_green.protocol import PolicyRequest
    from deep_green.rules import ActionSpace

    return PolicyRequest(
        mode="normal",
        turn=1,
        played_rows=(),
        action_space_sha256=ActionSpace.parse(action_words).digest,
        legality_mask=mask if mask is not None else (True,) * len(action_words),
    )


def _scripted_worker(body_or_script):
    """A worker command that runs a -c script under the same sandbox."""
    return [sys.executable, "-c", body_or_script]


_ECHO_TEMPLATE = """
import json, sys
line = sys.stdin.buffer.readline()
req = json.loads(line)
resp = {resp}
print(json.dumps(resp, sort_keys=True), flush=True)
sys.stdin.buffer.readline()
"""


def _ok_response(extra_argmax=0, **overrides):
    resp = {
        "schema": "wordle.policy-response/v1",
        "status": "OK",
        "request_hash": "ECHO",
        "scored_word_count": 5022,
        "argmax": extra_argmax,
        "artifact_sha256": WEIGHTS_SHA256,
    }
    resp.update(overrides)
    return resp


def _worker_for(response: dict) -> list[str]:
    fields = {
        key: ("req['request_hash']" if value == "ECHO" else json.dumps(value))
        for key, value in response.items()
    }
    body = "{" + ",".join(f"{json.dumps(k)}:{v}" for k, v in fields.items()) + "}"
    return _scripted_worker(_ECHO_TEMPLATE.format(resp=body))


def test_real_worker_launches_under_seatbelt(action_words):
    from deep_green.inference import PolicyRuntime

    with PolicyRuntime(release_dir=RELEASE, action_words=action_words) as runtime:
        assert runtime.sandboxed is True
        assert runtime._process.args[0] == "/usr/bin/sandbox-exec"
        assert "-p" in runtime._process.args
        index = runtime.guess(_request(action_words))
        assert 0 <= index < len(action_words)
        assert runtime.artifact_sha256 == WEIGHTS_SHA256
        assert len(runtime.profile_sha256) == 64


def test_reset_spawns_a_fresh_pid(action_words):
    from deep_green.inference import PolicyRuntime

    with PolicyRuntime(release_dir=RELEASE, action_words=action_words) as runtime:
        first_pid = runtime.pid
        runtime.reset()
        assert runtime.pid != first_pid
        assert runtime.sandboxed is True
        index = runtime.guess(_request(action_words))
        assert 0 <= index < len(action_words)


def test_shutdown_is_bounded(action_words):
    from deep_green.inference import PolicyRuntime

    runtime = PolicyRuntime(release_dir=RELEASE, action_words=action_words)
    started = time.monotonic()
    runtime.shutdown()
    assert time.monotonic() - started < 15
    assert runtime._process.poll() is not None


def test_timeout_poisons_and_terminates(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        timeout_seconds=0.5,
        _worker_command=_scripted_worker("import time; time.sleep(60)"),
    ) as runtime:
        pid = runtime.pid
        with pytest.raises(PolicyRuntimeError):
            runtime.guess(_request(action_words))
        deadline = time.monotonic() + 10
        while runtime._process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert runtime._process.poll() is not None, f"worker {pid} survived"


def test_crash_eof_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        timeout_seconds=5,
        _worker_command=_scripted_worker(
            "import sys; sys.stdin.readline(); sys.exit(3)"
        ),
    ) as runtime, pytest.raises(PolicyRuntimeError):
        runtime.guess(_request(action_words))


def test_malformed_json_response_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_scripted_worker(
            "import sys; sys.stdin.readline(); print('this is not json')"
        ),
    ) as runtime, pytest.raises(PolicyRuntimeError, match="malformed|JSON"):
        runtime.guess(_request(action_words))


def test_oversized_stdout_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_scripted_worker(
            "import sys; sys.stdin.readline(); sys.stdout.write('x' * 9000)"
        ),
    ) as runtime, pytest.raises(PolicyRuntimeError):
        runtime.guess(_request(action_words))


def test_stderr_flood_does_not_deadlock(action_words):
    from deep_green.inference import PolicyRuntime

    script = (
        "import sys\n"
        "line = sys.stdin.buffer.readline()\n"
        "sys.stderr.write('e' * 1048576)\n"
        "sys.stderr.flush()\n"
        "req = __import__('json').loads(line)\n"
        f"resp = {_ok_response()!r}\n"
        "resp['request_hash'] = req['request_hash']\n"
        "print(__import__('json').dumps(resp, sort_keys=True), flush=True)\n"
        "sys.stdin.buffer.readline()\n"
    )
    with PolicyRuntime(
        RELEASE, action_words, _worker_command=_scripted_worker(script)
    ) as runtime:
        index = runtime.guess(_request(action_words))
        assert index == 0


def test_wrong_request_hash_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(request_hash="0" * 64)),
    ) as runtime, pytest.raises(PolicyRuntimeError, match="request"):
        runtime.guess(_request(action_words))


def test_wrong_artifact_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(artifact_sha256="0" * 64)),
    ) as runtime, pytest.raises(PolicyRuntimeError, match="artifact"):
        runtime.guess(_request(action_words))


def test_wrong_scored_count_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(scored_word_count=3)),
    ) as runtime, pytest.raises(PolicyRuntimeError, match="count"):
        runtime.guess(_request(action_words))


def test_extra_response_field_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(logits=[1.0])),
    ) as runtime, pytest.raises(PolicyRuntimeError):
        runtime.guess(_request(action_words))


def test_out_of_range_argmax_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(extra_argmax=5022)),
    ) as runtime, pytest.raises(PolicyRuntimeError):
        runtime.guess(_request(action_words))


def test_bool_argmax_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(extra_argmax=True)),
    ) as runtime, pytest.raises(PolicyRuntimeError):
        runtime.guess(_request(action_words))


def test_illegal_argmax_raises_illegal_action(action_words):
    from deep_green.inference import (
        PolicyIllegalActionError,
        PolicyRuntime,
    )

    mask = (True, False) + (True,) * (len(action_words) - 2)
    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_worker_for(_ok_response(extra_argmax=1)),
    ) as runtime, pytest.raises(PolicyIllegalActionError):
        runtime.guess(_request(action_words, mask=mask))


def test_poisoned_runtime_cannot_guess_again(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with PolicyRuntime(
        RELEASE,
        action_words,
        _worker_command=_scripted_worker(
            "import sys; sys.stdin.readline(); print('garbage')"
        ),
    ) as runtime:
        with pytest.raises(PolicyRuntimeError):
            runtime.guess(_request(action_words))
        with pytest.raises(PolicyRuntimeError):
            runtime.guess(_request(action_words))


def test_runtime_loads_release_through_strict_loader(action_words, tmp_path):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    tampered = tmp_path / "release"
    import shutil

    shutil.copytree(RELEASE, tampered)
    (tampered / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(PolicyRuntimeError):
        PolicyRuntime(release_dir=tampered, action_words=action_words)


@pytest.mark.parametrize(
    "value", [True, 0, -1.0, float("inf"), float("nan"), "5", None]
)
def test_invalid_timeout_rejected(action_words, value):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    with pytest.raises(PolicyRuntimeError, match="timeout"):
        PolicyRuntime(RELEASE, action_words, timeout_seconds=value)


def test_oversized_valid_json_response_poisons(action_words):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    script = (
        "import sys, json\n"
        "sys.stdin.readline()\n"
        f"print(json.dumps({{'note': '{'x' * 5000}'}}))\n"
    )
    with PolicyRuntime(
        RELEASE, action_words, _worker_command=_scripted_worker(script)
    ) as runtime, pytest.raises(PolicyRuntimeError, match="exceed|bounded|malformed"):
        runtime.guess(_request(action_words))


def _exact_size_json_line(total: int) -> str:
    pad = "x" * (total - len(json.dumps({"note": ""})))
    line = json.dumps({"note": pad})
    assert len(line) == total
    return line


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (4096, "six public fields"),
        (4097, "bounded line length"),
    ],
)
def test_response_line_length_boundary(action_words, payload, match):
    from deep_green.inference import PolicyRuntime, PolicyRuntimeError

    script = (
        "import sys\n"
        "sys.stdin.readline()\n"
        f"print({_exact_size_json_line(payload)!r})\n"
    )
    with PolicyRuntime(
        RELEASE, action_words, _worker_command=_scripted_worker(script)
    ) as runtime, pytest.raises(PolicyRuntimeError, match=match):
        runtime.guess(_request(action_words))
