"""Typed, hash-bound process adapter around the vendored exact solver."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from deep_green.oracle.build import ensure_solver

_SOLVER_SCHEMA = "alex1770.wordle-events/v1"

_IDENTITY_FIELDS = (
    "type",
    "schema",
    "commitdesc",
    "hardmode",
    "exhaustive",
    "maxguesses",
    "beta",
    "all_count",
    "hidden_count",
    "depthonly",
)
_RESULT_FIELDS = (
    "type",
    "schema",
    "objective",
    "average",
    "best_first",
    "nodes",
    "time_seconds",
    "depthonly",
    "checkpoints",
)
_CHECKPOINT_FIELDS = ("type", "schema", "generation")
_ACTION_VALUE_FIELDS = (
    "type", "schema", "action_index", "word", "bound", "exact"
)
_COMPLETION_FIELDS = (
    "type", "schema", "status", "covered_actions", "declared_actions"
)


class OracleRequestError(ValueError):
    pass


class OracleEventsError(ValueError):
    pass


class OracleExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleRequest:
    mode: str
    horizon: int
    rules_sha256: str
    prior_sha256: str
    action_space_sha256: str
    answer_set_sha256: str
    executable_sha256: str
    compiler_id: str
    solver_source_sha256: str
    action_words: tuple[str, ...] = field(default_factory=tuple)
    answer_words: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        required = {
            "mode": self.mode,
            "rules_sha256": self.rules_sha256,
            "prior_sha256": self.prior_sha256,
            "action_space_sha256": self.action_space_sha256,
            "answer_set_sha256": self.answer_set_sha256,
            "executable_sha256": self.executable_sha256,
            "compiler_id": self.compiler_id,
            "solver_source_sha256": self.solver_source_sha256,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise OracleRequestError(f"request fields unset (no defaults exist): {missing}")
        if self.mode not in ("normal", "hard"):
            raise OracleRequestError(f"unsupported mode {self.mode!r}")
        if self.horizon < 1 or self.horizon > 6:
            raise OracleRequestError(f"horizon out of range: {self.horizon}")
        for name in (
            "rules_sha256",
            "prior_sha256",
            "action_space_sha256",
            "answer_set_sha256",
            "executable_sha256",
            "solver_source_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise OracleRequestError(f"{name} is not a sha256 hex digest")
        if not self.action_words or not self.answer_words:
            raise OracleRequestError("action_words and answer_words are required inputs")

    def identity_digest(self) -> str:
        import hashlib

        payload = json.dumps(
            {
                "mode": self.mode,
                "horizon": self.horizon,
                "rules_sha256": self.rules_sha256,
                "prior_sha256": self.prior_sha256,
                "action_space_sha256": self.action_space_sha256,
                "answer_set_sha256": self.answer_set_sha256,
                "compiler_id": self.compiler_id,
                "solver_source_sha256": self.solver_source_sha256,
                "action_words": list(self.action_words),
                "answer_words": list(self.answer_words),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OracleActionValue:
    action_index: int
    word: str
    bound: int
    exact: bool


@dataclass(frozen=True)
class OracleRunRecord:
    identity: dict
    result: dict | None
    action_values: tuple[OracleActionValue, ...]
    checkpoint_generations: tuple[int, ...]
    completion: dict | None


@dataclass(frozen=True)
class OracleResult:
    objective: int
    average: float
    best_first_index: int
    best_first_word: str
    nodes: int
    time_seconds: float
    checkpoints: int


def _parse_event_line(line: str, number: int) -> dict:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OracleEventsError(f"event line {number} is not valid JSON: {exc}") from exc
    if not isinstance(event, dict):
        raise OracleEventsError(f"event line {number} is not a JSON object")
    kind = event.get("type")
    if kind == "run_identity":
        fields = _IDENTITY_FIELDS
    elif kind == "result":
        fields = _RESULT_FIELDS
    elif kind == "checkpoint":
        fields = _CHECKPOINT_FIELDS
    elif kind == "action_value":
        fields = _ACTION_VALUE_FIELDS
    elif kind == "completion":
        fields = _COMPLETION_FIELDS
    else:
        raise OracleEventsError(
            f"event line {number} has unknown event type {kind!r}; refusing to guess"
        )
    if event.get("schema") != _SOLVER_SCHEMA:
        raise OracleEventsError(
            f"event line {number} has unsupported schema {event.get('schema')!r}"
        )
    missing = [name for name in fields if name not in event]
    if missing:
        raise OracleEventsError(f"event line {number} ({kind}) is missing fields: {missing}")
    unknown = sorted(set(event) - set(fields))
    if unknown:
        raise OracleEventsError(
            f"event line {number} ({kind}) has unknown fields: {unknown}"
        )
    return event


def _parse_action_value(event: dict, identity: dict, number: int) -> OracleActionValue:
    index = event["action_index"]
    word = event["word"]
    bound = event["bound"]
    exact = event["exact"]
    all_count = identity["all_count"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= all_count
    ):
        raise OracleEventsError(
            f"event line {number}: action_index {index!r} outside 0..{all_count - 1}"
        )
    if (
        not isinstance(word, str)
        or len(word) != 5
        or any(char < "a" or char > "z" for char in word)
    ):
        raise OracleEventsError(f"event line {number}: malformed action word {word!r}")
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise OracleEventsError(f"event line {number}: invalid bound {bound!r}")
    if not isinstance(exact, bool):
        raise OracleEventsError(f"event line {number}: exact must be a boolean, got {exact!r}")
    return OracleActionValue(action_index=index, word=word, bound=bound, exact=exact)


def parse_events(text: str) -> OracleRunRecord:
    identity: dict | None = None
    result: dict | None = None
    checkpoints: list[int] = []
    action_values: list[OracleActionValue] = []
    seen_action_indices: set[int] = set()
    seen_action_words: set[str] = set()
    completion: dict | None = None
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        event = _parse_event_line(raw_line, number)
        kind = event["type"]
        if kind == "run_identity":
            if identity is not None:
                raise OracleEventsError("duplicate run_identity event")
            if number != 1:
                raise OracleEventsError("run_identity must be the first event")
            all_count = event["all_count"]
            if not isinstance(all_count, int) or isinstance(all_count, bool) or all_count < 0:
                raise OracleEventsError("run_identity all_count must be a nonnegative integer")
            identity = event
        elif kind == "result":
            if identity is None:
                raise OracleEventsError("result event appeared before run_identity")
            if completion is not None:
                raise OracleEventsError("result event appeared after completion")
            if result is not None:
                raise OracleEventsError("duplicate result event")
            result = event
        elif kind == "checkpoint":
            if identity is None:
                raise OracleEventsError("checkpoint event appeared before run_identity")
            if result is not None:
                raise OracleEventsError("checkpoint event appeared after the result event")
            generation = event["generation"]
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
                raise OracleEventsError(f"invalid checkpoint generation {generation!r}")
            checkpoints.append(generation)
        elif kind == "action_value":
            if identity is None:
                raise OracleEventsError("action_value event appeared before run_identity")
            if completion is not None:
                raise OracleEventsError("action_value event appeared after completion")
            if result is not None:
                raise OracleEventsError("action_value event appeared after the result event")
            value = _parse_action_value(event, identity, number)
            if value.action_index in seen_action_indices:
                raise OracleEventsError(
                    f"event line {number}: duplicate action_index {value.action_index}"
                )
            if value.word in seen_action_words:
                raise OracleEventsError(
                    f"event line {number}: duplicate action word {value.word!r}"
                )
            seen_action_indices.add(value.action_index)
            seen_action_words.add(value.word)
            action_values.append(value)
        elif kind == "completion":
            if completion is not None:
                raise OracleEventsError("duplicate completion event")
            if result is None:
                raise OracleEventsError("completion event without a preceding result")
            for field in ("covered_actions", "declared_actions"):
                value = event[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise OracleEventsError(f"completion {field} must be a nonnegative integer")
            completion = event
    if identity is None:
        raise OracleEventsError("event stream has no run_identity event")
    if result is None:
        raise OracleEventsError("event stream has no result event")
    if completion is None:
        raise OracleEventsError("event stream truncated without a completion event")
    generations = list(checkpoints)
    if generations != sorted(generations) or len(set(generations)) != len(generations):
        raise OracleEventsError(f"checkpoint generations are not strictly increasing: {generations}")
    if result["checkpoints"] != len(generations):
        raise OracleEventsError(
            f"result reports {result['checkpoints']} checkpoints but {len(generations)} events seen"
        )
    if completion["covered_actions"] != len(action_values):
        raise OracleEventsError(
            f"completion covered_actions {completion['covered_actions']} does not match "
            f"the {len(action_values)} parsed action_value events"
        )
    if completion["declared_actions"] != identity["all_count"]:
        raise OracleEventsError(
            f"completion declared_actions {completion['declared_actions']} does not match "
            f"identity all_count {identity['all_count']}"
        )
    return OracleRunRecord(
        identity, result, tuple(action_values), tuple(generations), completion
    )


def _parse_stdout_facts(stdout: str) -> dict:
    facts: dict = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Hidden wordlist filename = ") and ", size = " in line:
            facts["hidden_count"] = int(line.rsplit(", size = ", 1)[1])
        elif line.startswith("Best first guess = "):
            facts["best_first_word"] = line.split(" = ", 1)[1]
        elif line.startswith("Best first guess score"):
            tail = line.rsplit("= ", 1)[1]
            facts["objective"] = int(tail)
        elif line.startswith("Average guesses reqd using best first guess = "):
            facts["average"] = float(line.rsplit("= ", 1)[1])
    return facts


def _solver_action_order(action_words: tuple[str, ...], answer_words: tuple[str, ...]) -> tuple[str, ...]:
    """Reproduce upstream orderwordlists(): sorted hidden first, then the rest sorted."""
    hidden = sorted(set(answer_words))
    rest = sorted(set(action_words) - set(answer_words))
    return tuple(hidden + rest)


def run_oracle(
    request: OracleRequest,
    *,
    repo_root: Path,
    work_dir: Path,
) -> OracleResult:
    import hashlib

    def sha_hex(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    from deep_green.rules import ActionSpace

    action_digest = ActionSpace.parse(request.action_words).digest
    if action_digest != request.action_space_sha256:
        raise OracleRequestError("action_words do not match action_space_sha256")
    answer_digest = sha_hex("\n".join(request.answer_words))
    if answer_digest != request.answer_set_sha256:
        raise OracleRequestError("answer_words do not match answer_set_sha256")

    from deep_green.oracle.build import vendor_dir

    cache_dir = work_dir / "build"
    exe_path, actual_digest, compiler_id = ensure_solver(vendor_dir(repo_root), cache_dir)
    if compiler_id != request.compiler_id:
        raise OracleExecutionError(
            f"solver compiler drift: request pinned {request.compiler_id!r}, "
            f"found {compiler_id!r}"
        )
    if actual_digest != request.executable_sha256:
        raise OracleExecutionError(
            f"solver binary hash drift: request pinned {request.executable_sha256}, found {actual_digest}"
        )

    run_dir = work_dir / "runs" / request.identity_digest()
    run_dir.mkdir(parents=True, exist_ok=True)
    all_path = run_dir / "all.txt"
    hidden_path = run_dir / "hidden.txt"
    events_path = run_dir / "events.jsonl"
    all_path.write_text("\n".join(request.action_words) + "\n", encoding="utf-8")
    hidden_path.write_text("\n".join(request.answer_words) + "\n", encoding="utf-8")

    command = [
        str(exe_path),
        "-a",
        str(all_path),
        "-h",
        str(hidden_path),
        "-g",
        str(request.horizon),
        "-E",
        str(events_path),
    ]
    if request.mode == "hard":
        command.append("-H")
    environment = {k: v for k, v in os.environ.items()}
    completed = subprocess.run(
        command,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    stdout_text = completed.stdout
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise OracleExecutionError(f"solver exited {completed.returncode}: {detail[:2000]}")
    if not events_path.is_file():
        raise OracleExecutionError("solver produced no events file despite exiting zero")

    record = parse_events(events_path.read_text(encoding="utf-8"))
    if record.completion is not None and record.completion["status"] != "completed":
        raise OracleExecutionError(f"solver reported status {record.completion['status']!r}")
    assert record.identity is not None and record.result is not None
    if record.identity["exhaustive"] is not True:
        raise OracleEventsError("refusing to treat a heuristic search as an exact proof")
    if record.identity["hidden_count"] != len(request.answer_words):
        raise OracleEventsError("identity hidden_count does not match the request")
    if record.identity["hardmode"] != (1 if request.mode == "hard" else 0):
        raise OracleEventsError("identity hardmode does not match the requested mode")
    if record.result["objective"] is None:
        raise OracleEventsError("exhaustive run reported no finite objective")

    facts = _parse_stdout_facts(stdout_text)
    if facts.get("objective") != record.result["objective"]:
        raise OracleExecutionError(
            f"stdout objective {facts.get('objective')!r} contradicts event objective "
            f"{record.result['objective']!r}"
        )
    best_index = record.result["best_first"]
    solver_order = _solver_action_order(request.action_words, request.answer_words)
    if best_index is None or not 0 <= best_index < len(solver_order):
        raise OracleExecutionError(f"best_first index out of range: {best_index!r}")
    best_word = solver_order[best_index]
    if facts.get("best_first_word") not in (None, best_word):
        raise OracleExecutionError(
            f"stdout best word {facts.get('best_first_word')!r} contradicts event index {best_index}"
        )

    return OracleResult(
        objective=int(record.result["objective"]),
        average=float(record.result["average"]),
        best_first_index=int(best_index),
        best_first_word=best_word,
        nodes=int(record.result["nodes"]),
        time_seconds=float(record.result["time_seconds"]),
        checkpoints=int(record.result["checkpoints"]),
    )
