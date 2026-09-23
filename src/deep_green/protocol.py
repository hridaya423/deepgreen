"""Strict canonical JSONL IPC between the evaluator and the policy worker.

The request allowlist is the firewall: mode, turn, played rows, action-space
identity, public legality mask, and a request hash over those public bytes.
Every forbidden field class fails at BOTH serializer and parser.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass

PROTOCOL_SCHEMA = "wordle.policy-ipc/v1"
MAX_REQUEST_BYTES = 65536

_ALLOWED_REQUEST_FIELDS = (
    "schema",
    "mode",
    "turn",
    "played_rows",
    "action_space_sha256",
    "action_count",
    "legality_mask",
    "request_hash",
)

_FORBIDDEN_FIELD_MARKERS = (
    "answer",
    "candidate",
    "posterior",
    "entropy",
    "diagnostic",
    "oracle",
    "secret",
    "target",
    "daily",
    "prior_",
    "extra",
    "fold",
    "exposure",
)


class ProtocolError(ValueError):
    pass


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _reject_constant(value: str):
    raise ProtocolError(f"non-finite JSON constant {value!r} is not allowed")


def _validate_turn(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"turn must be an integer, got {value!r}")
    if not 1 <= value <= 6:
        raise ProtocolError(f"turn {value} outside the horizon")
    return value


def _validate_rows(rows, turn: int) -> list[tuple[str, str]]:
    from deep_green.rules import Feedback, Word5

    if not isinstance(rows, list):
        raise ProtocolError("played_rows must be a list")
    parsed = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ProtocolError("played rows must be [word5, marks] pairs")
        guess_text, feedback_text = row
        if not isinstance(guess_text, str) or not isinstance(feedback_text, str):
            raise ProtocolError("played rows must contain strings")
        try:
            Word5.parse(guess_text)
        except ValueError as exc:
            raise ProtocolError(f"invalid guess word {guess_text!r}") from exc
        try:
            Feedback.parse(feedback_text)
        except ValueError as exc:
            raise ProtocolError(f"invalid marks in row {feedback_text!r}") from exc
        parsed.append((guess_text, feedback_text))
    if len(parsed) != turn - 1:
        raise ProtocolError(f"turn {turn} requires exactly {turn - 1} played rows")
    return parsed


def _check_public_board(mode: str, turn: int, rows) -> None:
    from deep_green.rules import board_from_public_state

    try:
        board_from_public_state(mode=mode, turn=turn, rows=rows)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"public board state is unreachable: {exc}") from exc


def _validate_mask(mask, expected: int) -> tuple[bool, ...]:
    if not isinstance(mask, list):
        raise ProtocolError("legality_mask must be a list")
    if len(mask) != expected:
        raise ProtocolError("mask length mismatch")
    for value in mask:
        if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
            raise ProtocolError(f"legality_mask values must be JSON ints 0/1, got {value!r}")
    parsed = tuple(bool(v) for v in mask)
    if not any(parsed):
        raise ProtocolError("empty legality mask ends the episode; no fallback exists")
    return parsed


def _canonical_action_words(action_words: tuple[str, ...]) -> tuple[str, ...]:
    from deep_green.rules import ActionSpace

    try:
        space = ActionSpace.parse(action_words)
    except ValueError as exc:
        raise ProtocolError(f"action words are not a valid space: {exc}") from exc
    if space.words != tuple(action_words):
        raise ProtocolError("action words are not in canonical sorted order")
    return space.words


@dataclass(frozen=True)
class PolicyRequest:
    mode: str
    turn: int
    played_rows: tuple[tuple[str, str], ...]
    action_space_sha256: str
    legality_mask: tuple[bool, ...]

    def request_hash(self, *, action_words: tuple[str, ...]) -> str:
        payload = json.dumps(
            {
                "mode": self.mode,
                "turn": self.turn,
                "played_rows": [list(row) for row in self.played_rows],
                "action_space_sha256": self.action_space_sha256,
                "action_words": list(action_words),
                "legality_mask": [bool(v) for v in self.legality_mask],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_request(request: PolicyRequest, *, action_words: tuple[str, ...]) -> str:
    words = _canonical_action_words(tuple(action_words))
    if not isinstance(request.mode, str) or request.mode not in ("normal", "hard"):
        raise ProtocolError(f"invalid mode {request.mode!r}")
    turn = _validate_turn(request.turn)
    try:
        rows = [list(row) for row in request.played_rows]
    except TypeError as exc:
        raise ProtocolError("played rows must be [word5, marks] pairs") from exc
    for row in rows:
        if len(row) != 2 or not all(isinstance(v, str) for v in row):
            raise ProtocolError("played rows must be [word5, marks] pairs")
    _check_public_board(request.mode, turn, rows)
    if len(rows) != turn - 1:
        raise ProtocolError(f"turn {turn} requires exactly {turn - 1} played rows")
    if not _is_sha256(request.action_space_sha256):
        raise ProtocolError("action_space_sha256 must be a lowercase 64-hex digest")
    from deep_green.rules import ActionSpace

    if request.action_space_sha256 != ActionSpace.parse(words).digest:
        raise ProtocolError("action-space digest mismatch")
    if len(request.legality_mask) != len(words):
        raise ProtocolError("mask length must equal the action count")
    for value in request.legality_mask:
        if not isinstance(value, bool):
            raise ProtocolError("request legality mask must hold exact bools")
    if not any(request.legality_mask):
        raise ProtocolError("empty legality mask ends the episode; no fallback exists")
    body = {
        "schema": PROTOCOL_SCHEMA,
        "mode": request.mode,
        "turn": turn,
        "played_rows": rows,
        "action_space_sha256": request.action_space_sha256,
        "action_count": len(words),
        "legality_mask": [1 if v else 0 for v in request.legality_mask],
        "request_hash": request.request_hash(action_words=words),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def parse_request(line) -> dict:
    encoded = line.encode("utf-8") if isinstance(line, str) else bytes(line)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ProtocolError(
            f"request line is {len(encoded)} bytes, above {MAX_REQUEST_BYTES}"
        )
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"request is not UTF-8: {exc}") from exc
    try:
        raw = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"malformed JSONL line: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("request must be a JSON object")
    if raw.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"unknown IPC schema {raw.get('schema')!r}")
    unknown = sorted(set(raw) - set(_ALLOWED_REQUEST_FIELDS))
    if unknown:
        for field in unknown:
            lowered = field.lower()
            for marker in _FORBIDDEN_FIELD_MARKERS:
                if marker in lowered:
                    raise ProtocolError(
                        f"forbidden field {field!r} violates the policy firewall"
                    )
        raise ProtocolError(f"unknown fields fail closed: {unknown}")
    missing = sorted(set(_ALLOWED_REQUEST_FIELDS) - set(raw))
    if missing:
        raise ProtocolError(f"missing required fields: {missing}")
    return raw


def validate_parsed_request(raw: dict, *, action_words: tuple[str, ...]) -> PolicyRequest:
    from deep_green.rules import ActionSpace

    mode = raw["mode"]
    if not isinstance(mode, str) or mode not in ("normal", "hard"):
        raise ProtocolError(f"invalid mode {mode!r}")
    turn = _validate_turn(raw["turn"])
    rows = _validate_rows(raw["played_rows"], turn)
    _check_public_board(mode, turn, [list(row) for row in rows])
    count = raw["action_count"]
    if not isinstance(count, int) or isinstance(count, bool):
        raise ProtocolError(f"action_count must be an integer, got {count!r}")
    if count != len(action_words):
        raise ProtocolError("action count mismatch")
    mask = _validate_mask(raw["legality_mask"], len(action_words))
    digest_text = raw["action_space_sha256"]
    if not _is_sha256(digest_text):
        raise ProtocolError("action_space_sha256 must be a lowercase 64-character hex digest")
    expected_action_digest = ActionSpace.parse(action_words).digest
    if digest_text != expected_action_digest:
        raise ProtocolError("action-space digest mismatch")
    request_hash = raw["request_hash"]
    if not _is_sha256(request_hash):
        raise ProtocolError("request_hash must be a lowercase 64-character hex digest")
    request = PolicyRequest(mode, turn, tuple(rows), digest_text, mask)
    if not hmac.compare_digest(
        request_hash, request.request_hash(action_words=action_words)
    ):
        raise ProtocolError("request hash mismatch over public bytes")
    return request
