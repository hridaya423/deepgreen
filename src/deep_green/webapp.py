"""Step 8 tournament web app: Assist and Solve over the certified release."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from deep_green.proof import load_public_proof
from deep_green.rules import (
    MAX_ROWS,
    ActionSpace,
    Board,
    Feedback,
    GuessRow,
    Mode,
    Tile,
    Word5,
    legal_actions,
    score_guess,
)

ERROR_SCHEMA = "wordle.web-error/v1"
ASSIST_SCHEMA = "wordle.assist/v1"
SOLVE_SCHEMA = "wordle.solve/v1"
_MAX_SUBMISSION_ID = 128
_WORD_PATTERN = r"^[a-z]{5}$"
_MARKS_PATTERN = r"^[byg]{5}$"


class SandboxPolicy:
    """Owns one sandboxed PolicyRuntime; sees only public board state."""

    def __init__(
        self, release_dir: Path, action_words: tuple[str, ...]
    ) -> None:
        from deep_green.inference import PolicyRuntime

        self._actions = ActionSpace.parse(action_words)
        self._runtime = PolicyRuntime(release_dir, self._actions.words)
        self._lock = threading.Lock()

    def next_guess(self, mode: str, rows: tuple[tuple[str, str], ...]) -> str:
        from deep_green.protocol import PolicyRequest

        mode_enum = Mode(mode)
        board = Board.create(
            mode_enum,
            tuple(
                GuessRow(Word5.parse(guess), Feedback.parse(marks))
                for guess, marks in rows
            ),
            len(rows) + 1,
        )
        mask = legal_actions(board, self._actions)
        request = PolicyRequest(
            mode_enum.value,
            len(rows) + 1,
            tuple((str(g), str(m)) for g, m in rows),
            self._actions.digest,
            tuple(mask),
        )
        with self._lock:
            index = self._runtime.guess(request)
        return self._actions.words[index]

    def shutdown(self) -> None:
        self._runtime.shutdown()


def _word_list(path: Path, *, expected_count: int | None) -> tuple[str, ...]:
    path = Path(path)
    words = tuple(path.read_text(encoding="utf-8").split())
    if expected_count is not None and len(words) != expected_count:
        raise ValueError(
            f"{path} must contain exactly {expected_count} words, got {len(words)}"
        )
    for word in words:
        Word5.parse(word)
    if len(set(words)) != len(words):
        raise ValueError(f"{path} contains duplicate words")
    _verify_manifest_digest(path)
    return words


def _is_sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _verify_manifest_digest(path: Path) -> None:
    resolved = path.resolve()
    manifest_path = next(
        (
            parent / "manifest.json"
            for parent in resolved.parents
            if (parent / "manifest.json").is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise ValueError(f"no manifest.json ancestor for {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"malformed manifest {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest {manifest_path} is not an object")  # noqa: TRY004
    if not isinstance(manifest.get("files"), dict):
        raise ValueError(f"manifest {manifest_path} has no files map")  # noqa: TRY004
    try:
        relative = str(resolved.relative_to(manifest_path.parent))
    except ValueError as exc:
        raise ValueError(
            f"{path} is not inside manifest root {manifest_path.parent}"
        ) from exc
    expected = manifest["files"].get(relative)
    if not _is_sha256_hex(expected):
        raise ValueError(
            f"manifest {manifest_path} has no sha256 entry for {relative}"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path} digest {actual} drifts from manifest {expected}"
        )


class WebError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _AssistCreate(_StrictModel):
    mode: Literal["normal", "hard"]


class _AssistFeedback(_StrictModel):
    expected_turn: int
    pending_guess: str = Field(pattern=_WORD_PATTERN)
    marks: str = Field(pattern=_MARKS_PATTERN)
    submission_id: str = Field(min_length=1, max_length=_MAX_SUBMISSION_ID)


class _SolveRequest(_StrictModel):
    mode: Literal["normal", "hard"]
    word: str = Field(pattern=_WORD_PATTERN)


@dataclass
class _AssistSession:
    session_id: str
    mode: str
    rows: list[tuple[str, str]] = field(default_factory=list)
    pending_guess: str = ""
    consistent: tuple[str, ...] = ()
    consistent_counts: list[int] = field(default_factory=list)
    terminal: bool = False
    outcome: str | None = None
    idempotency: dict = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


def _canonical_payload(body: BaseModel) -> str:
    return hashlib.sha256(
        json.dumps(
            body.model_dump(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _rows_payload(rows: list[tuple[str, str]]) -> list[dict]:
    return [{"guess": guess, "marks": marks} for guess, marks in rows]


def _is_solved(marks: str) -> bool:
    return marks == Tile.CORRECT.value * 5


def _consistent_answers(
    rows: list[tuple[str, str]], answers: tuple[str, ...]
) -> tuple[str, ...]:
    parsed = tuple(
        (Word5.parse(guess), Feedback.parse(marks)) for guess, marks in rows
    )
    return tuple(
        answer
        for answer in answers
        if all(
            score_guess(guess, Word5.parse(answer)) == feedback
            for guess, feedback in parsed
        )
    )


def _marks_of(feedback: Feedback) -> str:
    return "".join(tile.value for tile in feedback.tiles)


def create_app(
    *,
    release_dir: Path,
    proof_path: Path,
    tier_answers_path: Path,
    static_dir: Path | None = None,
    policy=None,
) -> FastAPI:
    release_dir = Path(release_dir)
    proof_path = Path(proof_path)
    tier_answers = _word_list(tier_answers_path, expected_count=None)
    tier_set = frozenset(tier_answers)
    if not tier_set:
        raise ValueError(f"{tier_answers_path} must contain at least one answer")
    tier_sha256 = hashlib.sha256(Path(tier_answers_path).read_bytes()).hexdigest()
    actions = ActionSpace.parse(_read_actions(release_dir))
    action_set = frozenset(actions.words)
    if policy is None:
        policy = SandboxPolicy(release_dir, actions.words)

    store: dict[str, _AssistSession] = {}
    store_lock = threading.Lock()

    def identity() -> dict:
        proof = load_public_proof(proof_path, release_dir)
        result = {"proof_status": proof["status"]}
        if proof["status"] == "VERIFIED":
            result["weights_sha256"] = proof["release"]["weights_sha256"]
            result["manifest_sha256"] = proof["release"]["manifest_sha256"]
        return result

    def certified_scope(proof: dict) -> bool:
        if proof.get("status") != "VERIFIED":
            return False
        certificates = proof.get("certificates")
        claim = proof.get("claim")
        return (
            isinstance(certificates, dict)
            and set(certificates) == {"normal", "hard"}
            and isinstance(claim, dict)
            and claim.get("tier") == len(tier_set)
            and all(
                isinstance(record, dict)
                and record.get("answers_sha256") == tier_sha256
                for record in certificates.values()
            )
        )

    def error_envelope(code: str, message: str) -> dict:
        return {
            "schema": ERROR_SCHEMA,
            "status": "ERROR",
            "code": code,
            "message": message,
            "identity": identity(),
        }

    def assist_payload(session: _AssistSession) -> dict:
        return {
            "schema": ASSIST_SCHEMA,
            "session_id": session.session_id,
            "mode": session.mode,
            "turn": len(session.rows) + 1,
            "rows": _rows_payload(session.rows),
            "pending_guess": session.pending_guess or None
            if not session.terminal
            else None,
            "consistent_count": len(session.consistent),
            "consistent_counts": list(session.consistent_counts),
            "vocabulary_size": len(tier_set),
            "consistent_preview": list(session.consistent[:8]),
            "out_of_scope": len(session.consistent) == 0,
            "terminal": session.terminal,
            "outcome": session.outcome,
            "certification": "EXHIBITION",
            "identity": identity(),
        }

    def get_session(session_id: str, kind: type):
        with store_lock:
            session = store.get(session_id)
        if session is None or not isinstance(session, kind):
            raise WebError(404, "unknown-session", "unknown session")
        return session

    def idempotent(session, body: BaseModel, compute) -> dict:
        digest = _canonical_payload(body)
        prior = session.idempotency.get(body.submission_id)
        if prior is not None:
            prior_digest, prior_response = prior
            if prior_digest != digest:
                raise WebError(
                    409,
                    "submission-conflict",
                    "submission_id was already used with a different request",
                )
            return prior_response
        response = compute()
        session.idempotency[body.submission_id] = (digest, response)
        return response

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutdown = getattr(policy, "shutdown", None)
        if callable(shutdown):
            shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(WebError)
    async def web_error_handler(_request: Request, exc: WebError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(exc.code, exc.message),
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException):
        code = "not-found" if exc.status_code == 404 else "http-error"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(code, str(exc.detail)),
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content=error_envelope(
                "malformed", "request failed strict validation"
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, _exc: Exception):
        return JSONResponse(
            status_code=500,
            content=error_envelope("internal-error", "internal error"),
            headers={"Cache-Control": "no-store"},
        )

    @app.middleware("http")
    async def api_cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/proof")
    async def get_proof():
        proof = load_public_proof(proof_path, release_dir)
        if proof["status"] != "VERIFIED":
            return dict(proof)
        return proof

    @app.post("/api/assist/sessions")
    async def create_assist(body: _AssistCreate):
        pending = policy.next_guess(body.mode, ())
        session = _AssistSession(
            session_id=secrets.token_urlsafe(24),
            mode=body.mode,
            pending_guess=pending,
            consistent=tier_answers,
        )
        with store_lock:
            store[session.session_id] = session
        return assist_payload(session)

    @app.post("/api/assist/sessions/{session_id}/feedback")
    async def assist_feedback(session_id: str, body: _AssistFeedback):
        session = get_session(session_id, _AssistSession)
        with session.lock:
            def compute() -> dict:
                if session.terminal:
                    raise WebError(
                        409, "session-closed", "session is terminal"
                    )
                if body.expected_turn != len(session.rows) + 1:
                    raise WebError(
                        409, "stale-turn", "expected_turn does not match session"
                    )
                if body.pending_guess != session.pending_guess:
                    raise WebError(
                        409,
                        "pending-mismatch",
                        "pending_guess does not match the issued guess",
                    )
                session.rows.append((session.pending_guess, body.marks))
                session.consistent = _consistent_answers(
                    session.rows, tier_answers
                )
                session.consistent_counts.append(len(session.consistent))
                if _is_solved(body.marks):
                    session.terminal = True
                    session.outcome = "WON"
                elif len(session.rows) >= MAX_ROWS:
                    session.terminal = True
                    session.outcome = "LOST"
                elif not session.consistent:
                    session.pending_guess = ""
                else:
                    guess = policy.next_guess(
                        session.mode, tuple(session.rows)
                    )
                    played = {guess for guess, _ in session.rows}
                    if guess in played or (
                        len(session.consistent) <= 2
                        and guess not in session.consistent
                    ):
                        guess = session.consistent[0]
                    session.pending_guess = guess
                return assist_payload(session)

            return idempotent(session, body, compute)

    @app.post("/api/solve")
    async def solve_word(body: _SolveRequest):
        if body.word not in action_set:
            raise WebError(
                422, "unknown-word", "word is outside the certified vocabulary"
            )
        target = Word5.parse(body.word)
        rows: list[tuple[str, str]] = []
        remaining: list[int] = []
        while len(rows) < MAX_ROWS:
            guess = policy.next_guess(body.mode, tuple(rows))
            marks = _marks_of(score_guess(Word5.parse(guess), target))
            rows.append((guess, marks))
            remaining.append(len(_consistent_answers(rows, tier_answers)))
            if _is_solved(marks):
                break
        proof = load_public_proof(proof_path, release_dir)
        if body.word in tier_set and certified_scope(proof):
            certification = "CERTIFIED"
        elif body.word in tier_set:
            certification = "UNVERIFIED"
        else:
            certification = "EXHIBITION"
        return {
            "schema": SOLVE_SCHEMA,
            "mode": body.mode,
            "word": body.word,
            "rows": _rows_payload(rows),
            "turns": len(rows),
            "consistent_counts": remaining,
            "vocabulary_size": len(tier_set),
            "outcome": "SOLVED" if _is_solved(rows[-1][1]) else "UNSOLVED",
            "certification": certification,
            "identity": identity(),
        }

    if static_dir is not None and Path(static_dir).is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    return app


def _read_actions(release_dir: Path) -> tuple[str, ...]:
    return tuple(
        (Path(release_dir) / "accepted-guesses.txt")
        .read_text(encoding="utf-8")
        .split()
    )
