"""Deterministic exhaustive census evaluation of the policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from deep_green.inference import (
    PolicyIllegalActionError,
    PolicyRuntime,
    PolicyRuntimeError,
)
from deep_green.protocol import PolicyRequest
from deep_green.rules import (
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

CENSUS_SCHEMA = "wordle.census/v2"
HORIZON_TURNS = 6


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CensusResult:
    answers: int
    losses: int
    score7_total: int
    score7_mean: float
    solved: int
    solved_only_mean: float | None
    histogram: tuple[int, ...]
    p95: int
    worst_score: int
    illegal_actions: int
    trace_digest: str
    per_answer_trace_digests: dict[str, str]
    latency_ms: dict[str, float | int]
    peak_rss_bytes: int

    def as_dict(self) -> dict:
        return {
            "answers": self.answers,
            "losses": self.losses,
            "score7_total": self.score7_total,
            "score7_mean": self.score7_mean,
            "solved": self.solved,
            "solved_only_mean": self.solved_only_mean,
            "histogram": list(self.histogram),
            "p95": self.p95,
            "worst_score": self.worst_score,
            "illegal_actions": self.illegal_actions,
            "trace_digest": self.trace_digest,
            "per_answer_trace_digests": dict(self.per_answer_trace_digests),
            "latency_ms": dict(self.latency_ms),
            "peak_rss_bytes": self.peak_rss_bytes,
        }


def _percentile(sorted_values: list[int | float], q: float):
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def run_census(
    *,
    runtime,
    action_words: tuple[str, ...],
    answer_words: tuple[str, ...],
    mode: str,
) -> CensusResult:
    """Play every answer through the runtime; the evaluator owns the secrets."""
    if mode not in ("normal", "hard"):
        raise EvaluationError(f"invalid mode {mode!r}")
    try:
        action_space = ActionSpace.parse(action_words)
    except ValueError as exc:
        raise EvaluationError(f"action words are not canonical: {exc}") from exc
    if action_space.words != tuple(action_words):
        raise EvaluationError("action words must be in canonical sorted order")
    if not answer_words:
        raise EvaluationError("answer set must be nonempty")
    parsed_answers = []
    seen: set[str] = set()
    for text in answer_words:
        try:
            parsed = Word5.parse(text)
        except ValueError as exc:
            raise EvaluationError(f"invalid answer word {text!r}") from exc
        if parsed.value in seen:
            raise EvaluationError(f"duplicate answer word {parsed.value!r}")
        seen.add(parsed.value)
        parsed_answers.append(parsed.value)

    mode_value = Mode.NORMAL if mode == "normal" else Mode.HARD
    scores: list[int] = []
    latencies_ns: list[int] = []
    per_answer_digests: dict[str, str] = {}
    illegal_actions = 0
    for answer_text in sorted(parsed_answers):
        answer = Word5.parse(answer_text)
        rows: list[GuessRow] = []
        hasher = hashlib.sha256()
        score = 7
        for turn in range(1, HORIZON_TURNS + 1):
            board = Board.create(mode_value, tuple(rows), turn)
            mask = tuple(legal_actions(board, action_space))
            request = PolicyRequest(
                mode=mode,
                turn=turn,
                played_rows=tuple(
                    (row.guess.value, _marks_text(row.feedback)) for row in rows
                ),
                action_space_sha256=action_space.digest,
                legality_mask=mask,
            )
            started = time.perf_counter_ns()
            try:
                guess_index = runtime.guess(request)
            except PolicyIllegalActionError as exc:
                illegal_actions += 1
                raise EvaluationError(
                    f"policy selected an illegal action: {exc}"
                ) from exc
            except PolicyRuntimeError as exc:
                raise EvaluationError(f"policy runtime fault: {exc}") from exc
            latencies_ns.append(time.perf_counter_ns() - started)
            if not isinstance(guess_index, int) or isinstance(guess_index, bool):
                raise EvaluationError(
                    f"runtime returned non-integer index {guess_index!r}"
                )
            if not 0 <= guess_index < len(action_space.words):
                raise EvaluationError(
                    f"runtime returned out-of-range index {guess_index}"
                )
            if not mask[guess_index]:
                illegal_actions += 1
                raise EvaluationError(
                    f"policy selected illegal action index {guess_index}"
                )
            guess = Word5.parse(action_space.words[guess_index])
            feedback = score_guess(guess, answer)
            hasher.update(
                f"{turn}:{guess.value}:{_marks_text(feedback)}\n".encode()
            )
            rows.append(GuessRow(guess, feedback))
            if all(tile is Tile.CORRECT for tile in feedback.tiles):
                score = turn
                break
        scores.append(score)
        per_answer_digests[answer_text] = hasher.hexdigest()

    answers = len(scores)
    losses = sum(1 for score in scores if score == 7)
    solved = answers - losses
    score7_total = sum(scores)
    ordered = sorted(scores)
    latency_ms = [ns / 1_000_000 for ns in latencies_ns]
    latency_sorted = sorted(latency_ms)
    trace_hasher = hashlib.sha256()
    for answer_text in sorted(per_answer_digests):
        trace_hasher.update(
            f"{answer_text}:{per_answer_digests[answer_text]}\n".encode()
        )
    return CensusResult(
        answers=answers,
        losses=losses,
        score7_total=score7_total,
        score7_mean=score7_total / answers,
        solved=solved,
        solved_only_mean=(
            (score7_total - 7 * losses) / solved if solved else None
        ),
        histogram=tuple(
            sum(1 for score in scores if score == bucket) for bucket in range(1, 8)
        ),
        p95=_percentile(ordered, 0.95),
        worst_score=max(scores),
        illegal_actions=illegal_actions,
        trace_digest=trace_hasher.hexdigest(),
        per_answer_trace_digests=per_answer_digests,
        latency_ms={
            "count": len(latency_ms),
            "mean": sum(latency_ms) / len(latency_ms) if latency_ms else 0.0,
            "p50": _percentile(latency_sorted, 0.50) if latency_sorted else 0.0,
            "p95": _percentile(latency_sorted, 0.95) if latency_sorted else 0.0,
            "max": max(latency_ms) if latency_ms else 0.0,
        },
        peak_rss_bytes=int(getattr(runtime, "peak_rss_bytes", 0) or 0),
    )


def _marks_text(feedback: Feedback) -> str:
    text = {Tile.ABSENT: "b", Tile.PRESENT: "y", Tile.CORRECT: "g"}
    return "".join(text[tile] for tile in feedback.tiles)


def require_certificate_parity(result: CensusResult, certificate: dict) -> None:
    """The neural census must equal the certified objective exactly."""
    objective = certificate.get("objective_value")
    if not isinstance(objective, dict) or set(objective) != {
        "losses",
        "score7_total",
        "worst_score",
    }:
        raise EvaluationError("certificate carries no well-formed objective_value")
    for field in ("losses", "score7_total", "worst_score"):
        value = objective[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationError(
                f"certificate objective {field} is not a nonnegative integer"
            )
    mismatched = []
    for field in ("losses", "score7_total", "worst_score"):
        if getattr(result, field) != objective.get(field):
            mismatched.append(
                f"{field}: census {getattr(result, field)} != "
                f"certificate {objective.get(field)}"
            )
    if result.illegal_actions != 0:
        mismatched.append(f"illegal_actions: {result.illegal_actions} != 0")
    if mismatched:
        raise EvaluationError(
            "census diverges from the certified objective: " + "; ".join(mismatched)
        )


def evaluate_certified(
    *,
    release_dir,
    certificate_path,
    frozen_root,
) -> tuple[CensusResult, dict]:
    """Verify the certificate, replay the census, and prove exact parity."""
    from deep_green.oracle.verifier import (
        CertificateError,
        VerificationError,
        verify_certificate,
    )
    from deep_green.release import ReleaseError, load_release

    certificate_path = Path(certificate_path)
    frozen_root = Path(frozen_root)
    release_dir = Path(release_dir)
    try:
        verify_certificate(certificate_path, frozen_root=frozen_root)
    except (CertificateError, VerificationError, OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"certificate verification failed: {exc}") from exc
    certificate_bytes = certificate_path.read_bytes()
    try:
        certificate = json.loads(certificate_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"certificate is not valid JSON: {exc}") from exc
    certificate_sha256 = hashlib.sha256(certificate_bytes).hexdigest()

    try:
        payload = load_release(release_dir)
    except (ReleaseError, OSError) as exc:
        raise EvaluationError(f"release load failed: {exc}") from exc

    cert_identity = certificate["identity"]
    manifest = json.loads(
        (frozen_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_files = manifest["files"]

    def bound_file(relative: str) -> Path:
        expected = manifest_files.get(relative)
        path = frozen_root / relative
        if not isinstance(expected, str) or not path.is_file():
            raise EvaluationError(f"frozen manifest lacks {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise EvaluationError(f"frozen file {relative} fails its manifest digest")
        return path

    views = json.loads(bound_file("views/views.json").read_text(encoding="utf-8"))
    view_items = views["views"] if isinstance(views, dict) else views
    matching = [
        view
        for view in view_items
        if view["view_id"] == cert_identity["view_id"]
        and view["tier"] == cert_identity["tier"]
        and view["mode"] == cert_identity["mode"]
    ]
    if len(matching) != 1:
        raise EvaluationError(
            f"certificate view {cert_identity['view_id']!r} is not unique in the frozen bundle"
        )
    view = matching[0]

    actions_path = bound_file("corpus/words/accepted-guesses.txt")
    answers_path = bound_file(f"corpus/{view['answers_file']}")
    if (
        hashlib.sha256(actions_path.read_bytes()).hexdigest()
        != view["action_space_sha256"]
    ):
        raise EvaluationError("frozen action file fails the view digest")
    if (
        hashlib.sha256(answers_path.read_bytes()).hexdigest()
        != view["answers_sha256"]
    ):
        raise EvaluationError("frozen answers file fails the view digest")

    identity_fields = {
        "benchmark_id": (payload.identity["benchmark_id"], manifest["benchmark_id"]),
        "action_space_sha256": (
            payload.identity["action_space_sha256"],
            view["action_space_sha256"],
        ),
        "rules_sha256": (payload.identity["rules_sha256"], view["rules_sha256"]),
        "prior_id": (payload.identity["prior_id"], view["prior_id"]),
        "track": (payload.identity["track"], view["track"]),
    }
    mismatched = [
        f"{field}: release {release_value!r} != frozen {frozen_value!r}"
        for field, (release_value, frozen_value) in identity_fields.items()
        if release_value != frozen_value
    ]
    if mismatched:
        raise EvaluationError(
            "release identity diverges from the frozen view: "
            + "; ".join(mismatched)
        )

    frozen_words = tuple(actions_path.read_text(encoding="utf-8").split())
    try:
        frozen_space = ActionSpace.parse(frozen_words)
    except ValueError as exc:
        raise EvaluationError(f"frozen action words are malformed: {exc}") from exc
    if frozen_space.words != frozen_words:
        raise EvaluationError("frozen action words are not in canonical order")
    if payload.action_words != frozen_words:
        raise EvaluationError("release action words differ from the frozen vocabulary")

    answer_words = tuple(answers_path.read_text(encoding="utf-8").split())
    try:
        with PolicyRuntime(
            release_dir=release_dir, action_words=payload.action_words
        ) as runtime:
            result = run_census(
                runtime=runtime,
                action_words=payload.action_words,
                answer_words=answer_words,
                mode=view["mode"],
            )
    except PolicyRuntimeError as exc:
        raise EvaluationError(f"policy runtime failed: {exc}") from exc
    require_certificate_parity(result, certificate)

    identity = {
        "benchmark_id": manifest["benchmark_id"],
        "view_id": view["view_id"],
        "tier": view["tier"],
        "mode": view["mode"],
        "answers_sha256": view["answers_sha256"],
        "action_space_sha256": view["action_space_sha256"],
        "rules_sha256": view["rules_sha256"],
        "prior_id": view["prior_id"],
        "track": view["track"],
        "inference_class": payload.identity["inference_class"],
        "artifact_sha256": payload.weights_sha256,
        "manifest_sha256": payload.manifest_sha256,
        "certificate_sha256": certificate_sha256,
    }
    return result, identity


def save_census_report(path: Path, result: CensusResult, identity: dict) -> None:
    import uuid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": CENSUS_SCHEMA, "identity": identity, "result": result.as_dict()}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
