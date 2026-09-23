"""Strict oracle-example records and the leak-free training projection.

An example binds public board state to exact teacher values from a named,
hash-bound certificate. Every leak class the plan names is rejected here:
answer identifiers, posterior features, untyped infinity, unknown fields,
and malformed or non-canonical tie/value sets.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from deep_green.rules import (
    ActionSpace,
    Word5,
    board_from_public_state,
    legal_actions,
    legality_digest,
)

EXAMPLE_SCHEMA = "wordle.oracle-example/v1"
CERTIFICATE_SCHEMA = "wordle.oracle-certificate/v1"
INFINITY = 10**9
_HEX = frozenset("0123456789abcdef")


class ExampleError(ValueError):
    pass


def canonical_example_bytes(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def example_id_for(record: dict) -> str:
    body = dict(record)
    body.pop("example_id", None)
    return hashlib.sha256(canonical_example_bytes(body)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def load_examples(path: Path) -> list[dict]:
    """Parse a JSONL example file; every line must validate."""
    examples = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExampleError(f"line {index + 1}: invalid JSON: {exc}") from None
        try:
            validate_example(payload)
        except ExampleError as exc:
            raise ExampleError(f"line {index + 1}: {exc}") from None
        examples.append(payload)
    if not examples:
        raise ExampleError(f"{path}: no examples found")
    return examples


def generate_root_examples(
    certificate_paths: list[Path], *, frozen_root: Path, repo_root: Path
) -> list[dict]:
    from deep_green.oracle.verifier import verify_certificate

    frozen_root = Path(frozen_root)
    repo_root = Path(repo_root)
    action_path = frozen_root / "corpus" / "words" / "accepted-guesses.txt"
    action_space = ActionSpace.parse(action_path.read_text(encoding="utf-8").split())
    examples = []
    seen_modes = set()
    for certificate_path in certificate_paths:
        certificate_path = Path(certificate_path)
        verify_certificate(
            certificate_path,
            frozen_root=frozen_root,
            vendor_root=repo_root / "vendor" / "alex1770-wordle",
            build_cache=repo_root / "artifacts" / "oracle" / "build-cache",
        )
        certificate_bytes = certificate_path.read_bytes()
        certificate = json.loads(certificate_bytes)
        if certificate.get("schema") != CERTIFICATE_SCHEMA:
            raise ExampleError(f"unsupported certificate {certificate_path}")
        identity = certificate["identity"]
        mode = identity["mode"]
        if mode in seen_modes:
            raise ExampleError(f"duplicate certificate mode {mode!r}")
        seen_modes.add(mode)
        proof = certificate["first_action_proof"]
        actions = proof["actions"]
        if len(actions) != len(action_space.words):
            raise ExampleError("certificate action count disagrees with the action space")
        if any(action["exact"] is not True or action["bound"] >= INFINITY for action in actions):
            raise ExampleError("teacher generation requires finite exact values for every action")
        winner_value = proof["winner_score7_total"]
        ties = sorted(action["word"] for action in actions if action["bound"] == winner_value)
        sparse = [
            {"word": action["word"], "losses": 0, "score7_total": action["bound"]}
            for action in actions
        ]
        state = {"mode": mode, "rows": [], "turn": 1}
        board = board_from_public_state(**state)
        legal_mask = tuple(legal_actions(board, action_space))
        record = {
            "schema": EXAMPLE_SCHEMA,
            "identity": {
                "benchmark_id": identity["benchmark_id"],
                "view_id": identity["view_id"],
                "tier": identity["tier"],
                "rules_sha256": identity["rules_sha256"],
                "prior_id": identity["prior_id"],
                "certificate_sha256": hashlib.sha256(certificate_bytes).hexdigest(),
                "action_space_sha256": identity["action_space_sha256"],
                "exposure_track": identity["track"],
            },
            "public_state": state,
            "legal_mask_digest": legality_digest(legal_mask),
            "optimal_tie_set": ties,
            "sparse_action_values": sparse,
        }
        record["example_id"] = example_id_for(record)
        validate_example(record)
        recompute_legal_mask(record, action_space)
        examples.append(record)
    return sorted(examples, key=lambda record: (record["public_state"]["mode"], record["example_id"]))


def write_examples(path: Path, examples: list[dict]) -> Path:
    if not examples:
        raise ExampleError("refusing to write an empty example set")
    for record in examples:
        validate_example(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_example_bytes(record) + b"\n" for record in examples)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    with open(staging, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def _reject_leak_markers(record: dict) -> None:
    blob = json.dumps(record)
    for marker in ("answer_id", "posterior", "candidate_count", "oracle_value_raw"):
        if marker in record:
            raise ExampleError(f"forbidden field {marker!r} violates the firewall")
    for token in ("Infinity", "-Infinity", "NaN"):
        if token in blob:
            raise ExampleError(f"untyped non-finite value ({token}) is rejected")


def validate_example(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ExampleError("example must be a JSON object")
    _reject_leak_markers(record)
    allowed_top = {
        "schema",
        "example_id",
        "identity",
        "public_state",
        "legal_mask_digest",
        "optimal_tie_set",
        "sparse_action_values",
    }
    unknown = sorted(set(record) - allowed_top)
    if unknown:
        raise ExampleError(f"unknown fields: {unknown}")
    missing = sorted(allowed_top - set(record))
    if missing:
        raise ExampleError(f"missing fields: {missing}")
    if record["schema"] != EXAMPLE_SCHEMA:
        raise ExampleError(f"unknown schema {record['schema']!r}")

    identity = record["identity"]
    required_identity = {
        "benchmark_id",
        "view_id",
        "tier",
        "rules_sha256",
        "prior_id",
        "certificate_sha256",
        "action_space_sha256",
        "exposure_track",
    }
    if not isinstance(identity, dict) or set(identity) != required_identity:
        raise ExampleError("identity block incomplete or has extra fields")
    for text_field in ("benchmark_id", "view_id", "prior_id"):
        if not isinstance(identity[text_field], str) or not identity[text_field]:
            raise ExampleError(f"identity.{text_field} must be a nonempty string")
    for digest_field in (
        "rules_sha256",
        "certificate_sha256",
        "action_space_sha256",
    ):
        if not _is_sha256(identity[digest_field]):
            raise ExampleError(f"identity.{digest_field} must be a lowercase sha256")
    tier = identity["tier"]
    if not isinstance(tier, int) or isinstance(tier, bool) or tier <= 0:
        raise ExampleError("identity.tier must be a positive integer")
    if identity["exposure_track"] != "full-compiled":
        raise ExampleError(
            f"exposure track {identity['exposure_track']!r} is not the compiled track"
        )

    state = record["public_state"]
    if not isinstance(state, dict) or set(state) != {"mode", "rows", "turn"}:
        raise ExampleError("public_state needs exactly mode, turn, rows")
    try:
        board_from_public_state(
            mode=state["mode"], turn=state["turn"], rows=state["rows"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExampleError(f"invalid public_state: {exc}") from None

    tie_set = record["optimal_tie_set"]
    if not isinstance(tie_set, list) or not tie_set:
        raise ExampleError("optimal_tie_set must be a nonempty list")
    for word in tie_set:
        _check_word(word)
    if tie_set != sorted(set(tie_set)):
        raise ExampleError("optimal_tie_set must be unique and canonically sorted")

    sparse = record["sparse_action_values"]
    if not isinstance(sparse, list) or not sparse:
        raise ExampleError("sparse_action_values must be a nonempty list")
    sparse_words = []
    for entry in sparse:
        if not isinstance(entry, dict) or set(entry) != {
            "losses",
            "score7_total",
            "word",
        }:
            raise ExampleError("sparse entries carry exactly word/losses/score7_total")
        _check_word(entry["word"])
        sparse_words.append(entry["word"])
        for name in ("losses", "score7_total"):
            value = entry[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExampleError(f"{name} must be a nonnegative integer")
    if sparse_words != sorted(set(sparse_words)):
        raise ExampleError("sparse_action_values must be unique and canonically sorted")
    if not set(tie_set).issubset(sparse_words):
        raise ExampleError("every optimal action must have an exact sparse value")

    if not _is_sha256(record["legal_mask_digest"]):
        raise ExampleError("legal_mask_digest must be a lowercase sha256")
    if record["example_id"] != example_id_for(record):
        raise ExampleError("example_id does not match the canonical record digest")
    return record


def _check_word(word) -> None:
    if not isinstance(word, str):
        raise ExampleError(f"malformed guess word {word!r}")
    try:
        Word5.parse(word)
    except ValueError as exc:
        raise ExampleError(str(exc)) from None


@dataclass(frozen=True)
class ModelBatch:
    """Projected training batch: encoded public inputs plus tie targets."""

    row_letters: list[list[list[int]]]
    row_marks: list[list[list[int]]]
    row_positions: tuple[int, ...]
    row_masks: list[list[bool]]
    turns: list[int]
    modes: list[tuple[float, float]]
    legal_masks: list[tuple[bool, ...]]
    tie_sets: list[list[str]]
    action_words: tuple[str, ...]

    def to_public_dict(self) -> dict:
        """Serialization used by the leak test: no answer-id field exists."""
        return {
            "row_letters": self.row_letters,
            "row_marks": self.row_marks,
            "row_positions": list(self.row_positions),
            "row_masks": self.row_masks,
            "turns": self.turns,
            "modes": self.modes,
            "legal_masks": [list(mask) for mask in self.legal_masks],
            "tie_sets": self.tie_sets,
            "action_space_sha256": ActionSpace.parse(self.action_words).digest,
        }


def project_batch(examples: list[dict], action_words: tuple[str, ...]) -> ModelBatch:
    """Encode examples into training tensors; audit_answer_id cannot survive."""
    letter_codes = {chr(ord("a") + i): i for i in range(26)}
    mark_codes = {"b": 0, "y": 1, "g": 2}
    mode_vectors = {"normal": (1.0, 0.0), "hard": (0.0, 1.0)}
    action_space = ActionSpace.parse(action_words)
    if action_space.words != action_words:
        raise ExampleError("action words must be in canonical order")
    word_index = {word: i for i, word in enumerate(action_words)}

    batch_rows_letters = []
    batch_rows_marks = []
    row_masks = []
    turns = []
    modes = []
    masks = []
    tie_sets = []
    positions = tuple(range(5))
    for record in examples:
        validate_example(record)
        if record["identity"]["action_space_sha256"] != action_space.digest:
            raise ExampleError("example action-space digest does not match projection")
        state = record["public_state"]
        letters_rows = []
        marks_rows = []
        for guess_text, marks_text in state["rows"]:
            letters_rows.append([letter_codes[c] for c in guess_text])
            marks_rows.append([mark_codes[c] for c in marks_text])
        batch_rows_letters.append(letters_rows)
        batch_rows_marks.append(marks_rows)
        row_masks.append([True] * len(letters_rows))
        turns.append(int(state["turn"]))
        modes.append(mode_vectors[state["mode"]])
        masks.append(recompute_legal_mask(record, action_space))
        ties = []
        for word in record["optimal_tie_set"]:
            if word not in word_index:
                raise ExampleError(f"tie word {word!r} missing from the action space")
            ties.append(word)
        tie_sets.append(ties)
    return ModelBatch(
        row_letters=batch_rows_letters,
        row_marks=batch_rows_marks,
        row_positions=positions,
        row_masks=row_masks,
        turns=turns,
        modes=modes,
        legal_masks=masks,
        tie_sets=tie_sets,
        action_words=action_words,
    )


def recompute_legal_mask(record: dict, action_space: ActionSpace) -> tuple[bool, ...]:
    state = record["public_state"]
    try:
        board = board_from_public_state(
            mode=state["mode"], turn=state["turn"], rows=state["rows"]
        )
        mask = tuple(legal_actions(board, action_space))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExampleError(f"cannot recompute legality: {exc}") from None
    if legality_digest(mask) != record["legal_mask_digest"]:
        raise ExampleError(
            "legal mask recompute disagrees with the stored legal_mask_digest"
        )
    return mask


def hard_negatives(
    *,
    student_logits: dict[str, float],
    teacher_ranks: dict[str, int],
    exact_values: dict[str, tuple[int, int]],
    tie_set: list[str],
    keyed_sample: list[str],
    limit: int = 32,
) -> list[str]:
    """Deterministic negatives: worst exact values, then student logits,
    then teacher ranks, then the keyed tail sample. No optimal word enters."""
    tie = set(tie_set)
    pool = [w for w in student_logits if w not in tie]
    ranked = sorted(
        pool,
        key=lambda w: (
            -exact_values.get(w, (9999, 9999))[0],
            -exact_values.get(w, (9999, 9999))[1],
            -student_logits.get(w, float("-inf")),
            teacher_ranks.get(w, 10**9),
        ),
    )
    tail = [
        word
        for word in keyed_sample
        if word in student_logits and word not in tie and word not in set(ranked)
    ]
    selected = ranked[:limit]
    for word in tail:
        if len(selected) >= limit:
            break
        selected.append(word)
    assert not (set(selected) & tie), "an optimal action entered the negative set"
    return selected
