"""Independent Python replay verifier for exact oracle certificates.

Recomputes feedback, legality, branches, per-answer scores, losses, score7
totals, worst case, and digests from the certificate's own records plus the
frozen benchmark, trusting nothing about how they were produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from deep_green.rules import (
    ActionSpace,
    Board,
    Feedback,
    GuessRow,
    Mode,
    Tile,
    Word5,
    is_legal_normal,
    is_legal_nyt_standard_hard,
    score_guess,
)

INFINITY = 10**9
CERTIFICATE_SCHEMA = "wordle.oracle-certificate/v1"
_CODE_TO_TILE = {0: Tile.ABSENT, 1: Tile.PRESENT, 2: Tile.CORRECT}


class CertificateError(ValueError):
    pass


class VerificationError(ValueError):
    pass


@dataclass(frozen=True)
class OracleValue:
    losses: int
    score7_total: int
    worst_score: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.losses, self.score7_total, self.worst_score)


@dataclass(frozen=True)
class ReplayStats:
    nodes: int
    answers_reached: int
    total_score: int
    losses: int
    worst_score: int


def _key_of(feedback: Feedback) -> str:
    order = {Tile.ABSENT: "0", Tile.PRESENT: "1", Tile.CORRECT: "2"}
    return "".join(order[tile] for tile in feedback.tiles)


def _is_solved_key(key: str) -> bool:
    return key == "22222"


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def verify_certificate(
    certificate_path: Path,
    *,
    frozen_root: Path,
    vendor_root: Path | None = None,
    build_cache: Path | None = None,
) -> ReplayStats:
    cert = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
    if cert.get("schema") != CERTIFICATE_SCHEMA:
        raise CertificateError(f"certificate schema mismatch: {cert.get('schema')!r}")
    identity = cert.get("identity")
    if not isinstance(identity, dict):
        raise CertificateError("certificate has no identity object")

    manifest = json.loads(
        (frozen_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_files = manifest.get("files")
    if not isinstance(manifest.get("benchmark_id"), str) or not isinstance(manifest_files, dict):
        raise CertificateError("frozen manifest is malformed")

    def verify_manifest_file(relative: str) -> Path:
        expected = manifest_files.get(relative)
        if not _is_sha256(expected):
            raise CertificateError(f"frozen manifest has no valid digest for {relative}")
        path = frozen_root / relative
        if not path.is_file():
            raise CertificateError(f"frozen manifest file missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise VerificationError(f"frozen manifest digest mismatch for {relative}")
        return path

    if identity.get("benchmark_id") != manifest["benchmark_id"]:
        raise CertificateError(
            f"certificate benchmark_id {identity.get('benchmark_id')!r} does not "
            f"match the frozen manifest {manifest['benchmark_id']!r}"
        )

    views = json.loads(verify_manifest_file("views/views.json").read_text(encoding="utf-8"))
    view_items = views["views"] if isinstance(views, dict) and "views" in views else views
    matching = [
        v
        for v in view_items
        if v["view_id"] == identity.get("view_id")
        and v["tier"] == identity.get("tier")
        and v["mode"] == identity.get("mode")
    ]
    if len(matching) != 1:
        raise CertificateError(
            f"view {identity.get('view_id')!r}@{identity.get('tier')}"
            f"/{identity.get('mode')} not uniquely present in the frozen bundle"
        )
    view = matching[0]
    for field in (
        "mode",
        "tier",
        "track",
        "rules_sha256",
        "prior_id",
        "action_space_sha256",
        "answers_sha256",
        "answers_file",
        "horizon_turns",
        "legality_profile_id",
        "tie_order",
        "loss_scoring",
        "objective",
    ):
        if identity.get(field) != view[field]:
            raise CertificateError(
                f"certificate identity field {field!r} does not match the frozen view"
            )

    actions_path = verify_manifest_file("corpus/words/accepted-guesses.txt")
    answers_path = verify_manifest_file(f"corpus/{view['answers_file']}")
    action_words = tuple(
        sorted(set(actions_path.read_text(encoding="utf-8").split()))
    )
    answer_words = tuple(answers_path.read_text(encoding="utf-8").split())
    action_space = ActionSpace.parse(action_words)
    raw_file_digest = hashlib.sha256(actions_path.read_bytes()).hexdigest()
    if raw_file_digest != view["action_space_sha256"]:
        raise VerificationError("frozen accepted vocabulary fails its pinned digest")
    answers_digest = hashlib.sha256(answers_path.read_bytes()).hexdigest()
    if answers_digest != view["answers_sha256"]:
        raise VerificationError("answer-set digest drift against the frozen bundle")

    binding = cert.get("binding")
    if not isinstance(binding, dict) or set(binding) != {
        "action_space_sha256",
        "answers_sha256",
        "action_space_canonical_digest",
        "answers_canonical_digest",
        "solver_source_sha256",
        "patch_sha256",
        "compiler_id",
        "executable_sha256",
    }:
        raise CertificateError("certificate binding has missing or unknown fields")
    for field in (
        "action_space_sha256",
        "answers_sha256",
        "action_space_canonical_digest",
        "answers_canonical_digest",
        "solver_source_sha256",
        "patch_sha256",
        "executable_sha256",
    ):
        if not _is_sha256(binding[field]):
            raise CertificateError(f"binding field {field!r} is not a sha256 hex digest")
    if not isinstance(binding["compiler_id"], str) or not binding["compiler_id"]:
        raise CertificateError("binding compiler_id must be a nonempty string")
    expected_binding = {
        "action_space_sha256": raw_file_digest,
        "answers_sha256": answers_digest,
        "action_space_canonical_digest": action_space.digest,
        "answers_canonical_digest": hashlib.sha256(
            "\n".join(answer_words).encode("utf-8")
        ).hexdigest(),
    }
    for field, expected in expected_binding.items():
        if binding[field] != expected:
            raise VerificationError(f"binding {field} does not match the frozen corpus")
    if vendor_root is None:
        vendor_root = frozen_root.parents[2] / "vendor" / "alex1770-wordle"
    if not Path(vendor_root).is_dir():
        raise CertificateError(f"vendor root missing: {vendor_root}")
    for relative, field in (
        ("wordle.cpp", "solver_source_sha256"),
        (Path("patches") / "events-v1.patch", "patch_sha256"),
    ):
        current = Path(vendor_root) / relative
        if not current.is_file():
            raise CertificateError(f"vendor file missing: {relative}")
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
        if digest != binding[field]:
            raise VerificationError(
                f"vendor file {relative} hash {digest} drifts from the "
                f"certificate binding {binding[field]}"
            )

    if build_cache is None:
        build_cache = (
            frozen_root.parents[2] / "artifacts" / "oracle" / "build-cache"
        )
    from deep_green.oracle.build import (
        OracleBuildError,
        inspect_certificate_solver,
    )

    try:
        inspect_certificate_solver(
            Path(vendor_root), Path(build_cache), binding
        )
    except OracleBuildError as exc:
        raise VerificationError(
            f"solver build identity could not be established: {exc}"
        ) from exc

    proof = cert.get("first_action_proof")
    if not isinstance(proof, dict) or set(proof) != {
        "declared_actions",
        "winner_index",
        "winner_word",
        "winner_score7_total",
        "actions",
    }:
        raise CertificateError("certificate has no well-formed first_action_proof")
    declared = proof["declared_actions"]
    if not isinstance(declared, int) or isinstance(declared, bool) or declared != len(
        action_space.words
    ):
        raise CertificateError("first_action_proof declared_actions mismatch")
    winner_index = proof["winner_index"]
    winner_word = proof["winner_word"]
    winner_total = proof["winner_score7_total"]
    if (
        not isinstance(winner_index, int)
        or isinstance(winner_index, bool)
        or not 0 <= winner_index < declared
        or not isinstance(winner_word, str)
        or not isinstance(winner_total, int)
        or isinstance(winner_total, bool)
        or winner_total < 0
    ):
        raise CertificateError("first_action_proof winner fields are malformed")
    records = proof["actions"]
    if not isinstance(records, list):
        raise CertificateError("first_action_proof actions must be a list")
    by_index: list[dict | None] = [None] * declared
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "action_index",
            "word",
            "bound",
            "exact",
        }:
            raise CertificateError("first_action_proof action record is malformed")
        index = record["action_index"]
        word = record["word"]
        bound = record["bound"]
        exact = record["exact"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < declared
        ):
            raise CertificateError(
                f"first_action_proof action_index {index!r} out of range"
            )
        if by_index[index] is not None:
            raise CertificateError(
                f"first_action_proof duplicates action_index {index}"
            )
        if not isinstance(word, str) or word != action_space.words[index]:
            raise CertificateError(
                f"first_action_proof word {word!r} does not match canonical "
                f"action {index} {action_space.words[index]!r}"
            )
        if (
            not isinstance(bound, int)
            or isinstance(bound, bool)
            or bound < 0
            or not isinstance(exact, bool)
        ):
            raise CertificateError("first_action_proof action bound/exact malformed")
        by_index[index] = record
    if any(record is None for record in by_index):
        raise CertificateError("first_action_proof does not cover every action")
    winner_record = by_index[winner_index]
    assert winner_record is not None
    if winner_record["word"] != winner_word:
        raise CertificateError("first_action_proof winner word/index are inconsistent")
    if winner_record["exact"] is not True:
        raise CertificateError("first_action_proof winner lacks an exact bound")
    if winner_record["bound"] != winner_total:
        raise CertificateError(
            "first_action_proof winner bound does not equal winner_score7_total"
        )
    optimal_indices = [
        record["action_index"]
        for record in by_index
        if record["exact"] is True and record["bound"] == winner_total
    ]
    if not optimal_indices or winner_index != min(optimal_indices):
        raise CertificateError(
            "first_action_proof winner violates canonical tie order"
        )
    for record in by_index:
        assert record is not None
        if record["bound"] < winner_total:
            raise VerificationError(
                f"action {record['word']!r} bound {record['bound']} undercuts the "
                f"claimed winner total {winner_total}; the winner is unproven"
            )

    search = cert.get("search")
    if not isinstance(search, dict):
        raise CertificateError("certificate has no search record")
    if search.get("best_first_word") != winner_word:
        raise VerificationError(
            f"search best_first_word {search.get('best_first_word')!r} does not "
            f"match the proven winner {winner_word!r}"
        )
    if search.get("exhaustive") is not True:
        raise CertificateError("certificate claims a non-exhaustive search")

    mode = Mode.NORMAL if view["mode"] == "normal" else Mode.HARD
    horizon = int(view["horizon_turns"])
    tree = cert.get("policy_tree")
    if not isinstance(tree, dict) or not tree:
        raise CertificateError("certificate has no policy_tree")

    if tree.get("guess") != winner_word:
        raise VerificationError(
            f"policy tree root guess {tree.get('guess')!r} does not match the "
            f"proven winner {winner_word!r}"
        )
    legality_fn = is_legal_nyt_standard_hard if mode is Mode.HARD else is_legal_normal
    stats = ReplayStats(0, 0, 0, 0, 0)

    def walk(node: dict, rows: tuple[GuessRow, ...], remaining: tuple[str, ...]) -> None:
        nonlocal stats
        turn = len(rows) + 1
        if turn > horizon:
            raise VerificationError(f"tree exceeds the {horizon}-turn horizon")
        guess_text = node.get("guess")
        if not isinstance(guess_text, str):
            raise VerificationError(f"branch node missing guess word at turn {turn}")
        guess_word = Word5.parse(guess_text)
        board = Board.create(mode, rows, turn)
        legal = legality_fn(board, action_space.words)
        try:
            guess_index = action_space.words.index(guess_word.value)
        except ValueError as exc:
            raise VerificationError(
                f"guess {guess_word.value!r} is outside the frozen action vocabulary"
            ) from exc
        if not legal[guess_index]:
            raise VerificationError(
                f"guess {guess_word.value!r} is illegal under {mode.value} rules at turn {turn}"
            )
        stats = ReplayStats(
            stats.nodes + 1,
            stats.answers_reached,
            stats.total_score,
            stats.losses,
            stats.worst_score,
        )
        children = node.get("children", {})
        if not isinstance(children, dict):
            raise VerificationError("children must be a mapping keyed by tile codes")
        partition: dict[str, list[str]] = {}
        for answer in remaining:
            key = _key_of(score_guess(guess_word, Word5.parse(answer)))
            partition.setdefault(key, []).append(answer)
        covered_keys = set(children.keys())
        missing_classes = set(partition.keys()) - covered_keys
        if missing_classes:
            raise VerificationError(
                f"child partition incomplete at turn {turn}: classes "
                f"{sorted(missing_classes)} have no child"
            )
        extra_classes = covered_keys - set(partition.keys())
        if extra_classes:
            raise VerificationError(
                f"child partition contains impossible feedback classes {sorted(extra_classes)} "
                f"at turn {turn}"
            )
        for key, child in children.items():
            group = partition[key]
            terminal = child.get("terminal")
            if _is_solved_key(key):
                solved_turn = int(child.get("turn", turn))
                stats = ReplayStats(
                    stats.nodes + len(group),
                    stats.answers_reached + len(group),
                    stats.total_score + solved_turn * len(group),
                    stats.losses,
                    max(stats.worst_score, solved_turn),
                )
                continue
            if terminal == "loss":
                stats = ReplayStats(
                    stats.nodes,
                    stats.answers_reached + len(group),
                    stats.total_score,
                    stats.losses + len(group),
                    stats.worst_score,
                )
                continue
            walk(
                child,
                rows + (GuessRow(guess_word, Feedback(tuple(_CODE_TO_TILE[int(c)] for c in key))),),
                tuple(group),
            )

    walk(tree, (), answer_words)
    if stats.answers_reached != len(answer_words):
        raise VerificationError(
            f"tree reaches only {stats.answers_reached} of {len(answer_words)} answers"
        )
    if winner_total < INFINITY and stats.losses != 0:
        raise VerificationError(
            f"tree records {stats.losses} losses for a claimed finite exact policy"
        )

    claimed_raw = cert.get("objective_value")
    if not isinstance(claimed_raw, dict) or set(claimed_raw) != {
        "losses",
        "score7_total",
        "worst_score",
    }:
        raise CertificateError("objective_value malformed")
    claimed = OracleValue(
        int(claimed_raw["losses"]),
        int(claimed_raw["score7_total"]),
        int(claimed_raw["worst_score"]),
    )
    recomputed = OracleValue(
        stats.losses,
        stats.total_score + 7 * stats.losses,
        stats.worst_score if stats.losses == 0 else 7,
    )
    if claimed.losses != stats.losses:
        raise VerificationError(
            f"losses mismatch: claimed {claimed.losses}, recomputed {stats.losses}"
        )
    if claimed.score7_total != recomputed.score7_total:
        raise VerificationError(
            f"score7_total mismatch: claimed {claimed.score7_total}, "
            f"recomputed {recomputed.score7_total}"
        )
    if recomputed.score7_total != winner_total:
        raise VerificationError(
            f"recomputed score7_total {recomputed.score7_total} does not match the "
            f"proven winner bound {winner_total}"
        )
    if claimed.worst_score != recomputed.worst_score:
        raise VerificationError(
            f"worst_score mismatch: claimed {claimed.worst_score}, "
            f"recomputed {recomputed.worst_score}"
        )
    return stats
