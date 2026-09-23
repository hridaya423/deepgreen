"""OpenWordle-v1 benchmark freeze: protocol pinning, candidate preflight, atomic publish.

The protocol loader fails closed on unknown sections or fields. The preflight
trusts nothing from a candidate manifest: every digest is recomputed from file
bytes and every word/split/tier file is recounted and re-derived before the
freeze may publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROTOCOL_SCHEMA = "wordle.protocol/v1"
BUNDLE_SCHEMA = "wordle.frozen-bundle/v1"
VIEW_SCHEMA = "wordle.view/v1"
AUDIT_LOG = Path(__file__).resolve().parents[2] / "artifacts" / "audit" / "view-access.log"

ALLOWED_SECTIONS: dict[str, frozenset[str]] = {
    "identity": frozenset({"benchmark_id", "protocol_schema", "corpus_manifest_sha256"}),
    "sources": frozenset(
        {"ledger", "scowl_v1_commit", "scowl_v1_sha256", "wordnet_3_version", "wordnet_3_sha256"}
    ),
    "corpus": frozenset(
        {"candidate_dir", "manifest", "salt_hex", "accepted_guesses_count", "primary_answer_count", "broad_answer_count"}
    ),
    "action_space": frozenset({"kind", "count"}),
    "tiers": frozenset({"sizes", "ranking"}),
    "ah_split": frozenset({"fractions", "counts", "train", "dev", "test"}),
    "rules": frozenset(
        {
            "document",
            "ruleset_id",
            "feedback_profile_id",
            "legality_profiles",
            "sha256",
            "normal",
            "hard",
            "legality_profile_id",
        }
    ),
    "prior": frozenset({"document", "prior_id", "distribution", "support", "sha256"}),
    "harness": frozenset({"horizon_turns", "loss_scoring", "objective", "tie_order"}),
    "model": frozenset({"ladder_widths"}),
    "campaign": frozenset({"seeds", "local_time_limit_hours"}),
    "clean_room": frozenset({"build_identity", "builder_image", "base_image_digest", "script"}),
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "sources": ("scowl_v1_commit", "scowl_v1_sha256", "wordnet_3_version", "wordnet_3_sha256"),
    "corpus": (
        "salt_hex",
        "accepted_guesses_count",
        "primary_answer_count",
        "broad_answer_count",
    ),
    "action_space": ("kind", "count"),
    "tiers": ("sizes",),
    "ah_split": ("counts",),
    "rules": ("document", "ruleset_id", "legality_profiles", "sha256"),
    "prior": ("document", "prior_id", "distribution", "support", "sha256"),
    "harness": ("horizon_turns", "loss_scoring", "objective", "tie_order"),
    "model": ("ladder_widths",),
    "campaign": ("seeds", "local_time_limit_hours"),
    "clean_room": ("build_identity",),
}

PINNED_TIERS = (512, 1024, 2048, 2095)
PINNED_FOLD_COUNTS = {"train": 1677, "dev": 209, "test": 209}
PINNED_LADDER = (64, 96, 128, 192, 256, 320, 384)
PINNED_SEEDS = (17, 29, 43, 71, 113)
PINNED_ACTION_COUNT = 5022
PINNED_HORIZON = 6
PINNED_LOSS_SCORING = "turns_to_solve_unsolved_scores_seven"
PINNED_TIE_ORDER = "lowest_canonical_action_index_wins"
PINNED_CAMPAIGN_LIMIT_HOURS = 72
PINNED_MODES = ("normal", "hard")
HARD_LEGALITY_PROFILE_ID = "nyt-standard-hard"
FULL_TRACK = "full-compiled"
AH_TRACK = "answer-held-out-ah-catalog-exposed"
PINNED_TRACKS = (FULL_TRACK, AH_TRACK)
PINNED_OBJECTIVE = (
    "lexicographic_minimize_losses_then_maximize_score7_total_then_minimize_worst_score"
)


def objective_order_key(
    objective: str, *, losses: int, score7_total: int, worst_score: int
) -> tuple[int, int, int]:
    """Order objective values; the frozen wire ID's 'maximize' token is a legacy typo."""
    if objective != PINNED_OBJECTIVE:
        raise ProtocolError(f"unknown objective {objective!r}")
    if min(losses, score7_total, worst_score) < 0:
        raise ValueError("objective values must be nonnegative")
    return losses, score7_total, worst_score


_HEX64 = frozenset("0123456789abcdef")
_SALT_CHARS = frozenset("0123456789abcdefABCDEF")


class ProtocolError(ValueError):
    pass


class PreflightError(ValueError):
    pass


class BundleError(ValueError):
    pass


def _fail(error_type, field, detail):
    raise error_type(f"{field} {detail}")


def _check_digest_field(value, error_type, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value):
        _fail(error_type, field, f"must be 64 lowercase hex characters, got {value!r}")


def decode_salt_bytes(hex_text: str) -> bytes:
    if not isinstance(hex_text, str) or not hex_text or any(c not in _SALT_CHARS for c in hex_text):
        raise ValueError(f"salt must be an even-length hex string of raw bytes: {hex_text!r}")
    if len(hex_text) % 2 != 0:
        raise ValueError(f"salt must be an even-length hex string of raw bytes: {hex_text!r}")
    return bytes.fromhex(hex_text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IdentityPin:
    benchmark_id: str
    corpus_manifest_sha256: str


@dataclass(frozen=True)
class SourcesPin:
    scowl_v1_commit: str
    scowl_v1_sha256: str
    wordnet_3_version: str
    wordnet_3_sha256: str


@dataclass(frozen=True)
class CorpusPin:
    salt_hex: str
    accepted_guesses_count: int
    primary_answer_count: int
    broad_answer_count: int


@dataclass(frozen=True)
class ActionSpacePin:
    kind: str
    count: int


@dataclass(frozen=True)
class TierPins:
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class AhSplitPin:
    counts: dict[str, int]


@dataclass(frozen=True)
class RulesPin:
    document: str
    ruleset_id: str
    legality_profiles: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class PriorPin:
    document: str
    prior_id: str
    distribution: str
    support: str
    sha256: str


@dataclass(frozen=True)
class HarnessPins:
    horizon_turns: int
    loss_scoring: str
    objective: str
    tie_order: str


@dataclass(frozen=True)
class ModelPins:
    ladder_widths: tuple[int, ...]


@dataclass(frozen=True)
class CampaignPins:
    seeds: tuple[int, ...]
    local_time_limit_hours: int


@dataclass(frozen=True)
class CleanRoomPin:
    build_identity: str


@dataclass(frozen=True)
class ModeBinding:
    name: str
    legality_profile_id: str | None
    ruleset_id: str
    rules_sha256: str


@dataclass(frozen=True)
class BenchmarkProtocol:
    identity: IdentityPin
    sources: SourcesPin
    corpus: CorpusPin
    action_space: ActionSpacePin
    tiers: TierPins
    ah_split: AhSplitPin
    rules: RulesPin
    prior: PriorPin
    harness: HarnessPins
    model: ModelPins
    campaign: CampaignPins
    clean_room: CleanRoomPin
    modes: dict[str, ModeBinding]

    @property
    def salt_bytes(self) -> bytes:
        return decode_salt_bytes(self.corpus.salt_hex)


def load_protocol(path) -> BenchmarkProtocol:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ProtocolError(f"{path}: invalid TOML: {error}") from None

    for section in sorted(raw):
        if section not in ALLOWED_SECTIONS:
            _fail(ProtocolError, f"[{section}]", "unknown section")
    for section, allowed in ALLOWED_SECTIONS.items():
        table = raw.get(section)
        if table is None:
            continue
        for key in sorted(table):
            value = table[key]
            if isinstance(value, dict):
                for sub_key in sorted(value):
                    if sub_key not in allowed:
                        _fail(ProtocolError, f"{section}.{key}.{sub_key}", "unknown field")
            elif key not in allowed:
                _fail(ProtocolError, f"{section}.{key}", "unknown field")

    for section, fields in REQUIRED_FIELDS.items():
        table = raw.get(section)
        if table is None:
            _fail(ProtocolError, f"[{section}]", "section missing")
        for field in fields:
            if field not in table:
                _fail(ProtocolError, f"{section}.{field}", "missing")

    identity_raw = raw["identity"]
    for field in ("corpus_manifest_sha256",):
        if field in identity_raw:
            _check_digest_field(identity_raw[field], ProtocolError, f"identity.{field}")
    protocol_schema = identity_raw.get("protocol_schema")
    if protocol_schema is not None and protocol_schema != PROTOCOL_SCHEMA:
        _fail(ProtocolError, "identity.protocol_schema", f"must be {PROTOCOL_SCHEMA!r}")

    sources_raw = raw["sources"]
    for field in ("scowl_v1_sha256", "wordnet_3_sha256"):
        _check_digest_field(sources_raw[field], ProtocolError, f"sources.{field}")

    corpus_raw = raw["corpus"]
    try:
        decode_salt_bytes(corpus_raw["salt_hex"])
    except ValueError as error:
        raise ProtocolError(f"corpus.salt_hex {error}") from None
    for field in ("accepted_guesses_count", "primary_answer_count", "broad_answer_count"):
        if not isinstance(corpus_raw[field], int) or corpus_raw[field] <= 0:
            _fail(ProtocolError, f"corpus.{field}", "must be a positive integer")

    action_raw = raw["action_space"]
    tiers_sizes = tuple(raw["tiers"]["sizes"])
    ah_counts = {str(k): int(v) for k, v in dict(raw["ah_split"]["counts"]).items()}
    rules_raw = raw["rules"]
    _check_digest_field(rules_raw["sha256"], ProtocolError, "rules.sha256")
    prior_raw = raw["prior"]
    _check_digest_field(prior_raw["sha256"], ProtocolError, "prior.sha256")
    ladder = tuple(int(w) for w in raw["model"]["ladder_widths"])
    seeds = tuple(int(s) for s in raw["campaign"]["seeds"])

    if tiers_sizes != PINNED_TIERS:
        _fail(ProtocolError, "tiers.sizes", f"must be exactly {PINNED_TIERS}, got {tiers_sizes}")
    if ah_counts != PINNED_FOLD_COUNTS:
        _fail(ProtocolError, "ah_split.counts", f"must be exactly {PINNED_FOLD_COUNTS}, got {ah_counts}")
    if action_raw["kind"] != "accepted_guesses":
        _fail(ProtocolError, "action_space.kind", "must pin accepted_guesses")
    if int(action_raw["count"]) != PINNED_ACTION_COUNT:
        _fail(ProtocolError, "action_space.count", f"must be exactly {PINNED_ACTION_COUNT}")
    if int(raw["harness"]["horizon_turns"]) != PINNED_HORIZON:
        _fail(ProtocolError, "harness.horizon_turns", f"must be exactly {PINNED_HORIZON}")
    if raw["harness"]["loss_scoring"] != PINNED_LOSS_SCORING:
        _fail(ProtocolError, "harness.loss_scoring", f"must pin {PINNED_LOSS_SCORING!r}")
    if raw["harness"]["objective"] != PINNED_OBJECTIVE:
        _fail(ProtocolError, "harness.objective", "drifts from the pinned lexicographic objective")
    if raw["harness"]["tie_order"] != PINNED_TIE_ORDER:
        _fail(ProtocolError, "harness.tie_order", f"must pin {PINNED_TIE_ORDER!r}")
    if ladder != PINNED_LADDER:
        _fail(ProtocolError, "model.ladder_widths", f"must be exactly {PINNED_LADDER}, got {ladder}")
    if seeds != PINNED_SEEDS:
        _fail(ProtocolError, "campaign.seeds", f"must be exactly {PINNED_SEEDS}, got {seeds}")
    if int(raw["campaign"]["local_time_limit_hours"]) != PINNED_CAMPAIGN_LIMIT_HOURS:
        _fail(
            ProtocolError,
            "campaign.local_time_limit_hours",
            f"must be exactly {PINNED_CAMPAIGN_LIMIT_HOURS}",
        )

    legality_profiles = tuple(str(p) for p in rules_raw["legality_profiles"])
    modes = _mode_bindings(legality_profiles, str(rules_raw["ruleset_id"]), str(rules_raw["sha256"]))

    repo_root = Path(__file__).resolve().parents[2]
    rules_document = repo_root / str(rules_raw["document"])
    if not rules_document.is_file():
        _fail(ProtocolError, f"rules.document {rules_raw['document']}", "does not resolve")
    actual_rules_sha = sha256_bytes(rules_document.read_bytes())
    if str(rules_raw["sha256"]) != actual_rules_sha:
        _fail(
            ProtocolError,
            "rules.sha256",
            f"drifts from the pinned document ({actual_rules_sha})",
        )
    prior_document = repo_root / str(prior_raw["document"])
    if prior_document.is_file():
        actual_prior_sha = sha256_bytes(prior_document.read_bytes())
        if str(prior_raw["sha256"]) != actual_prior_sha:
            _fail(
                ProtocolError,
                "prior.sha256",
                f"drifts from the pinned document ({actual_prior_sha})",
            )

    corpus_manifest_pin = str(identity_raw.get("corpus_manifest_sha256", ""))
    if corpus_manifest_pin:
        candidate_dir = repo_root / str(corpus_raw.get("candidate_dir", ""))
        candidate_manifest = candidate_dir / str(corpus_raw.get("manifest", "manifest.json"))
        if not candidate_manifest.is_file():
            _fail(ProtocolError, "identity.corpus_manifest_sha256", "cannot resolve the candidate manifest to compare against")
        actual_manifest_sha = sha256_bytes(candidate_manifest.read_bytes())
        if corpus_manifest_pin != actual_manifest_sha:
            _fail(
                ProtocolError,
                "identity.corpus_manifest_sha256",
                f"drifts from the recorded candidate manifest ({actual_manifest_sha}); "
                "re-pin the protocol after regenerating the candidate",
            )

    return BenchmarkProtocol(
        identity=IdentityPin(
            benchmark_id=str(identity_raw.get("benchmark_id", "openwordle-v1")),
            corpus_manifest_sha256=str(identity_raw.get("corpus_manifest_sha256", "")),
        ),
        sources=SourcesPin(
            scowl_v1_commit=str(sources_raw["scowl_v1_commit"]),
            scowl_v1_sha256=str(sources_raw["scowl_v1_sha256"]),
            wordnet_3_version=str(sources_raw["wordnet_3_version"]),
            wordnet_3_sha256=str(sources_raw["wordnet_3_sha256"]),
        ),
        corpus=CorpusPin(
            salt_hex=str(corpus_raw["salt_hex"]),
            accepted_guesses_count=int(corpus_raw["accepted_guesses_count"]),
            primary_answer_count=int(corpus_raw["primary_answer_count"]),
            broad_answer_count=int(corpus_raw["broad_answer_count"]),
        ),
        action_space=ActionSpacePin(kind=str(action_raw["kind"]), count=int(action_raw["count"])),
        tiers=TierPins(sizes=tiers_sizes),
        ah_split=AhSplitPin(counts=ah_counts),
        rules=RulesPin(
            document=str(rules_raw["document"]),
            ruleset_id=str(rules_raw["ruleset_id"]),
            legality_profiles=legality_profiles,
            sha256=str(rules_raw["sha256"]),
        ),
        prior=PriorPin(
            document=str(prior_raw["document"]),
            prior_id=str(prior_raw["prior_id"]),
            distribution=str(prior_raw["distribution"]),
            support=str(prior_raw["support"]),
            sha256=str(prior_raw["sha256"]),
        ),
        harness=HarnessPins(
            horizon_turns=int(raw["harness"]["horizon_turns"]),
            loss_scoring=str(raw["harness"]["loss_scoring"]),
            objective=str(raw["harness"]["objective"]),
            tie_order=str(raw["harness"]["tie_order"]),
        ),
        model=ModelPins(ladder_widths=ladder),
        campaign=CampaignPins(seeds=seeds, local_time_limit_hours=int(raw["campaign"]["local_time_limit_hours"])),
        clean_room=CleanRoomPin(build_identity=str(raw["clean_room"]["build_identity"])),
        modes=modes,
    )


def _mode_bindings(legality_profiles, ruleset_id, rules_sha256) -> dict[str, ModeBinding]:
    """Mode names are pinned; hard mode additionally binds the legality profile id."""

    if legality_profiles is None or len(legality_profiles) == 0:
        _fail(
            ProtocolError,
            "rules.legality_profiles",
            "missing; hard mode cannot be resolved by name alone without the legality profile",
        )
    unknown = [
        profile
        for profile in legality_profiles
        if profile not in PINNED_MODES and profile != HARD_LEGALITY_PROFILE_ID
    ]
    if unknown:
        _fail(
            ProtocolError,
            f"rules.legality_profiles mode {unknown[0]!r}",
            f"is outside the pinned set {PINNED_MODES}",
        )
    if HARD_LEGALITY_PROFILE_ID not in legality_profiles:
        _fail(
            ProtocolError,
            "rules.legality_profiles",
            f"hard mode must be identified by legality_profile_id={HARD_LEGALITY_PROFILE_ID!r}",
        )
    bindings: dict[str, ModeBinding] = {}
    for mode in PINNED_MODES:
        if mode == "hard" or (mode == "normal" and mode in legality_profiles):
            profile_id = HARD_LEGALITY_PROFILE_ID if mode == "hard" else None
            bindings[mode] = ModeBinding(
                name=mode,
                legality_profile_id=profile_id,
                ruleset_id=ruleset_id,
                rules_sha256=rules_sha256,
            )
    return bindings


def _keyed_rank(words, domain: bytes, salt: bytes) -> list[str]:
    def rank(word: str):
        return (hashlib.sha256(domain + salt + word.encode("utf-8")).hexdigest(), word)

    return sorted(words, key=rank)


@dataclass(frozen=True)
class PreflightReport:
    digests: dict[str, str]
    recomputed_counts: dict[str, int]
    registered_files: tuple[str, ...]
    notices: tuple[str, ...]
    sources: tuple[str, ...]


def run_preflight(candidate_dir) -> PreflightReport:
    """Recompute everything from candidate bytes; trust no manifest-declared count."""

    from deep_green.corpus import (
        FOLD_NAMES,
        TIER_SIZES,
        VerificationError,
        assign_answer_held_out_folds,
        verify_candidate,
    )

    candidate = Path(candidate_dir)
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        raise PreflightError(f"missing manifest.json under {candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_tree_files = {
        "words/accepted-guesses.txt",
        "intermediates/scowl-audit-35.txt",
        "intermediates/wordnet-five-letter-hex.txt",
        "answers/primary-2095.txt",
        "answers/broad-3048.txt",
        *(f"tiers/tier-{size}.txt" for size in TIER_SIZES),
        *(f"splits/ah-{name}.txt" for name in FOLD_NAMES),
        "NOTICE-SCOWL.txt",
        "NOTICE-WordNet.txt",
    }
    recorded_files: dict[str, str] = dict(manifest.get("files", {}))
    stray = sorted(set(recorded_files) - expected_tree_files)
    missing = sorted(expected_tree_files - set(recorded_files))
    if stray:
        raise PreflightError(f"unregistered output files present in the candidate tree: {stray[:3]}")
    if missing:
        raise PreflightError(f"registered tree files absent from the manifest: {missing[:3]}")

    on_disk = sorted(
        str(item.relative_to(candidate))
        for item in candidate.rglob("*")
        if item.is_file() and item.name != "build.log"
    )
    allowed_disk = set(recorded_files) | {"manifest.json"}
    unexpected_on_disk = sorted(set(on_disk) - allowed_disk)
    if unexpected_on_disk:
        raise PreflightError(
            f"unregistered extra output files under the candidate root: {unexpected_on_disk[:3]}"
        )

    digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    words_by_rel: dict[str, list[str]] = {}
    duplicated: list[str] = []
    for rel in sorted(recorded_files):
        data = (candidate / rel).read_bytes()
        digests[rel] = sha256_bytes(data)
        lines = data.decode("utf-8").splitlines()
        words_by_rel[rel] = lines
        counts[rel] = len(lines)
        if len(lines) != len(set(lines)):
            duplicated.append(rel)
    if duplicated and not any(rel.startswith("words/") for rel in duplicated):
        raise PreflightError(f"registered files hold duplicate entries: {duplicated[:3]}")

    declared_counts: dict[str, int] = {}
    for rel, declared in dict(manifest.get("counts", {})).items():
        declared_counts[rel] = int(declared)
    for rel in sorted(declared_counts):
        if rel not in counts:
            raise PreflightError(f"manifest declares a count for unregistered file {rel}")
        if counts[rel] != declared_counts[rel]:
            raise PreflightError(
                f"recount rejects manifest lie for {rel}: manifest declares "
                f"{declared_counts[rel]}, recount found {counts[rel]}"
            )

    frozen_counts = {
        "words/accepted-guesses.txt": PINNED_ACTION_COUNT,
        "intermediates/scowl-audit-35.txt": 3421,
        "intermediates/wordnet-five-letter-hex.txt": 5358,
        "answers/primary-2095.txt": 2095,
        "answers/broad-3048.txt": 3048,
    }
    for rel, frozen in frozen_counts.items():
        if counts[rel] != frozen:
            raise PreflightError(
                f"{rel}: recomputed count {counts[rel]} disagrees with the frozen preregistration {frozen}"
            )

    accepted = set(words_by_rel["words/accepted-guesses.txt"])
    primary = set(words_by_rel["answers/primary-2095.txt"])
    broad = set(words_by_rel["answers/broad-3048.txt"])
    audit = set(words_by_rel["intermediates/scowl-audit-35.txt"])
    wordnet = set(words_by_rel["intermediates/wordnet-five-letter-hex.txt"])
    if not primary <= accepted or not broad <= accepted:
        raise PreflightError("answer sets must be subsets of the accepted vocabulary")
    if primary != audit & wordnet or broad != accepted & wordnet:
        raise PreflightError("answer sets do not re-derive from their recorded definitions")

    salt_text = manifest.get("salt_hex", "")
    try:
        salt = decode_salt_bytes(salt_text)
    except ValueError as error:
        raise PreflightError(f"salt misinterpretation refused: {error}") from None
    ranked = _keyed_rank(sorted(primary), b"tier", salt)
    previous: set[str] | None = None
    for size in TIER_SIZES:
        rel = f"tiers/tier-{size}.txt"
        tier_lines = words_by_rel[rel]
        if len(tier_lines) != size:
            raise PreflightError(f"tier-{size} must hold exactly {size} answers")
        if tier_lines != ranked[:size]:
            raise PreflightError(
                f"tier-{size} does not match the preregistered hash-ranked prefix recomputed from bytes"
            )
        tier_set = set(tier_lines)
        if previous is not None and not previous < tier_set:
            raise PreflightError(f"tier-{size} breaks strict nesting above its predecessor")
        previous = tier_set

    fold_counts = {"train": 1677, "dev": 209, "test": 209}
    seen: set[str] = set()
    for name in FOLD_NAMES:
        rel = f"splits/ah-{name}.txt"
        members = words_by_rel[rel]
        if len(members) != fold_counts[name]:
            raise PreflightError(
                f"fold {name}: recomputed size {len(members)} must be exactly {fold_counts[name]}"
            )
        overlap = seen & set(members)
        if overlap:
            raise PreflightError(f"fold {name} overlaps another fold at {sorted(overlap)[:3]}")
        seen |= set(members)
        derived = sorted(getattr(assign_answer_held_out_folds(sorted(primary), salt), name))
        if members != derived:
            raise PreflightError(
                f"fold {name} does not match the assignment re-derived from the preregistered bytes"
            )
    if seen != primary:
        raise PreflightError("folds do not partition the primary answer set")

    notices = ("NOTICE-SCOWL.txt", "NOTICE-WordNet.txt")
    for notice in notices:
        if not (candidate / notice).is_file():
            raise PreflightError(f"required notice file missing: {notice}")
        notice_digest = sha256_bytes((candidate / notice).read_bytes())
        if recorded_files.get(notice) != notice_digest:
            raise PreflightError(
                f"{notice}: registered digest {recorded_files.get(notice)} does not match file bytes ({notice_digest})"
            )

    sources_table = manifest.get("sources")
    if sources_table is not None:
        foreign = sorted(set(sources_table) - {"scowl_v1", "wordnet_3"})
        if foreign:
            raise PreflightError(
                f"foreign source entries are not part of this bundle's provenance: {foreign}"
            )

    try:
        verify_candidate(candidate)
    except VerificationError as error:
        raise PreflightError(f"independent module verification rejected the candidate: {error}") from None

    return PreflightReport(
        digests=digests,
        recomputed_counts=dict(counts),
        registered_files=tuple(sorted(recorded_files)),
        notices=notices,
        sources=("scowl_v1", "wordnet_3"),
    )


def view_id_for(answer_base: str, mode: str, track: str) -> str:
    return f"{answer_base}.uniform.{mode}.{track}"


def build_views(protocol: BenchmarkProtocol, candidate_dir) -> list[dict]:
    candidate = Path(candidate_dir)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    action_digest = files["words/accepted-guesses.txt"]
    sealed_fold_size = protocol.ah_split.counts["test"]
    ah_base = f"{protocol.identity.benchmark_id}-ah-test-{sealed_fold_size}"

    views: list[dict] = []
    for size in protocol.tiers.sizes:
        answers_rel = f"tiers/tier-{size}.txt"
        for mode_name, binding in protocol.modes.items():
            views.append(
                {
                    "schema": VIEW_SCHEMA,
                    "view_id": view_id_for(protocol.identity.benchmark_id, mode_name, FULL_TRACK),
                    "tier": size,
                    "track": FULL_TRACK,
                    "mode": mode_name,
                    "answers_file": answers_rel,
                    "answers_sha256": files[answers_rel],
                    "action_space_sha256": action_digest,
                    "ruleset_id": binding.ruleset_id,
                    "rules_sha256": binding.rules_sha256,
                    "legality_profile_id": binding.legality_profile_id,
                    "prior_id": protocol.prior.prior_id,
                    "horizon_turns": protocol.harness.horizon_turns,
                    "loss_scoring": protocol.harness.loss_scoring,
                    "objective": protocol.harness.objective,
                    "tie_order": protocol.harness.tie_order,
                }
            )
    for mode_name, binding in protocol.modes.items():
        views.append(
            {
                "schema": VIEW_SCHEMA,
                "view_id": view_id_for(ah_base, mode_name, AH_TRACK),
                "tier": sealed_fold_size,
                "track": AH_TRACK,
                "mode": mode_name,
                "answers_file": "splits/ah-test.txt",
                "answers_sha256": files["splits/ah-test.txt"],
                "action_space_sha256": action_digest,
                "ruleset_id": binding.ruleset_id,
                "rules_sha256": binding.rules_sha256,
                "legality_profile_id": binding.legality_profile_id,
                "prior_id": protocol.prior.prior_id,
                "horizon_turns": protocol.harness.horizon_turns,
                "loss_scoring": protocol.harness.loss_scoring,
                "objective": protocol.harness.objective,
                "tie_order": protocol.harness.tie_order,
            }
        )
    return views


def write_bundle_files(bundle_dir: Path, protocol: BenchmarkProtocol, candidate_dir, views) -> None:
    """Stage a complete bundle directory without publishing it."""

    candidate = Path(candidate_dir)
    bundle_dir.mkdir(parents=True, exist_ok=False)

    def copy_file(source: Path, rel: str) -> None:
        target = bundle_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    repo_root = Path(__file__).resolve().parents[2]
    copy_file(repo_root / "benchmarks" / "rules" / "openwordle-rules-v1.json", "rules.json")
    copy_file(repo_root / "benchmarks" / "openwordle-v1" / "priors" / "uniform-v1.json", "prior.json")
    copy_file(protocol_document_path(), "protocol.toml")
    copy_file(candidate / "manifest.json", "corpus/manifest.json")
    copy_file(candidate / "NOTICE-SCOWL.txt", "corpus/NOTICE-SCOWL.txt")
    copy_file(candidate / "NOTICE-WordNet.txt", "corpus/NOTICE-WordNet.txt")

    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    for rel in sorted(manifest["files"]):
        copy_file(candidate / rel, f"corpus/{rel}")

    views_dir = bundle_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(views, indent=2, sort_keys=True) + "\n"
    (views_dir / "views.json").write_text(payload, encoding="utf-8")

    files: dict[str, str] = {}
    for item in sorted(bundle_dir.rglob("*")):
        if item.is_file():
            rel = str(item.relative_to(bundle_dir))
            if rel != "manifest.json":
                files[rel] = sha256_bytes(item.read_bytes())
    root_manifest = {
        "schema": BUNDLE_SCHEMA,
        "benchmark_id": protocol.identity.benchmark_id,
        "files": dict(sorted(files.items())),
    }
    manifest_data = json.dumps(root_manifest, indent=2, sort_keys=True).encode("utf-8")
    (bundle_dir / "manifest.json").write_bytes(manifest_data)


def protocol_document_path() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks" / "openwordle-v1" / "protocol.toml"


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_freeze(protocol: BenchmarkProtocol, candidate_dir, staging: Path) -> Path:
    staging = Path(staging)
    run_preflight(candidate_dir)
    views = build_views(protocol, candidate_dir)
    write_bundle_files(staging, protocol, candidate_dir, views)
    for current, _dirs, files in os.walk(staging):
        for name in files:
            path = Path(current) / name
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_directory(Path(current))
    return staging


def publish_freeze(protocol: BenchmarkProtocol, candidate_dir, target: Path) -> Path:
    target = Path(target)
    if target.exists():
        raise BundleError(f"frozen bundle already exists at {target}; refusing to overwrite immutable state")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target.name}-staging-{os.getpid()}-{time.time_ns()}"
    try:
        stage_freeze(protocol, candidate_dir, staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


@dataclass(frozen=True)
class FrozenBundleReport:
    verified: bool
    view_count: int
    views: tuple[dict, ...]
    bundle_digests: dict[str, str]


def _append_view_access_event(view_id: str, outcome: str) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps(
        {
            "ts_unix": round(time.time(), 3),
            "event": "view-access",
            "view_id": view_id,
            "outcome": outcome,
        },
        sort_keys=True,
    )
    fd = os.open(AUDIT_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (entry + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def load_frozen_benchmark(manifest_path, extra_view_paths=()) -> FrozenBundleReport:
    manifest_path = Path(manifest_path)
    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unknown bundle schema: {manifest.get('schema')!r}")

    files: dict[str, str] = dict(manifest.get("files", {}))
    for rel, recorded in sorted(files.items()):
        path = bundle_dir / rel
        if not path.is_file():
            raise BundleError(f"frozen artifact missing: {rel}")
        actual = sha256_bytes(path.read_bytes())
        if actual != recorded:
            raise BundleError(
                f"frozen artifact {rel}: recorded digest {recorded} does not resolve (found {actual})"
            )

    views_rel = "views/views.json"
    if views_rel not in files:
        raise BundleError(f"frozen artifact missing: {views_rel}")
    views = json.loads((bundle_dir / views_rel).read_text(encoding="utf-8"))

    for extra in extra_view_paths:
        record = json.loads(Path(extra).read_text(encoding="utf-8"))
        view_ids = {view["view_id"] for view in views}
        if record.get("view_id") not in view_ids:
            _append_view_access_event(str(record.get("view_id")), "refused-cross-bundle")
            raise BundleError(
                f"view {record.get('view_id')!r} is not part of this bundle; cross-bundle views are refused"
            )
        _check_view_digests(record, bundle_dir, files, strict=False)

    for view in views:
        _check_view_digests(view, bundle_dir, files, strict=True)
        _append_view_access_event(str(view["view_id"]), "loaded")

    return FrozenBundleReport(
        verified=True,
        view_count=len(views),
        views=tuple(views),
        bundle_digests={rel: digest for rel, digest in sorted(files.items())},
    )


def _check_view_digests(view: dict, bundle_dir: Path, files: dict[str, str], *, strict: bool) -> None:
    for field in ("answers_sha256", "action_space_sha256", "rules_sha256"):
        value = view.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value):
            raise BundleError(f"view {view.get('view_id')!r}: {field} is not a sha256 digest")
    answers_rel = view.get("answers_file")
    if not isinstance(answers_rel, str):
        raise BundleError(f"view {view.get('view_id')!r} lacks an answers_file binding")
    candidates = (f"corpus/{answers_rel}", answers_rel)
    resolved = next((rel for rel in candidates if rel in files), None)
    if resolved is None:
        raise BundleError(
            f"view {view.get('view_id')!r}: answers_file {answers_rel} does not resolve inside this bundle"
        )
    actual = sha256_bytes((bundle_dir / resolved).read_bytes())
    if actual != view["answers_sha256"]:
        raise BundleError(
            f"view {view.get('view_id')!r}: answers digest mismatch against {resolved}"
        )


def verify_frozen_bundle(manifest_path) -> FrozenBundleReport:
    report = load_frozen_benchmark(manifest_path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deep_green.benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a published frozen bundle")
    verify_parser.add_argument("frozen", help="path to the frozen manifest.json")
    args = parser.parse_args(argv)
    if args.command == "verify":
        try:
            report = verify_frozen_bundle(args.frozen)
        except BundleError as error:
            print(f"VERIFY FAILED: {error}", file=sys.stderr)
            return 1
        print(f"VERIFY OK: views={report.view_count}")
        return 0
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
