"""Deterministic OpenWordle-v1 corpus construction and verification.

Pure stdlib. The builder tools under tools/corpus/ drive this module; the
module entry point exposes `verify` so a candidate tree can be checked
without importing deep_green.cli.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

WORD_LENGTH = 5
LOWERCASE_ASCII = frozenset("abcdefghijklmnopqrstuvwxyz")

TIER_SIZES = (512, 1024, 2048, 2095)
FOLD_NAMES = ("train", "dev", "test")
FOLD_FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}

FORBIDDEN_SOURCES = ("esdb-v2", "coca", "nyt-word-lists", "wordfreq")

MANIFEST_SCHEMA = "wordle.corpus-manifest/v1"

SCOWL_COMMIT = "5ef55f9c42730ebe4394a78b77855468a6f15dd2"
SCOWL_SHA256 = "933cd8edab9297fc8cb6358d7cb3d05a50371107d1431af61de09eaa875043ac"
WORDNET_SHA256 = "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52"
SALT_HEX = "6f70656e776f72646c652d7631"

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"


def _benchmarks_on_path() -> None:
    import sys

    benchmarks_dir = str(BENCHMARKS_DIR)
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)


class SourceSpecError(ValueError):
    pass


class CorpusSpecError(ValueError):
    pass


def load_source_specs(path=None):
    _benchmarks_on_path()
    from corpus_ledger import load_source_specs as _load

    return _load(BENCHMARKS_DIR / "openwordle-v1" / "sources.toml" if path is None else path)


def load_corpus_spec(path=None):
    _benchmarks_on_path()
    from corpus_ledger import load_corpus_spec as _load

    return _load(BENCHMARKS_DIR / "openwordle-v1" / "corpus.toml" if path is None else path)


SourceSpecError.__module__ = "deep_green.corpus"
CorpusSpecError.__module__ = "deep_green.corpus"


def _alias_error_classes():
    _benchmarks_on_path()
    import corpus_ledger as ledger

    ledger.SourceSpecError = SourceSpecError
    ledger.CorpusSpecError = CorpusSpecError


_alias_error_classes()


def decode_salt_bytes(hex_text: str) -> bytes:
    text = hex_text.strip()
    if len(text) % 2 != 0 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"salt must be an even-length hex string of raw bytes: {hex_text!r}")
    return bytes.fromhex(text)


def normalize_scowl_line(line: str) -> str | None:
    stripped = unicodedata.normalize("NFKD", line)
    decomposed = "".join(
        char for char in stripped if not unicodedata.combining(char)
    )
    lowered = decomposed.lower()
    if any(ord(char) > 127 for char in lowered):
        return None
    if len(lowered) != WORD_LENGTH:
        return None
    if any(char not in LOWERCASE_ASCII for char in lowered):
        return None
    return lowered


def parse_wordnet_lexicon_line(line: str) -> list[tuple[str, bool]]:
    fields = line.split(" ")
    if len(fields) < 4 or len(fields[0]) != 8:
        return []
    try:
        word_count = int(fields[3], 16)
    except ValueError:
        return []
    entries: list[tuple[str, bool]] = []
    for index in range(word_count):
        lemma_field = fields[4 + index * 2]
        base = lemma_field.split("(")[0]
        predicate_only = lemma_field.endswith(")")
        entries.append((base.lower(), predicate_only))
    return entries


@dataclass(frozen=True)
class WordNetLemmas:
    five_letter_sorted: tuple[str, ...]
    sha256: str


def build_wordnet_lemmas(lines) -> WordNetLemmas:
    counts: Counter[str] = Counter()
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith(" "):
            continue
        for lemma, predicate_only in parse_wordnet_lexicon_line(line):
            if predicate_only:
                continue
            if len(lemma) == WORD_LENGTH and all(char in LOWERCASE_ASCII for char in lemma):
                counts[lemma] += 1
    ordered = sorted(counts)
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return WordNetLemmas(tuple(ordered), digest)


def _keyed_rank(words, salt: bytes, domain: bytes) -> list[str]:
    def rank(word: str) -> tuple[str, str]:
        digest = hashlib.sha256(domain + salt + word.encode("utf-8")).hexdigest()
        return (digest, word)

    return sorted(words, key=rank)


def rank_tier_members(primary_answers, salt: bytes, domain: bytes = b"tier") -> list[str]:
    return _keyed_rank(primary_answers, salt, domain)


def assign_answer_held_out_folds(primary_answers, salt: bytes, domain: bytes = b"fold"):
    ranked = _keyed_rank(primary_answers, salt, domain)
    total = len(ranked)
    dev_target = total // 10
    test_target = total // 10
    train_target = total - dev_target - test_target
    floors = {"train": train_target, "dev": dev_target, "test": test_target}
    offset = 0
    assigned = {}
    for name in FOLD_NAMES:
        count = floors[name]
        assigned[name] = sorted(ranked[offset : offset + count])
        offset += count
    assert offset == len(ranked)

    class _Folds:
        pass

    folds = _Folds()
    for name in FOLD_NAMES:
        setattr(folds, name, assigned[name])
    folds._counts = {name: len(assigned[name]) for name in FOLD_NAMES}
    return folds


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    with open(staging, "wb") as handle:
        handle.write(data)
        handle.flush()
    Path(staging).rename(path)


def write_candidate_tree(output: Path, artifact: dict) -> dict:
    files: dict[str, str] = {}
    for rel, words in artifact["word_sets"].items():
        data = ("\n".join(words) + "\n").encode("utf-8")
        _write_atomic(output / rel, data)
        files[rel] = _sha256_bytes(data)
    for notice_rel, body in artifact["notices"]:
        data = body.encode("utf-8")
        (output / notice_rel).write_text(body, encoding="utf-8")
        files[notice_rel] = _sha256_bytes(data)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": artifact["benchmark_id"],
        "scowl_commit": SCOWL_COMMIT,
        "salt_hex": SALT_HEX,
        "counts": artifact["counts"],
        "exclusions": [
            {"source": name, "reason": reason} for name, reason in artifact["exclusions"]
        ],
        "files": dict(sorted(files.items())),
    }
    manifest_data = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    _write_atomic(output / "manifest.json", manifest_data)
    return manifest


def write_candidate_tree_manifest(output: Path, artifact: dict) -> bytes:
    """Regenerate manifest.json bytes for an already-written candidate tree."""

    files: dict[str, str] = {}
    for rel in sorted(artifact["word_sets"]):
        files[rel] = _sha256_bytes((output / rel).read_bytes())
    for notice_rel, _body in artifact.get("notices", ()):
        files[notice_rel] = _sha256_bytes((output / notice_rel).read_bytes())
    counts = {
        rel: len((output / rel).read_text(encoding="utf-8").splitlines())
        for rel in sorted(artifact["word_sets"])
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": artifact["benchmark_id"],
        "scowl_commit": SCOWL_COMMIT,
        "salt_hex": SALT_HEX,
        "counts": counts,
        "exclusions": [
            {"source": name, "reason": reason} for name, reason in artifact["exclusions"]
        ],
        "files": dict(sorted(files.items())),
    }
    return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")


class VerificationError(ValueError):
    pass


REQUIRED_NOTICES = ("NOTICE-SCOWL.txt", "NOTICE-WordNet.txt")


def _resolve_candidate(candidate_dir) -> Path:
    """Accept either a candidate directory or a direct manifest.json path."""

    candidate = Path(candidate_dir)
    if candidate.is_file() and candidate.name == "manifest.json":
        return candidate.parent
    if candidate.name == "manifest.json":
        raise VerificationError(f"missing manifest.json under {candidate.parent}")
    if not (candidate / "manifest.json").is_file():
        raise VerificationError(f"missing manifest.json under {candidate}")
    return candidate


def verify_candidate(candidate_dir, expected_counts=None) -> dict:
    candidate = _resolve_candidate(candidate_dir)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise VerificationError(f"unknown manifest schema: {manifest.get('schema')!r}")
    excluded = sorted(entry["source"] for entry in manifest.get("exclusions", []))
    if excluded != sorted(FORBIDDEN_SOURCES):
        raise VerificationError(
            f"manifest exclusions must be exactly {sorted(FORBIDDEN_SOURCES)}, got {excluded}"
        )
    files = manifest.get("files", {})
    recomputed: dict[str, str] = {}
    for rel, recorded in sorted(files.items()):
        path = candidate / rel
        if not path.is_file():
            raise VerificationError(f"registered file missing: {rel}")
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        if digest != recorded:
            raise VerificationError(f"digest mismatch for {rel}: recorded {recorded}, got {digest}")
        recomputed[rel] = digest
        words = data.decode("utf-8").splitlines()
        if rel in REQUIRED_NOTICES or rel.startswith("NOTICE-"):
            continue
        if rel.startswith(("tiers/", "splits/")):
            if len(words) != len(set(words)):
                raise VerificationError(f"{rel} contains duplicates")
        elif words != sorted(words):
            raise VerificationError(f"{rel} is not sorted")
        for word in words:
            if len(word) != WORD_LENGTH or any(char not in LOWERCASE_ASCII for char in word):
                raise VerificationError(f"{rel} holds non-canonical spelling {word!r}")
    counts = manifest.get("counts", {})
    for notice in REQUIRED_NOTICES:
        if notice not in files:
            raise VerificationError(f"required notice file missing from the manifest: {notice}")
    for rel, declared in counts.items():
        if rel not in recomputed:
            raise VerificationError(f"count references unregistered file: {rel}")
        actual = len((candidate / rel).read_text(encoding="utf-8").splitlines())
        if actual != declared:
            raise VerificationError(f"{rel}: manifest declares {declared}, file holds {actual}")
    if expected_counts:
        for rel, frozen_count in expected_counts.items():
            actual = len((candidate / rel).read_text(encoding="utf-8").splitlines())
            if actual != frozen_count:
                raise VerificationError(f"{rel}: expected {frozen_count} lines, found {actual}")

    def words_of(rel):
        return (candidate / rel).read_text(encoding="utf-8").splitlines()

    accepted = set(words_of("words/accepted-guesses.txt"))
    audit = set(words_of("intermediates/scowl-audit-35.txt"))
    wordnet = set(words_of("intermediates/wordnet-five-letter-hex.txt"))
    primary = set(words_of("answers/primary-2095.txt"))
    broad = set(words_of("answers/broad-3048.txt"))
    if primary != audit & wordnet:
        raise VerificationError("primary answers are not scowl_audit_35 ∩ wordnet_five_letter_hex")
    if broad != accepted & wordnet:
        raise VerificationError("broad answers are not accepted_guesses ∩ wordnet_five_letter_hex")

    salt = decode_salt_bytes(manifest.get("salt_hex", ""))
    ranked = rank_tier_members(sorted(primary), salt)
    if TIER_SIZES[-1] != len(primary):
        raise VerificationError(
            f"outermost tier must equal the primary answer set ({len(primary)})"
        )
    previous: set[str] | None = None
    for size in TIER_SIZES:
        tier_rel = f"tiers/tier-{size}.txt"
        if tier_rel not in files:
            raise VerificationError(f"missing registered tier file: {tier_rel}")
        tier = words_of(tier_rel)
        if len(tier) != size:
            raise VerificationError(f"tier-{size} must hold exactly {size} answers")
        if tier != ranked[:size]:
            raise VerificationError(
                f"tier-{size} does not match the preregistered hash-ranked prefix"
            )
        if previous is not None and not previous < set(tier):
            raise VerificationError(f"tier-{size} breaks strict nesting above its predecessor")
        previous = set(tier)

    fold_counts = {"train": 1677, "dev": 209, "test": 209}
    seen: set[str] = set()
    for name in FOLD_NAMES:
        rel = f"splits/ah-{name}.txt"
        if rel not in files:
            raise VerificationError(f"missing registered split file: {rel}")
        members = words_of(rel)
        if len(members) != fold_counts[name]:
            raise VerificationError(
                f"ah-{name} must hold exactly {fold_counts[name]} answers, got {len(members)}"
            )
        overlap = seen & set(members)
        if overlap:
            raise VerificationError(f"ah-{name} overlaps another fold at {sorted(overlap)[:3]}")
        seen |= set(members)
    if seen != primary:
        missing = sorted(primary - seen)
        raise VerificationError(f"ah folds do not cover the primary set; first miss {missing[:3]}")
    for notice in REQUIRED_NOTICES:
        notice_bytes = (candidate / notice).read_bytes()
        digest = _sha256_bytes(notice_bytes)
        if digest != files[notice]:
            raise VerificationError(
                f"{notice}: registered digest {files[notice]} does not match file bytes ({digest})"
            )
    return {"verified": True, "files": len(recomputed), "counts": counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deep_green.corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a built corpus candidate tree")
    verify_parser.add_argument("candidate", help="path to the candidate directory holding manifest.json")
    args = parser.parse_args(argv)
    if args.command == "verify":
        try:
            report = verify_candidate(args.candidate)
        except (VerificationError, FileNotFoundError, json.JSONDecodeError) as error:
            print(f"VERIFY FAILED: {error}", file=sys.stderr)
            return 1
        print(
            f"VERIFY OK: files={report['files']} "
            + " ".join(f"{key}={value}" for key, value in report["counts"].items())
        )
        return 0
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
