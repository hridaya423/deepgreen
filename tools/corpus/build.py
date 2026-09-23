"""Build the OpenWordle-v1 corpus candidate from landed, hash-verified sources.

Reads only data/sources/openwordle-v1/ (hash-gated by tools/corpus/fetch.py)
and the ledgers under benchmarks/openwordle-v1/. Emits a deterministic
candidate tree with word files, tiers, AH splits, notices, and manifest.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
BENCHMARKS = REPO_ROOT / "benchmarks"
for entry in (str(SRC), str(BENCHMARKS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tarfile  # noqa: I001

from corpus_ledger import (
    CorpusSpecError,
    SourceSpecError,
    load_corpus_spec,
    load_source_specs,
    sha256_file,
)
from deep_green import corpus
from deep_green.corpus import (
    FORBIDDEN_SOURCES,
    VerificationError,
    verify_candidate,
    write_candidate_tree,
)

LEDGER_DIR = REPO_ROOT / "benchmarks" / "openwordle-v1"
LANDING = REPO_ROOT / "data" / "sources" / "openwordle-v1"

WORD_SETS = (
    ("words/accepted-guesses.txt", 5022),
    ("intermediates/scowl-audit-35.txt", 3421),
    ("intermediates/wordnet-five-letter-hex.txt", 5358),
    ("answers/primary-2095.txt", 2095),
    ("answers/broad-3048.txt", 3048),
)

AUDIT_SIZES = (10, 20, 35)
ACCEPTED_SIZES = (10, 20, 35, 40, 50, 55, 60)

NOTICES = (
    (
        "NOTICE-SCOWL.txt",
        (
            "OpenWordle-v1 accepted-guess and audit lists derive from SCOWL (and friends)\n"
            "v1 rel-2020.12.07 at commit 5ef55f9c42730ebe4394a78b77855468a6f15dd2.\n"
            "Copyright (c) Kevin Atkinson and SCOWL contributors.\n"
            "The upstream copyright file permits use, copying, modification, distribution,\n"
            "and sale provided the copyright notice is retained in redistributed versions.\n"
            "Upstream license text: https://github.com/en-wl/wordlist/blob/v2/Copyright\n"
        ),
    ),
    (
        "NOTICE-WordNet.txt",
        (
            "OpenWordle-v1 answer-set filtering uses Princeton WordNet 3.0 five-letter\n"
            "single-token lemmas parsed per the WordNet 3.0 file-format specification.\n"
            "WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.\n"
            "This software and database are licensed under the WordNet 3.0 terms reproduced\n"
            "at https://wordnet.princeton.edu/license-and-commercial-use; use, copying,\n"
            "modification, and distribution without fee are permitted so long as this\n"
            "notice and the warranty disclaimer accompany any copy of the database or\n"
            "derived work.\n"
        ),
    ),
)


def _scowl_cumulative(scowl_dir: Path, sizes) -> list[str]:
    words: set[str] = set()
    for size in sizes:
        path = scowl_dir / "scowl" / "final" / f"english-words.{size}"
        if not path.is_file():
            raise SystemExit(f"SCOWL build output missing: {path}")
        for raw_line in path.read_bytes().decode("latin-1").splitlines():
            normalized = corpus.normalize_scowl_line(raw_line)
            if normalized is not None:
                words.add(normalized)
    return sorted(words)


def _require_landed_sources() -> None:
    specs = load_source_specs(LEDGER_DIR / "sources.toml")
    for name in sorted(specs):
        spec = specs[name]
        target = LANDING / spec.download_name
        if not target.is_file() or sha256_file(target) != spec.sha256:
            raise SystemExit(
                f"source {name} missing or hash-mismatched under {LANDING}; "
                "run tools/corpus/fetch.py first"
            )


def _spec_field(spec, name):
    value = getattr(spec, name, None)
    if value is None and isinstance(spec, dict):
        value = spec.get(name)
    return value


def build_candidate(spec, repo_root: Path, output: Path):
    requested_sources = _spec_field(spec, "sources")
    if isinstance(requested_sources, dict) and any(
        name in FORBIDDEN_SOURCES or name not in ("scowl_v1", "wordnet_3")
        for name in requested_sources
    ):
        forbidden_requested = sorted(
            name
            for name in requested_sources
            if name in FORBIDDEN_SOURCES or name not in ("scowl_v1", "wordnet_3")
        )
        if forbidden_requested:
            raise SystemExit(
                f"refusing forbidden or unregistered sources {forbidden_requested}: "
                "the excluded sources are esdb-v2, coca, nyt-word-lists, and wordfreq; "
                "OpenWordle-v1 permits only scowl_v1 and wordnet_3"
            )
    normalization = _spec_field(spec, "normalization")
    if not isinstance(normalization, dict):
        raise SystemExit("corpus spec lacks a valid normalization table")

    recorded_exclusions = [
        {"source": name, "reason": reason}
        for name, reason in (
            (
                "esdb-v2",
                "current ESDB v2 lists are not redistributable for this benchmark; SCOWL v1 is pinned instead",
            ),
            (
                "coca",
                "COCA frequency data has no redistribution license for derived word lists",
            ),
            (
                "nyt-word-lists",
                "NYT Wordle answer/guess lists are first-party NYT data with no open license",
            ),
            (
                "wordfreq",
                "CC BY-SA derivative-data obligations are out of scope for v1; uniform prior instead",
            ),
        )
    ]

    sources_root = repo_root / "data" / "sources" / "openwordle-v1"
    scowl_dir = sources_root / "scowl-v1-rel-2020.12.07"
    if not scowl_dir.is_dir():
        raise SystemExit(
            f"extracted SCOWL tree missing: {scowl_dir}; run tools/corpus/build-linux.sh "
            "(the pinned Linux container build) to produce it"
        )
    wordnet_path = sources_root / "WordNet-3.0.tar.gz"
    if not wordnet_path.is_file():
        raise SystemExit(f"WordNet archive missing: {wordnet_path}")

    accepted = _scowl_cumulative(scowl_dir, tuple(normalization["scowl_sizes_accepted"]))
    audit = _scowl_cumulative(scowl_dir, tuple(normalization["scowl_sizes_audit"]))
    with tarfile.open(wordnet_path, "r:gz") as archive:
        member_names = {
            member.name: member for member in archive.getmembers() if member.name
        }
        lemma_sets = []
        for pos in ("noun", "verb", "adj", "adv"):
            member_name = f"WordNet-3.0/dict/data.{pos}"
            member = member_names.get(member_name)
            if member is None:
                raise SystemExit(f"WordNet archive lacks {member_name}")
            extracted = archive.extractfile(member)
            lemmas = corpus.build_wordnet_lemmas(
                line.decode("utf-8") for line in extracted
            )
            lemma_sets.append(lemmas)

    merged_lemmas = sorted({lemma for lemmas in lemma_sets for lemma in lemmas.five_letter_sorted})

    primary = sorted(set(audit) & set(merged_lemmas))
    broad = sorted(set(accepted) & set(merged_lemmas))
    salt = corpus.decode_salt_bytes(_spec_field(spec, "splits")["ah"].salt_hex)
    ranked = corpus.rank_tier_members(primary, salt)
    folds = corpus.assign_answer_held_out_folds(primary, salt)

    expected_counts = dict(WORD_SETS)
    actual = {
        "words/accepted-guesses.txt": len(accepted),
        "intermediates/scowl-audit-35.txt": len(audit),
        "intermediates/wordnet-five-letter-hex.txt": len(merged_lemmas),
        "answers/primary-2095.txt": len(primary),
        "answers/broad-3048.txt": len(broad),
    }
    drift = {
        rel: (expected, actual[rel])
        for rel, expected in expected_counts.items()
        if actual[rel] != expected
    }
    if drift:
        details = "; ".join(f"{rel}: built {got}, frozen {want}" for rel, (want, got) in drift.items())
        raise SystemExit(
            "built counts disagree with the frozen preregistration (library/"
            f"frozen-corpus-constants.md): {details}"
        )

    artifact = {
        "benchmark_id": _spec_field(spec, "benchmark_id"),
        "counts": {rel: count for rel, count in actual.items()},
        "exclusions": [(entry["source"], entry["reason"]) for entry in recorded_exclusions],
        "notices": list(NOTICES),
        "word_sets": {
            "words/accepted-guesses.txt": accepted,
            "intermediates/scowl-audit-35.txt": audit,
            "intermediates/wordnet-five-letter-hex.txt": merged_lemmas,
            "answers/primary-2095.txt": primary,
            "answers/broad-3048.txt": broad,
            **{f"tiers/tier-{size}.txt": ranked[:size] for size in _spec_field(spec, "tiers")["sizes"]},
            **{f"splits/ah-{name}.txt": list(getattr(folds, name)) for name in corpus.FOLD_NAMES},
        },
    }
    write_candidate_tree(output, artifact)

    return {
        "output": output,
        "counts": actual,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools/corpus/build.py")
    parser.add_argument("--output", required=True, help="candidate output directory")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    _require_landed_sources()
    try:
        spec = load_corpus_spec(LEDGER_DIR / "corpus.toml")
    except (CorpusSpecError, SourceSpecError) as error:
        raise SystemExit(str(error)) from None
    result = build_candidate(spec, REPO_ROOT, output)
    try:
        verify_candidate(output, expected_counts=dict(WORD_SETS))
    except VerificationError as error:
        raise SystemExit(f"built candidate failed self-verification: {error}") from None
    print(f"BUILD OK: {result['output']}")
    for rel, count in sorted(result["counts"].items()):
        print(f"  {rel}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
