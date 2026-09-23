import hashlib

import pytest

from tests.corpus import REPO_ROOT
from tests.corpus.conftest import read_candidate_json, read_words

FROZEN_COUNTS = {
    "words/accepted-guesses.txt": 5022,
    "intermediates/scowl-audit-35.txt": 3421,
    "intermediates/wordnet-five-letter-hex.txt": 5358,
    "answers/primary-2095.txt": 2095,
    "answers/broad-3048.txt": 3048,
}

FROZEN_HASHES = {
    "words/accepted-guesses.txt": "3f8ea3e8e93b775a3716979f65bfe3474ed8fbacbbd9394a01fb93606a7ce59c",
    "intermediates/scowl-audit-35.txt": "2b59954212cc86731cb5d2bf8da223055fe451ea3261c980fceaa721c73b97b7",
    "intermediates/wordnet-five-letter-hex.txt": "5e01e57695f9edc4f1b9b1dcbdf2fb40fadf26f229f2962b3617735751d07633",
    "answers/primary-2095.txt": "b56885a24701f73f2e475ee3139c6b89da86efe91abad1f8297947831bb7b490",
    "answers/broad-3048.txt": "7af9b593f500165e834a435969ab9e0c45fa9ea72db294d4d357841930cab312",
}


def _sha256_of_words(words):
    return hashlib.sha256(("\n".join(words) + "\n").encode("utf-8")).hexdigest()


def test_word_files_have_exact_counts_and_hashes(candidate):
    for rel, count in FROZEN_COUNTS.items():
        words = read_words(candidate, rel)
        assert len(words) == count, f"{rel}: {len(words)} != {count}"
        assert words == sorted(words), f"{rel} not sorted"
        assert len(set(words)) == len(words), f"{rel} has duplicates"
        assert _sha256_of_words(words) == FROZEN_HASHES[rel], f"{rel} hash drift"


def test_every_word_is_lowercase_ascii_five_letters(candidate):
    for rel in FROZEN_COUNTS:
        for word in read_words(candidate, rel):
            assert len(word) == 5
            assert word.isascii() and word.islower() and word.isalpha()


def test_primary_is_intersection_of_audit35_and_wordnet(candidate):
    audit = set(read_words(candidate, "intermediates/scowl-audit-35.txt"))
    wordnet = set(read_words(candidate, "intermediates/wordnet-five-letter-hex.txt"))
    primary = read_words(candidate, "answers/primary-2095.txt")
    assert primary == sorted(audit & wordnet)


def test_broad_is_intersection_of_accepted_and_wordnet(candidate):
    accepted = set(read_words(candidate, "words/accepted-guesses.txt"))
    wordnet = set(read_words(candidate, "intermediates/wordnet-five-letter-hex.txt"))
    broad = read_words(candidate, "answers/broad-3048.txt")
    assert broad == sorted(accepted & wordnet)


def test_accepted_contains_primary_and_broad_relationships_hold(candidate):
    accepted = set(read_words(candidate, "words/accepted-guesses.txt"))
    audit = set(read_words(candidate, "intermediates/scowl-audit-35.txt"))
    assert audit < accepted


FORBIDDEN = ("esdb-v2", "coca", "nyt-word-lists", "wordfreq")


def test_manifest_records_all_forbidden_source_exclusions(candidate):
    manifest = read_candidate_json(candidate, "manifest.json")
    names = {entry["source"] for entry in manifest["exclusions"]}
    for forbidden in FORBIDDEN:
        assert forbidden in names, f"manifest missing exclusion: {forbidden}"


def test_builder_refuses_a_forbidden_source(candidate, tmp_path):
    import sys

    from corpus_ledger import load_corpus_spec

    benchmarks_dir = str(REPO_ROOT / "benchmarks")
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    tools_dir = str(REPO_ROOT / "tools" / "corpus")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import build as tools_build

    spec = load_corpus_spec(REPO_ROOT / "benchmarks" / "openwordle-v1" / "corpus.toml")
    poisoned = {
        "benchmark_id": spec.benchmark_id,
        "normalization": spec.normalization,
        "tiers": spec.tiers,
        "splits": spec.splits,
        "sources": {"scowl_v1": {}, "wordnet_3": {}, "wordfreq": {}},
        "excluded_sources": list(FORBIDDEN),
    }
    with pytest.raises(SystemExit) as err:
        tools_build.build_candidate(poisoned, REPO_ROOT, tmp_path / "out")
    message = str(err.value)
    for forbidden in FORBIDDEN:
        assert forbidden in message
