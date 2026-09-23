import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(REPO_ROOT / "src"), str(REPO_ROOT / "benchmarks")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _repo_sources_path():
    return REPO_ROOT / "benchmarks" / "openwordle-v1" / "sources.toml"


def _repo_corpus_path():
    return REPO_ROOT / "benchmarks" / "openwordle-v1" / "corpus.toml"


def read_repo_sources():
    from corpus_ledger import load_source_specs

    return load_source_specs(_repo_sources_path())


def load_repo_corpus_spec():
    from corpus_ledger import load_corpus_spec

    return load_corpus_spec(_repo_corpus_path())


def decode_byte_salt(hex_text):
    from deep_green.corpus import decode_salt_bytes

    return decode_salt_bytes(hex_text)
