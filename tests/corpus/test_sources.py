import textwrap

import pytest

from deep_green.corpus import (
    CorpusSpecError,
    SourceSpecError,
    load_corpus_spec,
    load_source_specs,
)

SCOWL_COMMIT = "5ef55f9c42730ebe4394a78b77855468a6f15dd2"
SCOWL_HASH = "933cd8edab9297fc8cb6358d7cb3d05a50371107d1431af61de09eaa875043ac"
WORDNET_HASH = "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52"


def _write_sources(tmp_path, body):
    path = tmp_path / "sources.toml"
    path.write_text(body)
    return path


def _write_corpus(tmp_path, body):
    path = tmp_path / "corpus.toml"
    path.write_text(body)
    return path


def test_load_source_specs_pins_scowl_and_wordnet(tmp_path):
    path = _write_sources(
        tmp_path,
        textwrap.dedent(
            f"""
            [source.scowl_v1]
            tag = "rel-2020.12.07"
            commit = "{SCOWL_COMMIT}"
            url = "https://codeload.github.com/en-wl/wordlist/tar.gz/{SCOWL_COMMIT}"
            download_name = "scowl-v1-rel-2020.12.07-5ef55f9c.tar.gz"
            sha256 = "{SCOWL_HASH}"
            license = "MIT"
            license_url = "https://github.com/en-wl/wordlist/blob/v2/Copyright"
            notice = "SCOWL notices retained; see NOTICE-SCOWL.txt."

            [source.wordnet_3]
            version = "3.0"
            url = "https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
            download_name = "WordNet-3.0.tar.gz"
            sha256 = "{WORDNET_HASH}"
            license = "WordNet 3.0 license"
            license_url = "https://wordnet.princeton.edu/license-and-commercial-use"
            notice = "WordNet notice retained; see NOTICE-WordNet.txt."
            """
        ),
    )
    specs = load_source_specs(path)
    scowl = specs["scowl_v1"]
    assert scowl.tag == "rel-2020.12.07"
    assert scowl.commit == SCOWL_COMMIT
    assert scowl.sha256 == SCOWL_HASH
    assert specs["wordnet_3"].sha256 == WORDNET_HASH
    assert specs["wordnet_3"].version == "3.0"


def test_load_source_specs_rejects_unknown_field_with_name(tmp_path):
    body = textwrap.dedent(
        f"""
        [source.scowl_v1]
        tag = "rel-2020.12.07"
        commit = "{SCOWL_COMMIT}"
        url = "https://example.invalid/scowl.tar.gz"
        download_name = "scowl.tgz"
        sha256 = "{SCOWL_HASH}"
        license = "MIT"
        license_url = "https://example.invalid/license"
        extra_unknown_key = "nope"

        [source.wordnet_3]
        version = "3.0"
        url = "https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
        download_name = "WordNet-3.0.tar.gz"
        sha256 = "{WORDNET_HASH}"
        license = "WordNet 3.0 license"
        license_url = "https://wordnet.princeton.edu/license-and-commercial-use"
        """
    )
    path = _write_sources(tmp_path, body)
    with pytest.raises(SourceSpecError) as err:
        load_source_specs(path)
    message = str(err.value)
    assert "extra_unknown_key" in message
    assert "scowl_v1" in message


def test_load_source_specs_rejects_mutated_hash_naming_the_field(tmp_path):
    bad_hash = ("0" if SCOWL_HASH[0] != "0" else "1") + SCOWL_HASH[1:]
    body = textwrap.dedent(
        f"""
        [source.scowl_v1]
        tag = "rel-2020.12.07"
        commit = "{SCOWL_COMMIT}"
        url = "https://example.invalid/scowl.tar.gz"
        download_name = "scowl.tgz"
        sha256 = "{bad_hash}"
        license = "MIT"
        license_url = "https://example.invalid/license"

        [source.wordnet_3]
        version = "3.0"
        url = "https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
        download_name = "WordNet-3.0.tar.gz"
        sha256 = "{WORDNET_HASH}"
        license = "WordNet 3.0 license"
        license_url = "https://wordnet.princeton.edu/license-and-commercial-use"
        """
    )
    path = _write_sources(tmp_path, body)
    with pytest.raises(SourceSpecError) as err:
        load_source_specs(path)
    assert "sha256" in str(err.value)


def test_load_corpus_spec_rejects_unknown_field_and_hash_drift(tmp_path):
    body = textwrap.dedent(
        f"""
        [corpus]
        benchmark_id = "openwordle-v1"

        [source.scowl_v1]
        tag = "rel-2020.12.07"
        commit = "{SCOWL_COMMIT}"
        url = "https://example.invalid/scowl.tar.gz"
        download_name = "scowl.tgz"
        sha256 = "{SCOWL_HASH[:-1]}z"
        license = "MIT"
        license_url = "https://example.invalid/license"
        notice = "n"

        [source.wordnet_3]
        version = "3.0"
        url = "https://example.invalid/wn.tgz"
        download_name = "wn.tgz"
        sha256 = "{WORDNET_HASH}"
        license = "WordNet 3.0 license"
        license_url = "https://example.invalid/license"
        notice = "n"

        [normalization]
        scowl_sizes_audit = [10, 20, 35]
        scowl_sizes_accepted_max = 60
        dialect = "english"
        encoding = "latin-1"
        strip_marked = true
        lowercase = true
        pattern = "[a-z]{{5}}"

        [answers.primary]
        definition = "scowl_audit_35 ∩ wordnet_five_letter_hex"

        [answers.broad]
        definition = "accepted_guesses ∩ wordnet_five_letter_hex"

        [tiers]
        sizes = [512, 1024, 2048, 2095]

        [splits.ah]
        fractions = {{ train = 0.8, dev = 0.1, test = 0.1 }}
        salt_hex = "6f70656e776f72646c652d7631"

        [excluded_sources]
        names = ["esdb-v2", "coca", "nyt-word-lists", "wordfreq"]

        [unexpected_section]
        surprise = true
        """
    )
    path = _write_corpus(tmp_path, body)
    with pytest.raises(CorpusSpecError) as err:
        load_corpus_spec(path)
    assert "unexpected_section" in str(err.value)


def test_wordnet_download_name_is_canonical():
    from tests.corpus.benchmarks_support import read_repo_sources

    specs = read_repo_sources()
    assert specs["wordnet_3"].download_name == "WordNet-3.0.tar.gz"
    assert specs["scowl_v1"].download_name.startswith("scowl-v1-rel-2020.12.07")


def test_real_repo_ledgers_load_cleanly():
    from tests.corpus.benchmarks_support import load_repo_corpus_spec, read_repo_sources

    specs = read_repo_sources()
    corpus = load_repo_corpus_spec()
    assert specs["scowl_v1"].commit == SCOWL_COMMIT
    assert specs["wordnet_3"].sha256 == WORDNET_HASH
    assert corpus.benchmark_id == "openwordle-v1"


def test_salt_decodes_through_shared_path():
    from tests.corpus.benchmarks_support import decode_byte_salt

    raw = decode_byte_salt("6f70656e776f72646c652d7631")
    assert raw == b"openwordle-v1"
    with pytest.raises(ValueError):
        decode_byte_salt("not-hex-at-all")
