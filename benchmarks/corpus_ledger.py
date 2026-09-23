"""Typed ledgers for OpenWordle-v1 sources and corpus recipe.

Loaded by the fetcher, the builder, the Task 5 protocol, and the validators.
Unknown fields and malformed digests fail closed with the offending field named.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_PINNED_SOURCE_DIGESTS = {
    "scowl_v1": "933cd8edab9297fc8cb6358d7cb3d05a50371107d1431af61de09eaa875043ac",
    "wordnet_3": "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52",
}

_SOURCE_FIELDS = (
    "tag",
    "commit",
    "version",
    "url",
    "download_name",
    "sha256",
    "license",
    "license_url",
    "notice",
)

_CORPUS_SECTIONS = {
    "corpus": {"benchmark_id"},
    "source": None,
    "normalization": {
        "scowl_sizes_audit",
        "scowl_sizes_accepted",
        "dialect",
        "encoding",
        "strip_marked",
        "lowercase",
        "pattern",
    },
    "answers": {"primary", "broad"},
    "tiers": {"sizes", "ranking"},
    "splits": {"ah"},
    "excluded_sources": {"names"},
}

_SALT_HEX = re.compile(r"^(?:[0-9a-fA-F]{2})+$")


class SourceSpecError(ValueError):
    pass


class CorpusSpecError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    tag: str | None
    commit: str | None
    version: str | None
    url: str
    download_name: str
    sha256: str
    license: str
    license_url: str
    notice: str


@dataclass(frozen=True)
class AhSplitSpec:
    fractions: dict[str, float]
    salt_hex: str


@dataclass(frozen=True)
class AnswerSetSpec:
    definition: str


@dataclass(frozen=True)
class CorpusSpec:
    benchmark_id: str
    normalization: dict
    answers: dict[str, AnswerSetSpec]
    tiers: dict
    splits: dict[str, AhSplitSpec]
    sources: dict[str, SourceSpec]
    excluded_sources: tuple[str, ...]


def _fail(error_type, path, field, detail):
    raise error_type(f"{path}: field '{field}' {detail}")


def _check_digest(value, error_type, path, field):
    if not isinstance(value, str) or not _HEX64.match(value):
        _fail(error_type, path, field, f"must be 64 lowercase hex characters, got {value!r}")


def _parse_source_table(sources, error_type, path):
    if not isinstance(sources, dict) or not sources:
        _fail(error_type, path, "[source]", "section missing or empty")
    specs: dict[str, SourceSpec] = {}
    for name, fields in sources.items():
        if not isinstance(fields, dict):
            _fail(error_type, path, f"source.{name}", "must be a table")
        for key in sorted(fields):
            if key not in _SOURCE_FIELDS:
                _fail(error_type, path, f"source.{name}.{key}", "unknown field")
        for required in ("url", "download_name", "sha256", "license", "license_url"):
            if required not in fields:
                _fail(error_type, path, f"source.{name}.{required}", "missing")
        _check_digest(fields["sha256"], error_type, path, f"source.{name}.sha256")
        expected = _PINNED_SOURCE_DIGESTS.get(name)
        if expected is not None and str(fields["sha256"]) != expected:
            _fail(
                error_type,
                path,
                f"source.{name}.sha256",
                f"drifts from the pinned digest {expected}",
            )
        specs[name] = SourceSpec(
            name=name,
            tag=fields.get("tag"),
            commit=fields.get("commit"),
            version=fields.get("version"),
            url=str(fields["url"]),
            download_name=str(fields["download_name"]),
            sha256=str(fields["sha256"]),
            license=str(fields["license"]),
            license_url=str(fields["license_url"]),
            notice=str(fields.get("notice", "")),
        )
    return specs


def load_source_specs(path) -> dict[str, SourceSpec]:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return _parse_source_table(raw.get("source"), SourceSpecError, path)


def decode_salt_bytes(hex_text: str) -> bytes:
    if not isinstance(hex_text, str) or not _SALT_HEX.match(hex_text):
        raise ValueError("salt must be an even-length hex string of raw bytes")
    return bytes.fromhex(hex_text)


def load_corpus_spec(path) -> CorpusSpec:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    for section in sorted(raw):
        if section not in _CORPUS_SECTIONS:
            _fail(CorpusSpecError, path, f"[{section}]", "unknown section")
    corpus = raw.get("corpus")
    if not isinstance(corpus, dict):
        _fail(CorpusSpecError, path, "[corpus]", "section missing")
    for key in sorted(corpus):
        if key not in _CORPUS_SECTIONS["corpus"]:
            _fail(CorpusSpecError, path, f"corpus.{key}", "unknown field")
    if "benchmark_id" not in corpus:
        _fail(CorpusSpecError, path, "corpus.benchmark_id", "missing")

    normalization = raw.get("normalization")
    if not isinstance(normalization, dict):
        _fail(CorpusSpecError, path, "[normalization]", "section missing")
    for key in sorted(normalization):
        if key not in _CORPUS_SECTIONS["normalization"]:
            _fail(CorpusSpecError, path, f"normalization.{key}", "unknown field")

    answers_raw = raw.get("answers", {})
    for key in sorted(answers_raw):
        if key not in _CORPUS_SECTIONS["answers"]:
            _fail(CorpusSpecError, path, f"answers.{key}", "unknown field")
    answers = {}
    for name, fields in answers_raw.items():
        if "definition" not in fields:
            _fail(CorpusSpecError, path, f"answers.{name}.definition", "missing")
        answers[name] = AnswerSetSpec(definition=str(fields["definition"]))

    tiers = raw.get("tiers", {})
    for key in sorted(tiers):
        if key not in _CORPUS_SECTIONS["tiers"]:
            _fail(CorpusSpecError, path, f"tiers.{key}", "unknown field")

    splits_raw = raw.get("splits", {})
    for key in sorted(splits_raw):
        if key not in _CORPUS_SECTIONS["splits"]:
            _fail(CorpusSpecError, path, f"splits.{key}", "unknown field")
    splits: dict[str, AhSplitSpec] = {}
    for name, fields in splits_raw.items():
        for key in sorted(fields):
            if key not in ("fractions", "salt_hex", "ranking", "interpretation"):
                _fail(CorpusSpecError, path, f"splits.{name}.{key}", "unknown field")
        salt_hex = fields.get("salt_hex")
        try:
            decode_salt_bytes(salt_hex)
        except ValueError:
            _fail(CorpusSpecError, path, f"splits.{name}.salt_hex", "must be hex-encoded raw bytes")
        splits[name] = AhSplitSpec(
            fractions={str(k): float(v) for k, v in fields.get("fractions", {}).items()},
            salt_hex=str(salt_hex),
        )

    excluded = raw.get("excluded_sources", {})
    for key in sorted(excluded):
        if key not in _CORPUS_SECTIONS["excluded_sources"]:
            _fail(CorpusSpecError, path, f"excluded_sources.{key}", "unknown field")
    excluded_names = tuple(str(name) for name in excluded.get("names", ()))

    sources_section = raw.get("source")
    sources: dict[str, SourceSpec] = {}
    if sources_section is not None:
        sources = _parse_source_table(sources_section, CorpusSpecError, path)

    return CorpusSpec(
        benchmark_id=str(corpus["benchmark_id"]),
        normalization=dict(normalization),
        answers=answers,
        tiers=dict(tiers),
        splits=splits,
        sources=sources,
        excluded_sources=excluded_names,
    )


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
