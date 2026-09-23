"""Pin verification for the vendored alex1770 exact-search solver."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_URL = "https://github.com/alex1770/wordle"
UPSTREAM_COMMIT = "8fd3f1bdf884ff6757c24ad1227dc561fab7787d"
EVENTS_SCHEMA = "alex1770.wordle-events/v1"

COMMIT_FILES = (
    "LICENSE",
    "Makefile",
    "README",
    "compat/error.h",
    "nextword.py",
    "patches/events-v1.patch",
    "UPSTREAM.toml",
    "wordle.cpp",
)


class UpstreamSpecError(ValueError):
    pass


@dataclass(frozen=True)
class UpstreamFile:
    path: str
    sha256: str
    status: str


@dataclass(frozen=True)
class UpstreamSpec:
    upstream_url: str
    commit: str
    license: str
    files: tuple[UpstreamFile, ...]

    @staticmethod
    def load(path: Path) -> UpstreamSpec:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        meta = raw.get("upstream", None)
        if not isinstance(meta, dict) or set(meta) != {"url", "commit", "license"}:
            raise UpstreamSpecError("upstream table must have exactly url, commit, license")
        files_raw = raw.get("file", None)
        if not isinstance(files_raw, list) or not files_raw:
            raise UpstreamSpecError("at least one [[file]] entry is required")
        files: list[UpstreamFile] = []
        for index, entry in enumerate(files_raw):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "status"}:
                raise UpstreamSpecError(f"[[file]] entry {index} must have exactly path, sha256, status")
            status = entry["status"]
            if status not in ("committed", "withheld"):
                raise UpstreamSpecError(f"[[file]] entry {index} has unknown status {status!r}")
            digest = entry["sha256"]
            if digest != "self-referential; excluded by design" and (
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise UpstreamSpecError(f"[[file]] entry {index} has malformed sha256")
            files.append(UpstreamFile(entry["path"], digest, status))
        paths = [f.path for f in files]
        if len(paths) != len(set(paths)):
            raise UpstreamSpecError("duplicate file paths in the pin")
        return UpstreamSpec(meta["url"], meta["commit"], meta["license"], tuple(files))

    def committed_paths(self) -> frozenset[str]:
        return frozenset(f.path for f in self.files if f.status == "committed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_spec(pin_path: Path) -> UpstreamSpec:
    return UpstreamSpec.load(pin_path)


def validate_vendor_tree(vendor_dir: Path, spec: UpstreamSpec) -> None:
    expected = spec.committed_paths()
    present = {
        str(relative.relative_to(vendor_dir))
        for relative in vendor_dir.rglob("*")
        if relative.is_file()
    }
    missing = sorted(expected - present - {"UPSTREAM.toml"})
    if missing:
        raise UpstreamSpecError(f"pinned file(s) missing from vendor tree: {missing}")
    drifted = sorted(
        f.path
        for f in spec.files
        if f.status == "committed"
        and f.path != "UPSTREAM.toml"
        and len(f.sha256) == 64
        and _sha256(vendor_dir / f.path) != f.sha256
    )
    if drifted:
        raise UpstreamSpecError(f"committed file hash drift: {drifted}")
    withheld_present = sorted(
        f.path for f in spec.files if f.status == "withheld" and f.path in present
    )
    if withheld_present:
        raise UpstreamSpecError(
            f"withheld files must not exist in the vendor tree: {withheld_present}"
        )
    unrecorded = sorted(present - expected)
    if unrecorded:
        raise UpstreamSpecError(f"unrecorded files present in vendor tree: {unrecorded}")
