"""Fetch pinned OpenWordle-v1 sources into data/sources/ behind hash gates.

A payload is written under its canonical download name only after its
sha256 digest matches the ledger. Mismatched payloads are quarantined
under data/sources/quarantine/ and never land.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from corpus_ledger import load_source_specs, sha256_file

LANDING = REPO_ROOT / "data" / "sources" / "openwordle-v1"
QUARANTINE = REPO_ROOT / "data" / "sources" / "quarantine"
LEDGER_PATH = REPO_ROOT / "benchmarks" / "openwordle-v1" / "sources.toml"


def fetch_one(spec, landing: Path) -> Path:
    target = landing / spec.download_name
    if target.is_file() and sha256_file(target) == spec.sha256:
        print(f"OK (cached): {target}")
        return target
    tmp_dir = landing / ".incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    staged = tmp_dir / spec.download_name
    with urllib.request.urlopen(spec.url) as response:
        staged.write_bytes(response.read())
    digest = sha256_file(staged)
    if digest != spec.sha256:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        bad = QUARANTINE / spec.download_name
        staged.rename(bad)
        raise SystemExit(
            f"REFUSED {spec.name}: digest mismatch (expected {spec.sha256}, got {digest}); "
            f"payload held at {bad}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(target)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print(f"OK (fetched): {target}")
    return target


def main(argv=None) -> int:
    specs = load_source_specs(LEDGER_PATH)
    LANDING.mkdir(parents=True, exist_ok=True)
    for name in sorted(specs):
        fetch_one(specs[name], LANDING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
