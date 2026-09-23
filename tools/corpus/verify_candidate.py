"""Verify a corpus candidate tree; exits nonzero on any single-byte drift."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_green.corpus import main

if __name__ == "__main__":
    argv = ["verify", *sys.argv[1:]]
    sys.exit(main(argv))
