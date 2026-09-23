import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT / "src", REPO_ROOT / "benchmarks"):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)
