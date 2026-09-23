"""Build the tiny hash-verified release fixture used by firewall tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_green.backends.storage import save_policy_tensors
from deep_green.model_spec import (
    TENSOR_SPEC_VERSION,
    expected_shapes,
    tensor_names,
)
from deep_green.numeric_policy import encode_lexicon
from deep_green.release import (
    EXPECTED_FILES,
    INFERENCE_CLASS,
    RELEASE_SCHEMA,
)

FROZEN_ROOT = REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen"


def _frozen_file_sha256(relative: str) -> str:
    manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return manifest["files"][relative]


def build_release(output_dir: Path, *, width: int = 64) -> dict:
    """Deterministic tiny policy bundle over the frozen accepted vocabulary."""
    import numpy as np

    rng = np.random.default_rng(20260826)
    shapes = expected_shapes(width)
    tensors = {
        name: rng.normal(0.0, 0.05, size=shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    words = (
        (FROZEN_ROOT / "corpus/words/accepted-guesses.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_policy_tensors(output_dir / "weights.safetensors", tensors, metadata={"width": str(width)})
    lexicon_path = output_dir / "accepted-guesses.txt"
    lexicon_path.write_text("\n".join(words) + "\n", encoding="utf-8")
    letters, positions = encode_lexicon(words)
    np.savez_compressed(output_dir / "lexicon-indices.npz", letters=letters, positions=positions)
    config = {
        "schema": TENSOR_SPEC_VERSION,
        "width": width,
        "parameter_count": int(sum(int(np.prod(s)) for s in shapes.values())),
        "tensor_names": list(tensor_names()),
    }
    config_path = output_dir / "model-config.json"
    config_path.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    files = {}
    for name in sorted(EXPECTED_FILES):
        path = output_dir / name
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "role": EXPECTED_FILES[name],
        }
    manifest = {
        "schema": RELEASE_SCHEMA,
        "identity": {
            "benchmark_id": "openwordle-v1",
            "rules_sha256": _frozen_file_sha256("rules.json"),
            "prior_id": "uniform-v1",
            "action_space_sha256": _frozen_file_sha256(
                "corpus/words/accepted-guesses.txt"
            ),
            "track": "full-compiled",
            "inference_class": INFERENCE_CLASS,
        },
        "files": files,
    }
    manifest_path = output_dir / "payload-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"release_dir": str(output_dir), "files": len(files), "width": width}


def verify_release(release_dir: Path) -> list[str]:
    from deep_green.release import ReleaseError, load_release

    try:
        load_release(Path(release_dir))
    except (ReleaseError, OSError) as exc:
        return [str(exc)]
    return []


if __name__ == "__main__":
    target = REPO_ROOT / "tests/fixtures/release"
    print(build_release(target))
