"""Release accounting, nine-gate verification, clean reproduction, and sealing.

The candidate tree is assembled from an explicit allowlist only; the
verification receipt names every gate exactly once; clean reproduction runs the
committed tree in a detached worktree; sealing emits a deterministic archive.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

from deep_green.release import EXPECTED_FILES, MANIFEST_NAME, load_release

CANDIDATE_SCHEMA = "wordle.release-candidate/v1"
VERIFY_SCHEMA = "wordle.release-verification/v1"
CLEAN_SCHEMA = "wordle.clean-reproduction/v1"
SEAL_SCHEMA = "wordle.release-seal/v1"
GATE_NAMES = (
    "benchmark",
    "oracle",
    "payload",
    "model-parity",
    "census",
    "firewall",
    "web",
    "accounting",
    "clean-reproduction",
)
CLEAN_CHECK_NAMES = (
    "source-clean",
    "worktree-checkout",
    "uv-sync",
    "ruff",
    "pytest",
    "web-build-parity",
    "payload-strict-load",
    "certificates",
    "proof-regeneration",
    "firewall",
    "census-normal",
    "census-hard",
)
CLEAN_TEST_TARGETS = (
    "tests/benchmark",
    "tests/unit",
    "tests/oracle",
)
CLEAN_INPUT_KEYS = (
    "candidate",
    "corpus_candidate",
    "normal_certificate",
    "hard_certificate",
    "solver_binary",
    "solver_marker",
    "solver_source",
)
CORPUS_CANDIDATE_REL = "artifacts/corpus/openwordle-v1-candidate"
_LOG_TAIL_CHARS = 4000

_RUNTIME_SOURCES = (
    "__init__.py",
    "model_spec.py",
    "encoding.py",
    "numeric_policy.py",
    "protocol.py",
    "rules.py",
    "release.py",
    "inference.py",
    "proof.py",
    "webapp.py",
)
_EVIDENCE_FILES = {
    "proof.json": "proof_path",
    "campaign-receipt.json": "campaign_receipt",
    "census-normal.json": "normal_census",
    "census-hard.json": "hard_census",
    "firewall.json": "firewall_evidence",
    "web-contract.json": "web_contract",
    "desktop-assist.png": "desktop_screenshot",
    "mobile-solve.png": "mobile_screenshot",
}
_WEB_FILES = ("index.html", "main.js", "styles.css")
FORBIDDEN_PATH_TOKENS = (
    "answer",
    "oracle",
    "train",
    "optimizer",
    "checkpoint",
    "cache",
    "trajectory",
    "posterior",
    "candidate_count",
)

EXPECTED_DIRECTORIES = frozenset(
    {
        "payload",
        "evidence",
        "web",
        "runtime",
        "runtime/src",
        "runtime/src/deep_green",
        "runtime/vendor",
        "runtime/vendor/alex1770-wordle",
    }
)
EXPECTED_FILE_PATHS = frozenset(
    {
        "candidate-manifest.json",
        *(f"payload/{name}" for name in (MANIFEST_NAME, *EXPECTED_FILES)),
        *(f"evidence/{name}" for name in _EVIDENCE_FILES),
        *(f"web/{name}" for name in _WEB_FILES),
        "runtime/pyproject.toml",
        "runtime/uv.lock",
        "runtime/README.md",
        "runtime/vendor/alex1770-wordle/LICENSE",
        *(
            f"runtime/src/deep_green/{name}"
            for name in _RUNTIME_SOURCES
        ),
    }
)
EXPECTED_ROLES: dict[str, str] = {
    **{
        f"payload/{name}": "release-payload"
        for name in (MANIFEST_NAME, *EXPECTED_FILES)
    },
    "evidence/proof.json": "proof-receipt",
    "evidence/campaign-receipt.json": "campaign-receipt",
    "evidence/census-normal.json": "census-normal",
    "evidence/census-hard.json": "census-hard",
    "evidence/firewall.json": "firewall-evidence",
    "evidence/web-contract.json": "web-contract",
    "evidence/desktop-assist.png": "web-screenshot",
    "evidence/mobile-solve.png": "web-screenshot",
    **{f"web/{name}": "web-dist" for name in _WEB_FILES},
    "runtime/pyproject.toml": "runtime-manifest",
    "runtime/uv.lock": "runtime-lockfile",
    "runtime/README.md": "runtime-doc",
    "runtime/vendor/alex1770-wordle/LICENSE": "runtime-license",
    **{
        f"runtime/src/deep_green/{name}": "runtime-source"
        for name in _RUNTIME_SOURCES
    },
}
_BINDING_HASH_KEYS = (
    "weights_sha256",
    "manifest_sha256",
    "proof_sha256",
    "campaign_receipt_sha256",
    "firewall_sha256",
    "web_contract_sha256",
)

_FIREWALL_CHECK_NAMES = frozenset(
    {
        "strict-release-load",
        "actual-seatbelt-launch",
        "request-determinism",
        "reset-determinism",
        "full-vocabulary-count",
        "forbidden-field-parser-battery",
        "network-denied",
        "file-write-denied",
        "fail-closed-empty-mask",
    }
)
_BACKEND_KEYS = {
    "states",
    "tolerance",
    "max_abs_delta_torch_numpy",
    "max_abs_delta_torch_mlx",
    "max_abs_delta_numpy_mlx",
    "torch_disagreements",
    "numpy_disagreements",
    "mlx_disagreements",
}
_MODES = ("normal", "hard")


class ReleasePipelineError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_hex40(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleasePipelineError(f"cannot read {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ReleasePipelineError(f"{path} is not a JSON object")
    return value


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    _fsync_file(path)


def _write_fsync(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_fsync(temp, data)
        os.replace(temp, path)
        _fsync_dir(path.parent)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
    return path


def _tree_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleasePipelineError(
                f"declared input contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleasePipelineError(
                f"declared input contains a non-regular entry: {path}"
            )
        relative = path.relative_to(root)
        hasher.update(str(relative).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_sha256_file(path).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _sanitize_log(text: str, temp_root: Path) -> str:
    aliases = {str(temp_root), str(temp_root.resolve())}
    sanitized = text
    for alias in sorted(aliases, key=len, reverse=True):
        sanitized = sanitized.replace(alias, "<TEMP>")
    return sanitized[-_LOG_TAIL_CHARS:]


def _check_forbidden_tokens(relative: str) -> None:
    lowered = relative.lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise ReleasePipelineError(
                f"forbidden token {token!r} in candidate path {relative!r}"
            )


def _read_source(path: Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not os.path.lexists(path):
        raise ReleasePipelineError(f"{label} source missing or symlink: {path}")
    if not path.is_file():
        raise ReleasePipelineError(f"{label} source is not a regular file: {path}")
    return path.read_bytes()


def _staging_dir(destination: Path) -> Path:
    return destination.parent / (
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )


def build_candidate(
    *,
    output,
    payload_dir,
    proof_path,
    campaign_receipt,
    normal_census,
    hard_census,
    firewall_evidence,
    web_contract,
    web_dist,
    desktop_screenshot,
    mobile_screenshot,
    repo_root,
) -> dict:
    """Assemble the allowlist-only candidate tree and its accounting manifest."""
    from deep_green.proof import load_public_proof

    output = Path(output)
    repo_root = Path(repo_root)
    if os.path.lexists(output) or output.is_symlink():
        raise ReleasePipelineError(f"candidate output already exists: {output}")
    payload_dir = Path(payload_dir)
    source_payload = load_release(payload_dir)

    sources: dict[str, Path] = {
        "proof_path": Path(proof_path),
        "campaign_receipt": Path(campaign_receipt),
        "normal_census": Path(normal_census),
        "hard_census": Path(hard_census),
        "firewall_evidence": Path(firewall_evidence),
        "web_contract": Path(web_contract),
        "desktop_screenshot": Path(desktop_screenshot),
        "mobile_screenshot": Path(mobile_screenshot),
    }
    copies: list[tuple[Path, str]] = []
    for name in (MANIFEST_NAME, *EXPECTED_FILES):
        copies.append((payload_dir / name, f"payload/{name}"))
    for name, key in _EVIDENCE_FILES.items():
        copies.append((sources[key], f"evidence/{name}"))
    web_dist = Path(web_dist)
    for name in _WEB_FILES:
        copies.append((web_dist / name, f"web/{name}"))
    copies.extend(
        [
            (repo_root / "pyproject.toml", "runtime/pyproject.toml"),
            (repo_root / "uv.lock", "runtime/uv.lock"),
            (repo_root / "README.md", "runtime/README.md"),
            (
                repo_root / "vendor/alex1770-wordle/LICENSE",
                "runtime/vendor/alex1770-wordle/LICENSE",
            ),
        ]
    )
    for name in _RUNTIME_SOURCES:
        copies.append(
            (
                repo_root / "src/deep_green" / name,
                f"runtime/src/deep_green/{name}",
            )
        )

    staged_bytes: dict[str, bytes] = {}
    for source, relative in copies:
        _check_forbidden_tokens(relative)
        staged_bytes[relative] = _read_source(source, relative)

    staging = _staging_dir(output)
    try:
        staging.mkdir(parents=True)
        for relative, data in staged_bytes.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_fsync(target, data)

        evidence = staging / "evidence"
        bindings = {
            "weights_sha256": source_payload.weights_sha256,
            "manifest_sha256": source_payload.manifest_sha256,
            "proof_sha256": _sha256_file(evidence / "proof.json"),
            "campaign_receipt_sha256": _sha256_file(
                evidence / "campaign-receipt.json"
            ),
            "census_sha256s": {
                "normal": _sha256_file(evidence / "census-normal.json"),
                "hard": _sha256_file(evidence / "census-hard.json"),
            },
            "firewall_sha256": _sha256_file(evidence / "firewall.json"),
            "web_contract_sha256": _sha256_file(evidence / "web-contract.json"),
        }
        files = {}
        for relative in sorted(staged_bytes):
            data = staged_bytes[relative]
            files[relative] = {
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
                "role": EXPECTED_ROLES[relative],
            }
        manifest = {
            "schema": CANDIDATE_SCHEMA,
            "bindings": bindings,
            "files": files,
        }
        _write_fsync(staging / "candidate-manifest.json", _canonical_bytes(manifest))

        verify_accounting(staging)
        load_release(staging / "payload")
        proof = load_public_proof(
            staging / "evidence/proof.json", staging / "payload"
        )
        if proof.get("status") != "VERIFIED":
            raise ReleasePipelineError(
                "staged proof receipt did not verify against the staged payload"
            )
        for directory in sorted(
            path for path in staging.rglob("*") if path.is_dir()
        ):
            _fsync_dir(directory)
        _fsync_dir(staging)
        os.rename(staging, output)
        _fsync_dir(output.parent)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    manifest_sha256 = _sha256_file(output / "candidate-manifest.json")
    return {
        "schema": CANDIDATE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
    }


def verify_accounting(candidate, *, _allowed_extra: frozenset = frozenset()) -> dict:
    """Require the exact declared tree: no extra files, dirs, or symlinks."""
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ReleasePipelineError(
            f"candidate root missing or symlink: {candidate}"
        )
    allowed_files = set(EXPECTED_FILE_PATHS) | set(_allowed_extra)
    seen_files: set[str] = set()
    stack = [candidate]
    while stack:
        directory = stack.pop()
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            relative = str(Path(entry.path).relative_to(candidate))
            if entry.is_symlink():
                raise ReleasePipelineError(
                    f"candidate entry is a symlink: {relative}"
                )
            if entry.is_dir():
                if relative not in EXPECTED_DIRECTORIES:
                    raise ReleasePipelineError(
                        f"undeclared directory in candidate: {relative}"
                    )
                stack.append(Path(entry.path))
            elif entry.is_file():
                if relative not in allowed_files:
                    raise ReleasePipelineError(
                        f"undeclared file in candidate: {relative}"
                    )
                seen_files.add(relative)
            else:
                raise ReleasePipelineError(
                    f"candidate entry is not regular: {relative}"
                )
    if seen_files != allowed_files:
        missing = sorted(allowed_files - seen_files)
        raise ReleasePipelineError(f"candidate files missing: {missing}")

    manifest_path = candidate / "candidate-manifest.json"
    manifest = _load_json(manifest_path)
    if set(manifest) != {"schema", "bindings", "files"}:
        raise ReleasePipelineError(
            "candidate manifest must carry exactly schema/bindings/files"
        )
    if manifest["schema"] != CANDIDATE_SCHEMA:
        raise ReleasePipelineError(
            f"candidate manifest schema {manifest['schema']!r} != {CANDIDATE_SCHEMA!r}"
        )
    bindings = manifest["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(
        _BINDING_HASH_KEYS
    ) | {"census_sha256s"}:
        raise ReleasePipelineError("candidate bindings key set drifted")
    for key in _BINDING_HASH_KEYS:
        if not _is_sha256(bindings[key]):
            raise ReleasePipelineError(
                f"candidate binding {key!r} is not a lowercase sha256"
            )
    census_hashes = bindings["census_sha256s"]
    if (
        not isinstance(census_hashes, dict)
        or set(census_hashes) != set(_MODES)
        or any(not _is_sha256(census_hashes[mode]) for mode in _MODES)
    ):
        raise ReleasePipelineError(
            "candidate census bindings must be sha256 for exactly normal/hard"
        )
    declared = manifest["files"]
    if not isinstance(declared, dict):
        raise ReleasePipelineError("candidate manifest files must be an object")
    expected_files = allowed_files - {"candidate-manifest.json"} - set(_allowed_extra)
    if set(declared) != expected_files:
        raise ReleasePipelineError("candidate manifest file set drifted")
    for relative, entry in declared.items():
        _check_forbidden_tokens(relative)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "role"}:
            raise ReleasePipelineError(
                f"candidate manifest entry {relative!r} is malformed"
            )
        if not _is_sha256(entry["sha256"]):
            raise ReleasePipelineError(
                f"candidate manifest entry {relative!r} has a bad digest"
            )
        if entry["role"] != EXPECTED_ROLES[relative]:
            raise ReleasePipelineError(
                f"candidate manifest entry {relative!r} role "
                f"{entry['role']!r} != {EXPECTED_ROLES[relative]!r}"
            )
        if (
            not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] < 0
        ):
            raise ReleasePipelineError(
                f"candidate manifest entry {relative!r} has a bad byte count"
            )
        data = (candidate / relative).read_bytes()
        if len(data) != entry["bytes"]:
            raise ReleasePipelineError(
                f"{relative} holds {len(data)} bytes, manifest declares "
                f"{entry['bytes']}"
            )
        if _sha256_bytes(data) != entry["sha256"]:
            raise ReleasePipelineError(f"{relative} sha256 drifted")
    return manifest


def _bindings(candidate: Path) -> dict:
    manifest = _load_json(candidate / "candidate-manifest.json")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        raise ReleasePipelineError("candidate manifest has no bindings")
    return bindings


def _gate_benchmark(frozen_root: Path) -> str:
    from deep_green.benchmark import verify_frozen_bundle

    report = verify_frozen_bundle(Path(frozen_root) / "manifest.json")
    return f"frozen bundle verified: views={report.view_count}"


def _gate_oracle(
    candidate: Path,
    frozen_root: Path,
    normal_certificate: Path,
    hard_certificate: Path,
) -> str:
    from deep_green.oracle.verifier import verify_certificate

    proof = _load_json(candidate / "evidence/proof.json")
    campaign = _load_json(candidate / "evidence/campaign-receipt.json")
    certificates = proof.get("certificates")
    campaign_certs = campaign.get("certificate_sha256s")
    if not isinstance(certificates, dict) or not isinstance(campaign_certs, dict):
        raise ReleasePipelineError("proof/campaign certificate bindings missing")
    tier = None
    for mode, certificate_path in (
        ("normal", Path(normal_certificate)),
        ("hard", Path(hard_certificate)),
    ):
        verify_certificate(certificate_path, frozen_root=frozen_root)
        certificate = _load_json(certificate_path)
        identity = certificate.get("identity", {})
        certificate_tier = identity.get("tier")
        if (
            identity.get("mode") != mode
            or not isinstance(certificate_tier, int)
            or isinstance(certificate_tier, bool)
            or certificate_tier <= 0
            or (tier is not None and certificate_tier != tier)
        ):
            raise ReleasePipelineError(
                f"{mode} certificate identity is inconsistent"
            )
        tier = certificate_tier
        digest = _sha256_file(certificate_path)
        record = certificates.get(mode)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            raise ReleasePipelineError(
                f"{mode} certificate hash disagrees with the proof receipt"
            )
        if campaign_certs.get(mode) != digest:
            raise ReleasePipelineError(
                f"{mode} certificate hash disagrees with the campaign receipt"
            )
    return f"both @{tier} certificates verified and hash-bound"


def _gate_payload(candidate: Path) -> str:
    payload = load_release(candidate / "payload")
    bindings = _bindings(candidate)
    if (
        bindings.get("weights_sha256") != payload.weights_sha256
        or bindings.get("manifest_sha256") != payload.manifest_sha256
    ):
        raise ReleasePipelineError(
            "candidate bindings diverge from the strict payload"
        )
    return f"strict payload verified: width={payload.width}"


def _gate_model_parity(
    candidate: Path,
    normal_certificate: Path,
    hard_certificate: Path,
) -> str:
    from deep_green.oracle.tree_stats import count_policy_states

    campaign = _load_json(candidate / "evidence/campaign-receipt.json")
    if campaign.get("schema") != "wordle.adaptive-campaign/v1":
        raise ReleasePipelineError("campaign receipt schema mismatch")
    payload = load_release(candidate / "payload")
    if campaign.get("status") != "PARITY" or campaign.get("selected_width") != payload.width:
        raise ReleasePipelineError("campaign did not select this release")
    backend = campaign.get("backend_parity")
    if not isinstance(backend, dict) or set(backend) != _BACKEND_KEYS:
        raise ReleasePipelineError("campaign backend report is malformed")
    expected_states = sum(
        count_policy_states(_load_json(path)["policy_tree"])
        for path in (Path(normal_certificate), Path(hard_certificate))
    )
    if backend["states"] != expected_states or backend["tolerance"] != 1e-5:
        raise ReleasePipelineError("campaign backend coverage is incomplete")
    deltas = (
        backend["max_abs_delta_torch_numpy"],
        backend["max_abs_delta_torch_mlx"],
        backend["max_abs_delta_numpy_mlx"],
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in deltas
    ):
        raise ReleasePipelineError("campaign backend deltas are malformed")
    if max(deltas) > backend["tolerance"]:
        raise ReleasePipelineError("campaign backend deltas exceed tolerance")
    if any(
        backend[name] != 0
        for name in (
            "torch_disagreements",
            "numpy_disagreements",
            "mlx_disagreements",
        )
    ):
        raise ReleasePipelineError("campaign backend actions disagree")
    bindings = _bindings(candidate)
    release = campaign.get("release")
    if not isinstance(release, dict):
        raise ReleasePipelineError("campaign release binding missing")
    if (
        release.get("weights_sha256") != bindings.get("weights_sha256")
        or release.get("manifest_sha256") != bindings.get("manifest_sha256")
    ):
        raise ReleasePipelineError("campaign release hashes diverge")
    scale = release.get("score_scale")
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or scale != campaign.get("score_scale")
        or not 0 < scale <= 1
    ):
        raise ReleasePipelineError("campaign release score scale is invalid")
    if release.get("backend_parity") != backend:
        raise ReleasePipelineError("campaign release backend report diverges")
    return (
        f"PARITY width={payload.width} states={backend['states']} "
        f"max_delta={max(deltas)} disagreements=0"
    )


def _gate_census(
    candidate: Path,
    frozen_root: Path,
    normal_certificate: Path,
    hard_certificate: Path,
) -> str:
    from deep_green.proof import generate_proof_receipt

    evidence = candidate / "evidence"
    generated = generate_proof_receipt(
        release_dir=candidate / "payload",
        campaign_receipt_path=evidence / "campaign-receipt.json",
        census_paths={
            "normal": evidence / "census-normal.json",
            "hard": evidence / "census-hard.json",
        },
        firewall_path=evidence / "firewall.json",
        certificate_paths={
            "normal": Path(normal_certificate),
            "hard": Path(hard_certificate),
        },
        frozen_root=Path(frozen_root),
    )
    if _canonical_bytes(generated) != (evidence / "proof.json").read_bytes():
        raise ReleasePipelineError(
            "regenerated proof receipt differs from the candidate evidence"
        )
    objectives = generated["claim"]["objectives"]
    return (
        "proof regenerates byte-exact: "
        + ", ".join(
            f"{mode} losses={objectives[mode]['losses']} "
            f"score7={objectives[mode]['score7_total']} "
            f"worst={objectives[mode]['worst_score']}"
            for mode in _MODES
        )
    )


def _gate_firewall(candidate: Path, run_live_firewall: bool) -> str:
    firewall = _load_json(candidate / "evidence/firewall.json")
    bindings = _bindings(candidate)
    if (
        firewall.get("schema") != "wordle.firewall/v1"
        or firewall.get("status") != "PASS"
        or firewall.get("weights_sha256") != bindings.get("weights_sha256")
        or firewall.get("manifest_sha256") != bindings.get("manifest_sha256")
    ):
        raise ReleasePipelineError("copied firewall evidence is not bound PASS")
    checks = firewall.get("checks")
    names = (
        [check.get("check") for check in checks] if isinstance(checks, list) else []
    )
    if (
        len(names) != len(_FIREWALL_CHECK_NAMES)
        or set(names) != _FIREWALL_CHECK_NAMES
        or len(set(names)) != len(names)
        or any(check.get("status") != "PASS" for check in checks)
    ):
        raise ReleasePipelineError("copied firewall evidence is incomplete")
    if run_live_firewall:
        from deep_green.firewall_audit import audit_release

        live = audit_release(candidate / "payload")
        if live.get("status") != "PASS":
            raise ReleasePipelineError("live firewall audit failed")
        return "copied evidence PASS; live audit PASS"
    return "copied evidence PASS; live audit skipped"


def _gate_web(candidate: Path) -> str:
    contract = _load_json(candidate / "evidence/web-contract.json")
    bindings = _bindings(candidate)
    if (
        contract.get("schema") != "wordle.web-contract/v1"
        or contract.get("status") != "PASS"
    ):
        raise ReleasePipelineError("web contract is not PASS")
    release = contract.get("release")
    if not isinstance(release, dict) or (
        release.get("weights_sha256") != bindings.get("weights_sha256")
        or release.get("manifest_sha256") != bindings.get("manifest_sha256")
    ):
        raise ReleasePipelineError("web contract release binding diverges")
    if contract.get("proof_sha256") != bindings.get("proof_sha256"):
        raise ReleasePipelineError("web contract proof binding diverges")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReleasePipelineError("web contract carries no artifact hashes")
    dist = artifacts.get("dist")
    screenshots = artifacts.get("screenshots")
    if not isinstance(dist, dict) or not isinstance(screenshots, dict):
        raise ReleasePipelineError("web contract artifact sections are malformed")
    for name in _WEB_FILES:
        entry = dist.get(name)
        if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
            raise ReleasePipelineError(f"web contract dist entry {name!r} missing")
        if _sha256_file(candidate / "web" / name) != entry["sha256"]:
            raise ReleasePipelineError(f"candidate web/{name} hash diverges")
    for name in ("desktop-assist.png", "mobile-solve.png"):
        entry = screenshots.get(name)
        if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
            raise ReleasePipelineError(
                f"web contract screenshot entry {name!r} missing"
            )
        if _sha256_file(candidate / "evidence" / name) != entry["sha256"]:
            raise ReleasePipelineError(f"candidate evidence/{name} hash diverges")
    return "web contract binds release, proof, dist, and screenshots"


def _validate_clean_receipt(receipt, candidate_manifest_sha256: str | None) -> None:
    if not isinstance(receipt, dict):
        raise ReleasePipelineError("clean reproduction receipt is not an object")
    if receipt.get("schema") != CLEAN_SCHEMA:
        raise ReleasePipelineError("clean reproduction schema mismatch")
    if receipt.get("status") != "PASS":
        raise ReleasePipelineError("clean reproduction did not PASS")
    if not _is_hex40(receipt.get("source_revision")):
        raise ReleasePipelineError("clean reproduction source revision is invalid")
    if (
        candidate_manifest_sha256 is None
        or receipt.get("candidate_manifest_sha256") != candidate_manifest_sha256
    ):
        raise ReleasePipelineError(
            "clean reproduction receipt does not bind this candidate"
        )
    checks = receipt.get("checks")
    if not isinstance(checks, list) or [
        check.get("name") for check in checks if isinstance(check, dict)
    ] != list(CLEAN_CHECK_NAMES):
        raise ReleasePipelineError("clean reproduction check list drifted")
    if any(
        not isinstance(check, dict)
        or set(check) != {"name", "status"}
        or check["status"] != "PASS"
        for check in checks
    ):
        raise ReleasePipelineError("clean reproduction checks did not all pass")
    declared = receipt.get("declared_inputs")
    if not isinstance(declared, dict) or set(declared) != set(CLEAN_INPUT_KEYS):
        raise ReleasePipelineError(
            "clean reproduction declared inputs drifted"
        )
    for name, entry in declared.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or not _is_sha256(entry["sha256"])
        ):
            raise ReleasePipelineError(
                f"clean reproduction input {name!r} is malformed"
            )


def verify_candidate(
    *,
    candidate,
    frozen_root,
    normal_certificate,
    hard_certificate,
    clean_receipt=None,
    run_live_firewall: bool = True,
) -> dict:
    """Run the nine named gates; every name appears exactly once, in order."""
    candidate = Path(candidate)
    frozen_root = Path(frozen_root)
    try:
        manifest_sha256 = _sha256_file(candidate / "candidate-manifest.json")
    except OSError:
        manifest_sha256 = None

    clean_receipt_obj = clean_receipt
    if isinstance(clean_receipt_obj, (str, Path)):
        try:
            clean_receipt_obj = json.loads(
                Path(clean_receipt_obj).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            clean_receipt_obj = {"__load_error__": str(exc)[:200]}

    gate_runners = {
        "benchmark": lambda: _gate_benchmark(frozen_root),
        "oracle": lambda: _gate_oracle(
            candidate,
            frozen_root,
            Path(normal_certificate),
            Path(hard_certificate),
        ),
        "payload": lambda: _gate_payload(candidate),
        "model-parity": lambda: _gate_model_parity(
            candidate,
            Path(normal_certificate),
            Path(hard_certificate),
        ),
        "census": lambda: _gate_census(
            candidate,
            frozen_root,
            Path(normal_certificate),
            Path(hard_certificate),
        ),
        "firewall": lambda: _gate_firewall(candidate, run_live_firewall),
        "web": lambda: _gate_web(candidate),
        "accounting": lambda: (
            f"accounting verified: {len(verify_accounting(candidate)['files'])} files"
        ),
        "clean-reproduction": lambda: _clean_gate_detail(
            clean_receipt_obj, manifest_sha256
        ),
    }
    gates = []
    for name in GATE_NAMES:
        try:
            detail = gate_runners[name]()
            gates.append({"name": name, "status": "PASS", "detail": detail})
        except Exception as exc:  # noqa: BLE001 - a failed gate must not hide later gates
            gates.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "detail": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
    return {
        "schema": VERIFY_SCHEMA,
        "status": "PASS" if all(g["status"] == "PASS" for g in gates) else "FAIL",
        "candidate_manifest_sha256": manifest_sha256,
        "gates": gates,
    }


def _clean_gate_detail(receipt, manifest_sha256: str | None) -> str:
    if receipt is None:
        raise ReleasePipelineError("no clean reproduction receipt supplied")
    _validate_clean_receipt(receipt, manifest_sha256)
    return f"clean reproduction PASS at {receipt['source_revision']}"


def _run_command(argv, *, cwd: Path, timeout: int = 3600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed


def _python_check_script(name: str) -> str:
    scripts = {
        "payload-strict-load": (
            "import json,sys\n"
            "from deep_green.release import load_release\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "load_release(cfg['payload'])\n"
            "print('strict payload ok')\n"
        ),
        "certificates": (
            "import json,sys\n"
            "from pathlib import Path\n"
            "from deep_green.oracle.verifier import verify_certificate\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "for key in ('normal_certificate','hard_certificate'):\n"
            "    verify_certificate(Path(cfg[key]), frozen_root=Path(cfg['frozen_root']),"
            " vendor_root=Path(cfg['vendor_root']), build_cache=Path(cfg['build_cache']))\n"
            "print('certificates ok')\n"
        ),
        "proof-regeneration": (
            "import json,sys\n"
            "from deep_green.proof import generate_proof_receipt\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "candidate=cfg['candidate']\n"
            "r=generate_proof_receipt(release_dir=candidate+'/payload',"
            " campaign_receipt_path=candidate+'/evidence/campaign-receipt.json',"
            " census_paths={'normal':candidate+'/evidence/census-normal.json',"
            "'hard':candidate+'/evidence/census-hard.json'},"
            " firewall_path=candidate+'/evidence/firewall.json',"
            " certificate_paths={'normal':cfg['normal_certificate'],"
            "'hard':cfg['hard_certificate']},"
            " frozen_root=cfg['frozen_root'])\n"
            "canon=(json.dumps(r,sort_keys=True,separators=(',',':'))+'\\n').encode()\n"
            "assert canon==open(candidate+'/evidence/proof.json','rb').read(),'proof drift'\n"
            "print('proof regenerates byte-exact')\n"
        ),
        "firewall": (
            "import json,sys\n"
            "from deep_green.firewall_audit import audit_release\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "evidence=audit_release(cfg['payload'])\n"
            "assert evidence['status']=='PASS','live firewall failed'\n"
            "print('firewall ok')\n"
        ),
        "census-normal": (
            "import json,sys\n"
            "from deep_green.evaluate import evaluate_certified\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "result,_=evaluate_certified(release_dir=cfg['payload'],"
            " certificate_path=cfg['normal_certificate'],"
            " frozen_root=cfg['frozen_root'])\n"
            "expected=json.load(open(cfg['candidate']+'/evidence/census-normal.json'))['result']\n"
            "actual=result.as_dict()\n"
            "for frame in (actual,expected):\n"
            "    frame.pop('latency_ms',None);frame.pop('peak_rss_bytes',None)\n"
            "assert actual==expected,'normal census diverges'\n"
            "print('normal census ok')\n"
        ),
        "census-hard": (
            "import json,sys\n"
            "from deep_green.evaluate import evaluate_certified\n"
            "cfg=json.load(open(sys.argv[1]))\n"
            "result,_=evaluate_certified(release_dir=cfg['payload'],"
            " certificate_path=cfg['hard_certificate'],"
            " frozen_root=cfg['frozen_root'])\n"
            "expected=json.load(open(cfg['candidate']+'/evidence/census-hard.json'))['result']\n"
            "actual=result.as_dict()\n"
            "for frame in (actual,expected):\n"
            "    frame.pop('latency_ms',None);frame.pop('peak_rss_bytes',None)\n"
            "assert actual==expected,'hard census diverges'\n"
            "print('hard census ok')\n"
        ),
    }
    return scripts[name]


def run_clean_reproduction(
    *,
    repo_root,
    candidate,
    output,
    frozen_root,
    normal_certificate,
    hard_certificate,
) -> dict:
    """Replay verification in a detached clean worktree of the committed tree."""
    repo_root = Path(repo_root)
    candidate = Path(candidate)
    output = Path(output)
    frozen_root = Path(frozen_root)
    normal_certificate = Path(normal_certificate)
    hard_certificate = Path(hard_certificate)

    checks: list[dict] = []
    declared_inputs: dict[str, dict] = {}
    failures: list[dict] = []
    source_revision = ""
    try:
        candidate_manifest_sha256 = _sha256_file(
            candidate / "candidate-manifest.json"
        )
    except OSError:
        candidate_manifest_sha256 = None

    def record(name: str, ok: bool) -> bool:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL"})
        return ok

    def run_step(name: str, argv, cwd: Path) -> bool:
        completed = _run_command(argv, cwd=cwd)
        ok = completed.returncode == 0
        if not ok:
            failures.append(
                {
                    "check": name,
                    "command": [str(part) for part in argv],
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                }
            )
        return record(name, ok)

    temp = Path(tempfile.mkdtemp(prefix=".clean-repro-"))
    checkout = temp / "checkout"
    inputs = temp / "inputs"
    worktree_added = False
    try:
        status = _run_command(
            ["git", "-C", str(repo_root), "status", "--porcelain",
             "--untracked-files=no"],
            cwd=repo_root,
            timeout=60,
        )
        revision = _run_command(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            cwd=repo_root,
            timeout=60,
        )
        source_revision = revision.stdout.strip()
        clean_tree = (
            status.returncode == 0
            and status.stdout == ""
            and revision.returncode == 0
            and _is_hex40(source_revision)
        )
        if not record("source-clean", clean_tree):
            raise ReleasePipelineError("source tree is not a clean commit")

        added = _run_command(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach",
             str(checkout), "HEAD"],
            cwd=repo_root,
            timeout=300,
        )
        worktree_added = added.returncode == 0
        if not record("worktree-checkout", worktree_added):
            raise ReleasePipelineError("git worktree checkout failed")
        checkout.mkdir(parents=True, exist_ok=True)

        inputs.mkdir(parents=True, exist_ok=True)
        input_map = {
            "normal_certificate": normal_certificate,
            "hard_certificate": hard_certificate,
        }
        bound_solver_shas = []
        for key, path in input_map.items():
            data = _read_source(path, key)
            target = inputs / f"{key}.json"
            _write_fsync(target, data)
            declared_inputs[key] = {
                "path": str(path),
                "sha256": _sha256_bytes(data),
            }
            bound_solver_shas.append(
                json.loads(data)["binding"]["executable_sha256"]
            )
        cache_dir = repo_root / "artifacts/oracle/build-cache"
        cache_files = {
            "solver_binary": cache_dir / "deep-green-built",
            "solver_marker": cache_dir / "build-marker.json",
            "solver_source": cache_dir / "src/wordle.cpp",
        }
        checkout_cache = checkout / "artifacts/oracle/build-cache"
        checkout_cache.mkdir(parents=True, exist_ok=True)
        for key, path in cache_files.items():
            data = _read_source(path, key)
            relative = Path(path).relative_to(cache_dir)
            target = checkout_cache / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_fsync(target, data)
            if key == "solver_binary":
                os.chmod(target, 0o755)
                _fsync_file(target)
            declared_inputs[key] = {
                "path": str(path),
                "sha256": _sha256_bytes(data),
            }
        # certificates may bind solver builds from different toolchains; stage
        # each bound content-addressed build dir so verification resolves them
        for wanted in bound_solver_shas:
            bound_dir = cache_dir / wanted
            if not bound_dir.is_dir():
                raise ReleasePipelineError(
                    f"certificate-bound solver dir missing: {bound_dir}"
                )
            for name in ("deep-green-built", "build-marker.json"):
                src = bound_dir / name
                data = _read_source(src, f"solver {wanted[:8]} {name}")
                target = checkout_cache / wanted / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_fsync(target, data)
                if name == "deep-green-built":
                    os.chmod(target, 0o755)
                    _fsync_file(target)
        verify_accounting(candidate)
        shutil.copytree(candidate, inputs / "candidate", symlinks=True)
        verify_accounting(inputs / "candidate")
        declared_inputs["candidate"] = {
            "path": str(candidate),
            "sha256": candidate_manifest_sha256 or "",
        }
        corpus_source = repo_root / CORPUS_CANDIDATE_REL
        if corpus_source.is_symlink() or not corpus_source.is_dir():
            raise ReleasePipelineError(
                f"corpus candidate missing or symlink: {corpus_source}"
            )
        corpus_digest = _tree_digest(corpus_source)
        declared_inputs["corpus_candidate"] = {
            "path": str(corpus_source),
            "sha256": corpus_digest,
        }
        copied_corpus = checkout / CORPUS_CANDIDATE_REL
        shutil.copytree(corpus_source, copied_corpus, symlinks=True)
        if _tree_digest(copied_corpus) != corpus_digest:
            raise ReleasePipelineError("copied corpus candidate digest drifted")

        check_frozen = checkout / Path(frozen_root)
        run_cfg = {
            "candidate": str(inputs / "candidate"),
            "payload": str(inputs / "candidate/payload"),
            "frozen_root": str(check_frozen),
            "normal_certificate": str(inputs / "normal_certificate.json"),
            "hard_certificate": str(inputs / "hard_certificate.json"),
            "vendor_root": str(checkout / "vendor/alex1770-wordle"),
            "build_cache": str(checkout_cache),
        }
        cfg_path = inputs / "run-config.json"
        _write_fsync(cfg_path, json.dumps(run_cfg).encode("utf-8"))

        steps = [
            ("uv-sync", ["uv", "sync", "--extra", "dev", "--frozen"], checkout),
            ("ruff", ["uv", "run", "--no-sync", "ruff", "check", "."], checkout),
            (
                "pytest",
                ["uv", "run", "--no-sync", "pytest", *CLEAN_TEST_TARGETS, "-q"],
                checkout,
            ),
        ]
        for name, argv, cwd in steps:
            if not run_step(name, argv, cwd):
                raise ReleasePipelineError(f"clean reproduction step {name} failed")

        dist_ok = True
        for argv in (
            ["npm", "ci"],
            ["npm", "run", "typecheck"],
            ["npm", "run", "build"],
        ):
            completed = _run_command(argv, cwd=checkout / "web")
            if completed.returncode != 0:
                failures.append(
                    {
                        "check": "web-build-parity",
                        "command": [str(part) for part in argv],
                        "stdout": completed.stdout or "",
                        "stderr": completed.stderr or "",
                    }
                )
                dist_ok = False
                break
        for name in _WEB_FILES:
            built = checkout / "web/dist" / name
            staged = candidate / "web" / name
            if (
                not built.is_file()
                or _sha256_file(built) != _sha256_file(staged)
            ):
                dist_ok = False
        record("web-build-parity", dist_ok)
        if not dist_ok:
            raise ReleasePipelineError("clean web dist diverged from candidate")

        for name in (
            "payload-strict-load",
            "certificates",
            "proof-regeneration",
            "firewall",
            "census-normal",
            "census-hard",
        ):
            completed = _run_command(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "-c",
                    _python_check_script(name),
                    str(cfg_path),
                ],
                cwd=checkout,
            )
            if completed.returncode != 0:
                failures.append(
                    {
                        "check": name,
                        "command": [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "-c",
                            f"<{name}>",
                            str(cfg_path),
                        ],
                        "stdout": completed.stdout or "",
                        "stderr": completed.stderr or "",
                    }
                )
            if not record(name, completed.returncode == 0):
                raise ReleasePipelineError(
                    f"clean reproduction check {name} failed"
                )
    except Exception:  # noqa: BLE001, S110 - any failure still yields a FAIL receipt
        pass
    finally:
        recorded = {check["name"] for check in checks}
        for name in CLEAN_CHECK_NAMES:
            if name not in recorded:
                checks.append({"name": name, "status": "SKIP"})
        if worktree_added:
            _run_command(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force",
                 str(checkout)],
                cwd=repo_root,
                timeout=300,
            )
        shutil.rmtree(temp, ignore_errors=True)

    receipt = {
        "schema": CLEAN_SCHEMA,
        "status": (
            "PASS"
            if all(check["status"] == "PASS" for check in checks)
            else "FAIL"
        ),
        "source_revision": source_revision,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "declared_inputs": declared_inputs,
        "checks": checks,
    }
    _write_atomic(output, _canonical_bytes(receipt))
    if receipt["status"] == "FAIL" and failures:
        sections = []
        for failure in failures:
            command = _sanitize_log(" ".join(failure["command"]), temp)
            sections.append(
                f"== {failure['check']}: {command}\n"
                f"--- stdout ---\n"
                f"{_sanitize_log(failure['stdout'], temp)}\n"
                f"--- stderr ---\n"
                f"{_sanitize_log(failure['stderr'], temp)}\n"
            )
        _write_atomic(
            Path(str(output) + ".log"), "".join(sections).encode("utf-8")
        )
    return receipt


def _receipt_bytes(source) -> bytes:
    if isinstance(source, dict):
        return _canonical_bytes(source)
    return Path(source).read_bytes()


def _load_receipt(source) -> dict:
    if isinstance(source, dict):
        return source
    return _load_json(Path(source))


def _tar_members(candidate: Path, extra_files: dict[str, bytes]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for relative in sorted(EXPECTED_FILE_PATHS):
        members[f"deep-green/{relative}"] = (candidate / relative).read_bytes()
    for name, data in extra_files.items():
        members[f"deep-green/{name}"] = data
    return members


def _build_archive(members: dict[str, bytes]) -> bytes:
    directories = {"deep-green/"}
    for name in members:
        parts = name.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]) + "/")
    entries = sorted(set(members) | directories)
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for name in entries:
            info = tarfile.TarInfo(name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                tar.addfile(info)
            else:
                data = members[name]
                info.type = tarfile.REGTYPE
                info.mode = 0o644
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def seal_candidate(
    *,
    candidate,
    verification_receipt,
    clean_receipt,
    output_dir,
) -> dict:
    """Verify both receipts, emit a deterministic archive, and seal it."""
    from deep_green.firewall_audit import audit_release
    from deep_green.proof import load_public_proof

    candidate = Path(candidate)
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir) or output_dir.is_symlink():
        raise ReleasePipelineError(f"seal output already exists: {output_dir}")
    manifest_sha256 = _sha256_file(candidate / "candidate-manifest.json")
    bindings = _bindings(candidate)

    verification = _load_receipt(verification_receipt)
    gates = verification.get("gates")
    if (
        set(verification)
        != {"schema", "status", "candidate_manifest_sha256", "gates"}
        or verification["schema"] != VERIFY_SCHEMA
        or verification["status"] != "PASS"
        or verification["candidate_manifest_sha256"] != manifest_sha256
        or not isinstance(gates, list)
        or any(
            not isinstance(gate, dict)
            or set(gate) != {"name", "status", "detail"}
            or not isinstance(gate["detail"], str)
            for gate in gates
        )
        or [gate["name"] for gate in gates] != list(GATE_NAMES)
        or any(gate["status"] != "PASS" for gate in gates)
    ):
        raise ReleasePipelineError(
            "verification receipt is not a bound nine-gate PASS"
        )
    clean = _load_receipt(clean_receipt)
    _validate_clean_receipt(clean, manifest_sha256)

    verification_bytes = _receipt_bytes(verification_receipt)
    clean_bytes = _receipt_bytes(clean_receipt)
    archive_bytes = _build_archive(
        _tar_members(
            candidate,
            {
                "clean-reproduction.json": clean_bytes,
                "release-verification.json": verification_bytes,
            },
        )
    )
    archive_sha256 = _sha256_bytes(archive_bytes)

    staging = _staging_dir(output_dir)
    try:
        staging.mkdir(parents=True)
        _write_fsync(staging / "release.tar.gz", archive_bytes)
        seal = {
            "schema": SEAL_SCHEMA,
            "status": "SEALED",
            "candidate_manifest_sha256": manifest_sha256,
            "archive_sha256": archive_sha256,
            "archive_bytes": len(archive_bytes),
            "verification_sha256": _sha256_bytes(verification_bytes),
            "clean_reproduction_sha256": _sha256_bytes(clean_bytes),
            "release": {
                "weights_sha256": bindings["weights_sha256"],
                "manifest_sha256": bindings["manifest_sha256"],
            },
            "proof_sha256": bindings["proof_sha256"],
        }
        _write_fsync(staging / "seal.json", _canonical_bytes(seal))

        extract_dir = staging / ".selfcheck"
        extract_dir.mkdir()
        with (
            tarfile.open(
                fileobj=io.BytesIO(archive_bytes), mode="r:gz"
            ) as tar
        ):
            tar.extractall(extract_dir, filter="data")
        extracted = extract_dir / "deep-green"
        verify_accounting(
            extracted,
            _allowed_extra=frozenset(
                {"clean-reproduction.json", "release-verification.json"}
            ),
        )
        load_release(extracted / "payload")
        proof = load_public_proof(
            extracted / "evidence/proof.json", extracted / "payload"
        )
        if proof.get("status") != "VERIFIED":
            raise ReleasePipelineError(
                "sealed proof receipt did not verify"
            )
        live = audit_release(extracted / "payload")
        if live.get("status") != "PASS":
            raise ReleasePipelineError("seal-time firewall audit failed")
        shutil.rmtree(extract_dir)
        _fsync_dir(staging)
        os.rename(staging, output_dir)
        _fsync_dir(output_dir.parent)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    for name in ("release.tar.gz", "seal.json"):
        target = output_dir / name
        os.chmod(target, 0o444)
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    os.chmod(output_dir, 0o555)
    _fsync_dir(output_dir.parent)
    return seal
