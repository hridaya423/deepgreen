"""Deterministic Apple Clang build of the vendored exact solver.

The vendored tree stays byte-identical to upstream; the JSONL event-stream
patch is applied into the build cache, never into the pinned source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

VENDOR_DIR_NAME = "alex1770-wordle"
COMPILE_FLAGS = ("-std=c++17", "-O3", "-Wall")
BINARY_NAME = "deep-green-built"


class OracleBuildError(RuntimeError):
    pass


def vendor_dir(repo_root: Path) -> Path:
    return repo_root / "vendor" / VENDOR_DIR_NAME


def solver_digest(exe_path: Path) -> str:
    digest = hashlib.sha256()
    with exe_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_commit(vendor_dir: Path) -> str:
    import tomllib

    pin = vendor_dir / "UPSTREAM.toml"
    if not pin.is_file():
        raise OracleBuildError(f"pin record missing: {pin}")
    raw = tomllib.loads(pin.read_text(encoding="utf-8"))
    return str(raw["upstream"]["commit"])


def _stage_patched_source(vendor_dir: Path, stage_dir: Path) -> Path:
    """Copy the pristine source and apply the recorded local patch."""
    source = vendor_dir / "wordle.cpp"
    if not source.is_file():
        raise OracleBuildError(f"solver source missing: {source}")
    patch_path = vendor_dir / "patches" / "events-v1.patch"
    if not patch_path.is_file():
        raise OracleBuildError(f"event-stream patch missing: {patch_path}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / "wordle.cpp"
    shutil.copyfile(source, staged)
    patch_tool = shutil.which("patch") or "/usr/bin/patch"
    completed = subprocess.run(
        [patch_tool, "-s", "-N", "-i", str(patch_path)],
        cwd=str(stage_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise OracleBuildError(f"failed to apply events-v1.patch: {detail[:2000]}")
    return staged


def compiler_identity() -> str:
    compiler = shutil.which("clang++") or "/usr/bin/clang++"
    completed = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise OracleBuildError(f"failed to identify compiler: {(completed.stderr or '').strip()[:2000]}")
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    if not first_line:
        raise OracleBuildError("compiler produced no version identity")
    return f"{first_line}; flags={' '.join(COMPILE_FLAGS)}"


def _sdk_fallbacks() -> tuple[Path, ...]:
    sdk_root = Path("/Library/Developer/CommandLineTools/SDKs")
    default_sdk = (sdk_root / "MacOSX.sdk").resolve()
    if not sdk_root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(sdk_root.glob("MacOSX*.sdk"), reverse=True)
        if path.resolve() != default_sdk
    )


def build_solver(vendor_dir: Path, output_dir: Path) -> Path:
    compat_dir = vendor_dir / "compat"
    if not (compat_dir / "error.h").is_file():
        raise OracleBuildError("compat shim missing: compat/error.h")
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = _stage_patched_source(vendor_dir, output_dir / "src")
    exe_path = output_dir / BINARY_NAME
    compiler = shutil.which("clang++") or "/usr/bin/clang++"
    commit_desc = _upstream_commit(vendor_dir)
    command = [
        compiler,
        *COMPILE_FLAGS,
        f"-DCOMMITDESC=\"{commit_desc} (local events patch)\"",
        f"-I{compat_dir}",
        "-o",
        str(exe_path),
        str(staged),
    ]
    env = dict(os.environ)
    attempts = (None, *_sdk_fallbacks())
    completed = None
    for sdk_root in attempts:
        if sdk_root is not None:
            env["SDKROOT"] = str(sdk_root)
        exe_path.unlink(missing_ok=True)
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=env
        )
        detail = completed.stderr or completed.stdout or ""
        if completed.returncode == 0 and exe_path.is_file():
            break
        if "tapi error: malformed file" not in detail:
            break
    if completed is None or completed.returncode != 0 or not exe_path.is_file():
        detail = "" if completed is None else completed.stderr or completed.stdout
        raise OracleBuildError(f"solver build failed: {detail.strip()[:2000]}")
    return exe_path


def inspect_solver(vendor_dir: Path, cache_dir: Path) -> tuple[Path, str, str]:
    """Read-only check of a cached solver build; raises OracleBuildError on any drift."""
    exe_path = cache_dir / BINARY_NAME
    marker = cache_dir / "build-marker.json"
    if not exe_path.is_file():
        raise OracleBuildError(f"solver binary missing: {exe_path}")
    if not marker.is_file():
        raise OracleBuildError(f"solver build marker missing: {marker}")
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OracleBuildError(f"solver build marker is malformed: {exc}") from exc
    if not isinstance(recorded, dict):
        raise OracleBuildError("solver build marker is not a JSON object")

    source_digest = _sha256(vendor_dir / "wordle.cpp")
    patch_digest = _sha256(vendor_dir / "patches" / "events-v1.patch")
    shim_digest = _sha256(vendor_dir / "compat" / "error.h")
    compiler_id = compiler_identity()
    if (
        recorded.get("source_sha256") != source_digest
        or recorded.get("patch_sha256") != patch_digest
        or recorded.get("shim_sha256") != shim_digest
        or recorded.get("flags") != list(COMPILE_FLAGS)
        or recorded.get("compiler_id") != compiler_id
    ):
        raise OracleBuildError("solver build marker does not match current inputs")
    binary_sha256 = solver_digest(exe_path)
    if recorded.get("binary_sha256") != binary_sha256:
        raise OracleBuildError("solver build marker does not match current inputs")
    return exe_path, binary_sha256, compiler_id


def inspect_certificate_solver(
    vendor_dir: Path, cache_dir: Path, expected: dict
) -> tuple[Path, str, str]:
    """Locate the exact binary a certificate binds, by content address.

    Certificates bind a specific solver executable produced by a recorded
    toolchain. A verifying machine may not run that toolchain, so the bound
    artifact itself must be present: either staged under
    ``cache_dir/<executable_sha256>/`` or as the flat cache binary.
    """
    wanted = expected["executable_sha256"]
    candidates = (cache_dir / wanted, cache_dir)
    for candidate in candidates:
        exe_path = candidate / BINARY_NAME
        marker_path = candidate / "build-marker.json"
        if not exe_path.is_file() or not marker_path.is_file():
            continue
        try:
            recorded = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(recorded, dict):
            continue
        if (
            recorded.get("source_sha256") != _sha256(vendor_dir / "wordle.cpp")
            or recorded.get("patch_sha256")
            != _sha256(vendor_dir / "patches" / "events-v1.patch")
            or recorded.get("shim_sha256")
            != _sha256(vendor_dir / "compat" / "error.h")
            or recorded.get("flags") != list(COMPILE_FLAGS)
            or recorded.get("compiler_id") != expected["compiler_id"]
        ):
            continue
        binary_sha256 = solver_digest(exe_path)
        if binary_sha256 != wanted or recorded.get("binary_sha256") != wanted:
            continue
        return exe_path, binary_sha256, expected["compiler_id"]
    raise OracleBuildError(
        f"certificate-bound solver {wanted} not present under {cache_dir}"
    )


def ensure_solver(vendor_dir: Path, cache_dir: Path) -> tuple[Path, str, str]:
    """Build once per cache dir; rebuild when the cached marker no longer matches."""
    try:
        return inspect_solver(vendor_dir, cache_dir)
    except OracleBuildError:
        pass
    built = build_solver(vendor_dir, cache_dir)
    binary_digest = solver_digest(built)
    compiler_id = compiler_identity()
    marker = cache_dir / "build-marker.json"
    marker.write_text(
        json.dumps(
            {
                "source_sha256": _sha256(vendor_dir / "wordle.cpp"),
                "patch_sha256": _sha256(vendor_dir / "patches" / "events-v1.patch"),
                "shim_sha256": _sha256(vendor_dir / "compat" / "error.h"),
                "flags": list(COMPILE_FLAGS),
                "compiler_id": compiler_id,
                "binary_sha256": binary_digest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.utime(marker, None)
    store = cache_dir / binary_digest
    store.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, store / BINARY_NAME)
    shutil.copyfile(marker, store / "build-marker.json")
    return built, binary_digest, compiler_id
