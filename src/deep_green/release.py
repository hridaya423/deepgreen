"""The single strict release-payload loader used by every production consumer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RELEASE_SCHEMA = "wordle.release-payload/v1"
INFERENCE_CLASS = "numpy-f32-pure-neural"
MANIFEST_NAME = "payload-manifest.json"
MAX_MANIFEST_BYTES = 1 << 20
MAX_RELEASE_FILE_BYTES = 64 << 20
MAX_LEXICON_UNCOMPRESSED_BYTES = 16 << 20
EXPECTED_FILES = {
    "accepted-guesses.txt": "public-action-space",
    "lexicon-indices.npz": "public-lexicon-indices",
    "model-config.json": "model-config",
    "weights.safetensors": "neural-payload",
}
_IDENTITY_KEYS = {
    "benchmark_id",
    "rules_sha256",
    "prior_id",
    "action_space_sha256",
    "track",
    "inference_class",
}
_SHA256_FIELDS = {"rules_sha256", "action_space_sha256"}


class ReleaseError(ValueError):
    pass


@dataclass(frozen=True)
class ReleasePayload:
    release_dir: Path
    action_words: tuple[str, ...]
    action_space_sha256: str
    weights_sha256: str
    manifest_sha256: str
    width: int
    identity: dict
    tensors: dict[str, np.ndarray]


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _read_regular(path: Path, name: str, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ReleaseError(f"{name} could not be opened: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ReleaseError(f"{name} is not a regular file")
        size = info.st_size
        if size < 0 or size > max_bytes:
            raise ReleaseError(
                f"{name} holds {size} bytes, over the {max_bytes}-byte limit"
            )
        data = bytearray()
        remaining = size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise ReleaseError(f"{name} shrank while being read")
            data += chunk
            remaining -= len(chunk)
        if len(data) != size or os.read(fd, 1):
            raise ReleaseError(f"{name} changed size while being read")
        return bytes(data)
    except OSError as exc:
        raise ReleaseError(f"{name} could not be read: {exc}") from exc
    finally:
        os.close(fd)


def _read_checked(release_dir: Path, name: str, entry: dict) -> bytes:
    data = _read_regular(release_dir / name, name, entry["bytes"])
    if len(data) != entry["bytes"]:
        raise ReleaseError(
            f"{name} holds {len(data)} bytes, manifest declares {entry['bytes']}"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise ReleaseError(f"{name} sha256 {digest} != manifest {entry['sha256']}")
    return data


def _reject_json_constant(value: str):
    raise ReleaseError(f"non-finite JSON constant {value!r} is not allowed")


def _parse_json(data: bytes, name: str):
    try:
        return json.loads(
            data.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{name} is not valid UTF-8 JSON: {exc}") from exc


def _validate_identity(identity) -> dict:
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ReleaseError(
            f"manifest identity must carry exactly {sorted(_IDENTITY_KEYS)}"
        )
    for key, value in identity.items():
        if not isinstance(value, str) or not value:
            raise ReleaseError(f"identity {key!r} must be a nonempty string")
    for key in _SHA256_FIELDS:
        if not _is_sha256(identity[key]):
            raise ReleaseError(f"identity {key!r} must be a lowercase sha256")
    if identity["track"] != "full-compiled":
        raise ReleaseError(f"identity track {identity['track']!r} is not full-compiled")
    if identity["inference_class"] != INFERENCE_CLASS:
        raise ReleaseError(
            f"identity inference_class {identity['inference_class']!r} "
            f"is not {INFERENCE_CLASS!r}"
        )
    return dict(identity)


def parse_model_config(data: bytes) -> int:
    """Strictly validate a model-config payload; return the registered width."""
    from deep_green.model_spec import (
        TENSOR_SPEC_VERSION,
        RegisteredWidths,
        parameter_count,
        tensor_names,
    )

    config = _parse_json(data, "model-config.json")
    expected = {"schema", "width", "parameter_count", "tensor_names"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ReleaseError(
            f"model config must carry exactly {sorted(expected)}"
        )
    if config["schema"] != TENSOR_SPEC_VERSION:
        raise ReleaseError(
            f"model config schema {config['schema']!r} != {TENSOR_SPEC_VERSION!r}"
        )
    width = config["width"]
    if not isinstance(width, int) or isinstance(width, bool):
        raise ReleaseError("model config width must be an integer")
    if width not in RegisteredWidths.default().widths:
        raise ReleaseError(f"width {width} is not a registered ladder width")
    if config["parameter_count"] != parameter_count(width):
        raise ReleaseError(
            f"model config parameter_count {config['parameter_count']} != "
            f"{parameter_count(width)}"
        )
    if config["tensor_names"] != list(tensor_names()):
        raise ReleaseError("model config tensor_names do not match the spec order")
    return width


def validate_tensors(tensors: dict[str, np.ndarray], width: int) -> None:
    """Require the exact float32/finite/shape-correct tensor contract."""
    from deep_green.model_spec import expected_shapes, tensor_names

    names = tensor_names()
    if set(tensors) != set(names):
        missing = sorted(set(names) - set(tensors))
        extra = sorted(set(tensors) - set(names))
        raise ReleaseError(f"tensor set mismatch: missing={missing} extra={extra}")
    shapes = expected_shapes(width)
    for name in names:
        array = tensors[name]
        if array.dtype != np.float32:
            raise ReleaseError(f"tensor {name} dtype {array.dtype} != float32")
        if tuple(array.shape) != shapes[name]:
            raise ReleaseError(
                f"tensor {name} shape {tuple(array.shape)} != {shapes[name]}"
            )
        if not np.isfinite(array).all():
            raise ReleaseError(f"tensor {name} contains nonfinite values")


def _parse_action_words(data: bytes) -> tuple[str, ...]:
    from deep_green.rules import Word5

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseError(f"accepted-guesses.txt is not UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise ReleaseError("accepted-guesses.txt lacks a terminal newline")
    lines = text.split("\n")[:-1]
    if not lines:
        raise ReleaseError("accepted-guesses.txt is empty")
    words: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if not line:
            raise ReleaseError("accepted-guesses.txt contains a blank line")
        try:
            parsed = Word5.parse(line)
        except ValueError as exc:
            raise ReleaseError(f"invalid action word {line!r}: {exc}") from exc
        if parsed.value in seen:
            raise ReleaseError(f"duplicate action word {parsed.value!r}")
        seen.add(parsed.value)
        words.append(parsed.value)
    if words != sorted(words):
        raise ReleaseError("accepted-guesses.txt is not in canonical sorted order")
    return tuple(words)


def _load_weights(data: bytes) -> dict[str, np.ndarray]:
    from safetensors import SafetensorError
    from safetensors.numpy import load

    try:
        return load(data)
    except (SafetensorError, OSError, ValueError, KeyError) as exc:
        raise ReleaseError(f"weights.safetensors failed to parse: {exc}") from exc


def _load_lexicon(data: bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if {m.filename for m in members} != {"letters.npy", "positions.npy"}:
                raise ReleaseError(
                    "lexicon npz members "
                    f"{sorted(m.filename for m in members)} != "
                    "['letters.npy', 'positions.npy']"
                )
            total = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise ReleaseError("lexicon npz member is encrypted")
                if "/" in member.filename or "\\" in member.filename:
                    raise ReleaseError(
                        f"lexicon npz member {member.filename!r} carries a path"
                    )
                total += member.file_size
            if total > MAX_LEXICON_UNCOMPRESSED_BYTES:
                raise ReleaseError(
                    f"lexicon npz decompresses to {total} bytes, over the "
                    f"{MAX_LEXICON_UNCOMPRESSED_BYTES}-byte limit"
                )
    except ReleaseError:
        raise
    except zipfile.BadZipFile as exc:
        raise ReleaseError(f"lexicon-indices.npz is not a valid zip: {exc}") from exc
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != {"letters", "positions"}:
                raise ReleaseError(
                    f"lexicon npz keys {sorted(keys)} != ['letters', 'positions']"
                )
            letters = archive["letters"].copy()
            positions = archive["positions"].copy()
    except ReleaseError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ReleaseError(f"lexicon-indices.npz failed to parse: {exc}") from exc
    return letters, positions


def _write_file_fsync(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _deterministic_npz_bytes(letters: np.ndarray, positions: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, array in (
            ("letters.npy", letters),
            ("positions.npy", positions),
        ):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            payload = io.BytesIO()
            np.save(payload, array, allow_pickle=False)
            archive.writestr(member, payload.getvalue())
    return buffer.getvalue()


def publish_release(
    destination,
    *,
    tensors: dict[str, np.ndarray],
    width: int,
    action_words_path,
    identity: dict,
) -> ReleasePayload:
    """Stage, strictly validate, and atomically publish a release directory."""
    import shutil
    import uuid

    from safetensors.numpy import save_file

    from deep_green.model_spec import (
        TENSOR_SPEC_VERSION,
        parameter_count,
        tensor_names,
    )
    from deep_green.numeric_policy import encode_lexicon

    destination = Path(destination)
    if os.path.lexists(destination):
        raise ReleaseError(f"release destination already exists: {destination}")
    identity = _validate_identity(identity)
    validate_tensors(tensors, width)

    words_path = Path(action_words_path)
    words_data = words_path.read_bytes()
    words = tuple(words_data.decode("utf-8").split())
    raw_digest = hashlib.sha256(words_data).hexdigest()
    if raw_digest != identity["action_space_sha256"]:
        raise ReleaseError("action words bytes do not match the release identity")
    letters, positions = encode_lexicon(list(words))

    config = {
        "schema": TENSOR_SPEC_VERSION,
        "width": width,
        "parameter_count": parameter_count(width),
        "tensor_names": list(tensor_names()),
    }
    staging = destination.parent / (
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir(parents=True)
        save_file(
            {name: np.asarray(value).astype(np.float32) for name, value in tensors.items()},
            str(staging / "weights.safetensors"),
            metadata={"width": str(width)},
        )
        _fsync_file(staging / "weights.safetensors")
        _write_file_fsync(staging / "accepted-guesses.txt", words_data)
        _write_file_fsync(
            staging / "lexicon-indices.npz",
            _deterministic_npz_bytes(letters, positions),
        )
        _write_file_fsync(
            staging / "model-config.json",
            (json.dumps(config, indent=1, sort_keys=True) + "\n").encode("utf-8"),
        )
        files = {}
        for name in sorted(EXPECTED_FILES):
            data = (staging / name).read_bytes()
            files[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "role": EXPECTED_FILES[name],
            }
        manifest = {"schema": RELEASE_SCHEMA, "identity": identity, "files": files}
        _write_file_fsync(
            staging / MANIFEST_NAME,
            (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"),
        )
        _fsync_dir(staging)
        load_release(staging)
        os.rename(staging, destination)
        _fsync_dir(destination.parent)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return load_release(destination)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_release(release_dir) -> ReleasePayload:
    """Load a release only when every byte satisfies the payload contract."""
    release_dir = Path(release_dir)
    if release_dir.is_symlink():
        raise ReleaseError(f"release directory is a symlink: {release_dir}")
    if not release_dir.is_dir():
        raise ReleaseError(f"release directory is not a directory: {release_dir}")

    expected_names = {MANIFEST_NAME, *EXPECTED_FILES}
    children = list(release_dir.iterdir())
    names = {child.name for child in children}
    if names != expected_names:
        unexpected = sorted(names - expected_names)
        missing = sorted(expected_names - names)
        raise ReleaseError(
            f"release children mismatch: missing={missing} unexpected={unexpected}"
        )
    for child in children:
        if child.is_symlink():
            raise ReleaseError(f"release child is a symlink: {child.name}")
        if not child.is_file():
            raise ReleaseError(f"release child is not a regular file: {child.name}")

    manifest_data = _read_regular(
        release_dir / MANIFEST_NAME, MANIFEST_NAME, MAX_MANIFEST_BYTES
    )
    manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
    manifest = _parse_json(manifest_data, MANIFEST_NAME)
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "identity", "files"}:
        raise ReleaseError("manifest must carry exactly schema, identity, files")
    if manifest["schema"] != RELEASE_SCHEMA:
        raise ReleaseError(
            f"manifest schema {manifest['schema']!r} != {RELEASE_SCHEMA!r}"
        )
    identity = _validate_identity(manifest["identity"])

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise ReleaseError("manifest files must carry exactly the four payload names")
    for name, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "role"}:
            raise ReleaseError(f"files entry {name!r} must carry sha256/bytes/role")
        if not _is_sha256(entry["sha256"]):
            raise ReleaseError(f"files entry {name!r} sha256 is not lowercase hex")
        if (
            not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] < 0
        ):
            raise ReleaseError(f"files entry {name!r} bytes must be a nonnegative int")
        if entry["bytes"] > MAX_RELEASE_FILE_BYTES:
            raise ReleaseError(
                f"files entry {name!r} declares {entry['bytes']} bytes, over "
                f"the {MAX_RELEASE_FILE_BYTES}-byte limit"
            )
        if entry["role"] != EXPECTED_FILES[name]:
            raise ReleaseError(
                f"files entry {name!r} role {entry['role']!r} != "
                f"{EXPECTED_FILES[name]!r}"
            )

    payloads = {
        name: _read_checked(release_dir, name, entry)
        for name, entry in files.items()
    }

    words = _parse_action_words(payloads["accepted-guesses.txt"])
    from deep_green.rules import ActionSpace

    try:
        action_space = ActionSpace.parse(words)
    except ValueError as exc:
        raise ReleaseError(f"action space failed to parse: {exc}") from exc
    if action_space.words != words:
        raise ReleaseError("action space is not canonical after parsing")
    raw_digest = hashlib.sha256(payloads["accepted-guesses.txt"]).hexdigest()
    expected_digest = identity["action_space_sha256"]
    if raw_digest != expected_digest or action_space.digest != expected_digest:
        raise ReleaseError("action-space digest does not match the release identity")

    width = parse_model_config(payloads["model-config.json"])

    tensors = _load_weights(payloads["weights.safetensors"])
    validate_tensors(tensors, width)

    letters, positions = _load_lexicon(payloads["lexicon-indices.npz"])
    from deep_green.numeric_policy import encode_lexicon

    expected_letters, expected_positions = encode_lexicon(list(words))
    for name, array, expected in (
        ("letters", letters, expected_letters),
        ("positions", positions, expected_positions),
    ):
        if array.dtype != np.int64:
            raise ReleaseError(f"lexicon {name} dtype {array.dtype} != int64")
        if tuple(array.shape) != (len(words), 5):
            raise ReleaseError(
                f"lexicon {name} shape {tuple(array.shape)} != {(len(words), 5)}"
            )
        if not np.array_equal(array, expected):
            raise ReleaseError(f"lexicon {name} indices do not match the action words")

    return ReleasePayload(
        release_dir=release_dir,
        action_words=words,
        action_space_sha256=expected_digest,
        weights_sha256=files["weights.safetensors"]["sha256"],
        manifest_sha256=manifest_sha256,
        width=width,
        identity=identity,
        tensors=tensors,
    )
