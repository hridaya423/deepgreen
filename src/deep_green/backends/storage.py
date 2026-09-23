"""Backend-neutral model loading and scoring built on safetensors."""

from __future__ import annotations

from pathlib import Path

from safetensors import SafetensorError
from safetensors.numpy import load_file

from deep_green.release import (
    ReleaseError,
    parse_model_config,
    validate_tensors,
)


class ModelArtifactError(ValueError):
    pass


def save_policy_tensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    import numpy
    import safetensors.numpy

    arrays = {}
    for name, value in tensors.items():
        array = numpy.asarray(value)
        arrays[name] = array.astype(numpy.float32)
    safetensors.numpy.save_file(arrays, str(path), metadata=metadata or {})


def load_policy_tensors(path: Path, *, expected_width: int):
    try:
        raw = load_file(str(path))
        validate_tensors(raw, expected_width)
    except (SafetensorError, OSError) as exc:
        raise ModelArtifactError(f"model payload failed to parse: {exc}") from exc
    except ReleaseError as exc:
        raise ModelArtifactError(str(exc)) from exc
    return raw


def read_model_config(path: Path, *, expected_width: int | None = None) -> dict:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ModelArtifactError(f"model config unreadable: {exc}") from exc
    try:
        width = parse_model_config(data)
    except ReleaseError as exc:
        raise ModelArtifactError(str(exc)) from exc
    if expected_width is not None and width != expected_width:
        raise ModelArtifactError(
            f"model width {width} does not match requested {expected_width}"
        )
    import json

    return json.loads(data.decode("utf-8"))
