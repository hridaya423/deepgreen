"""Strict release loader: every tamper class fails closed."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
FIXTURE = REPO_ROOT / "tests/fixtures/release"
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

REQUIRED = (
    "payload-manifest.json",
    "accepted-guesses.txt",
    "lexicon-indices.npz",
    "model-config.json",
    "weights.safetensors",
)


@pytest.fixture()
def release(tmp_path):
    target = tmp_path / "release"
    shutil.copytree(FIXTURE, target)
    return target


def _manifest(release_dir):
    return json.loads(
        (release_dir / "payload-manifest.json").read_text(encoding="utf-8")
    )


def _rehash(release_dir, name):
    manifest = _manifest(release_dir)
    path = release_dir / name
    manifest["files"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["files"][name]["bytes"] = path.stat().st_size
    (release_dir / "payload-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_manifest(release_dir, mutate):
    manifest = _manifest(release_dir)
    mutate(manifest)
    (release_dir / "payload-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_weights(release_dir, mutate):
    import numpy as np
    from safetensors.numpy import load_file, save_file

    arrays = load_file(str(release_dir / "weights.safetensors"))
    arrays = mutate(arrays)
    save_file(arrays, str(release_dir / "weights.safetensors"))
    _rehash(release_dir, "weights.safetensors")
    del np


def _load(release_dir):
    from deep_green.release import load_release

    return load_release(release_dir)


def _expect_error(release_dir, match):
    from deep_green.release import ReleaseError

    with pytest.raises(ReleaseError, match=match):
        _load(release_dir)


def test_clean_fixture_loads_with_exact_fields(release):
    payload = _load(release)
    assert len(payload.action_words) == 5022
    assert payload.width == 64
    assert payload.identity["benchmark_id"] == "openwordle-v1"
    assert payload.identity["inference_class"] == "numpy-f32-pure-neural"
    assert payload.identity["track"] == "full-compiled"
    assert payload.weights_sha256 == hashlib.sha256(
        (release / "weights.safetensors").read_bytes()
    ).hexdigest()
    assert payload.manifest_sha256 == hashlib.sha256(
        (release / "payload-manifest.json").read_bytes()
    ).hexdigest()
    assert payload.action_space_sha256 == payload.identity["action_space_sha256"]
    from deep_green.model_spec import expected_shapes, tensor_names

    assert set(payload.tensors) == set(tensor_names())
    for name, array in payload.tensors.items():
        assert tuple(array.shape) == expected_shapes(64)[name]


def test_release_dir_symlink_rejected(release, tmp_path):
    link = tmp_path / "release-link"
    link.symlink_to(release)
    _expect_error(link, "symlink")


def test_release_dir_not_directory_rejected(tmp_path):
    path = tmp_path / "release-file"
    path.write_text("x", encoding="utf-8")
    _expect_error(path, "directory")
    _expect_error(tmp_path / "missing", "directory")


def test_undeclared_file_rejected(release):
    (release / "smuggled.bin").write_bytes(b"x")
    _expect_error(release, "unexpected|extra")


def test_undeclared_directory_rejected(release):
    (release / "nested").mkdir()
    _expect_error(release, "unexpected|extra|directory")


def test_dotfile_rejected(release):
    (release / ".hidden").write_bytes(b"x")
    _expect_error(release, "unexpected|extra|dotfile")


def test_symlink_child_rejected(release, tmp_path):
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"x" * 64)
    (release / "weights.safetensors").unlink()
    (release / "weights.safetensors").symlink_to(outside)
    _expect_error(release, "symlink")


def test_symlink_manifest_rejected(release, tmp_path):
    outside = tmp_path / "manifest.json"
    outside.write_text((release / "payload-manifest.json").read_text())
    (release / "payload-manifest.json").unlink()
    (release / "payload-manifest.json").symlink_to(outside)
    _expect_error(release, "symlink")


def test_digest_mismatch_rejected(release):
    words_path = release / "accepted-guesses.txt"
    words_path.write_bytes(words_path.read_bytes().replace(b"aback", b"abaca", 1))
    _expect_error(release, "sha256|digest")


def test_file_size_mismatch_rejected(release):
    (release / "weights.safetensors").write_bytes(b"tampered")
    _expect_error(release, "bytes|sha256|digest")


def test_byte_count_mismatch_rejected(release):
    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["bytes"] += 1

    _rewrite_manifest(release, mutate)
    _expect_error(release, "bytes|size")


def test_role_mismatch_rejected(release):
    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["role"] = "public-lexicon"

    _rewrite_manifest(release, mutate)
    _expect_error(release, "role")


def test_manifest_schema_mismatch_rejected(release):
    def mutate(manifest):
        manifest["schema"] = "wordle.release-payload/v9"

    _rewrite_manifest(release, mutate)
    _expect_error(release, "schema")


def test_manifest_extra_top_key_rejected(release):
    def mutate(manifest):
        manifest["extra"] = {}

    _rewrite_manifest(release, mutate)
    _expect_error(release, "manifest")


def test_identity_mismatch_rejected(release):
    def mutate(manifest):
        manifest["identity"]["action_space_sha256"] = "0" * 64

    _rewrite_manifest(release, mutate)
    _expect_error(release, "identity|action|digest|sha256")


def test_identity_empty_value_rejected(release):
    def mutate(manifest):
        manifest["identity"]["benchmark_id"] = ""

    _rewrite_manifest(release, mutate)
    _expect_error(release, "identity")


def test_identity_bad_sha_rejected(release):
    def mutate(manifest):
        manifest["identity"]["rules_sha256"] = "0" * 63 + "G"

    _rewrite_manifest(release, mutate)
    _expect_error(release, "identity|sha256")


def test_files_entry_extra_key_rejected(release):
    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["comment"] = "hi"

    _rewrite_manifest(release, mutate)
    _expect_error(release, "files|entry")


def test_bytes_bool_rejected(release):
    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["bytes"] = True

    _rewrite_manifest(release, mutate)
    _expect_error(release, "bytes")


def test_model_config_mismatch_rejected(release):
    config = json.loads((release / "model-config.json").read_text(encoding="utf-8"))
    config["parameter_count"] += 1
    (release / "model-config.json").write_text(
        json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash(release, "model-config.json")
    _expect_error(release, "config|parameter")


def test_model_config_width_string_rejected(release):
    config = json.loads((release / "model-config.json").read_text(encoding="utf-8"))
    config["width"] = "64"
    (release / "model-config.json").write_text(
        json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash(release, "model-config.json")
    _expect_error(release, "width")


def test_tensor_name_mismatch_rejected(release):
    def mutate(arrays):
        arrays = dict(arrays)
        arrays.pop("score.bias")
        return arrays

    _rewrite_weights(release, mutate)
    _expect_error(release, "tensor")


def test_tensor_extra_name_rejected(release):
    def mutate(arrays):
        import numpy as np

        arrays = dict(arrays)
        arrays["smuggled.tensor"] = np.zeros(4, dtype=np.float32)
        return arrays

    _rewrite_weights(release, mutate)
    _expect_error(release, "tensor")


def test_tensor_shape_mismatch_rejected(release):
    def mutate(arrays):
        import numpy as np

        arrays = dict(arrays)
        arrays["score.bias"] = np.zeros(arrays["score.bias"].shape[0] + 1, np.float32)
        return arrays

    _rewrite_weights(release, mutate)
    _expect_error(release, "shape")


def test_tensor_dtype_mismatch_rejected(release):
    def mutate(arrays):
        arrays = dict(arrays)
        arrays["score.bias"] = arrays["score.bias"].astype("float64")
        return arrays

    _rewrite_weights(release, mutate)
    _expect_error(release, "dtype|float32")


def test_nonfinite_tensor_rejected(release):
    def mutate(arrays):
        arrays = dict(arrays)
        arrays["score.bias"] = arrays["score.bias"].copy()
        arrays["score.bias"][0] = float("nan")
        return arrays

    _rewrite_weights(release, mutate)
    _expect_error(release, "finite")


def test_npz_mismatch_rejected(release):
    import numpy as np

    payload = np.load(release / "lexicon-indices.npz")
    letters = payload["letters"].copy()
    letters[0, 0] = (letters[0, 0] + 1) % 26
    np.savez_compressed(
        release / "lexicon-indices.npz",
        letters=letters,
        positions=payload["positions"],
    )
    _rehash(release, "lexicon-indices.npz")
    _expect_error(release, "lexicon|indices|npz")


def test_npz_extra_key_rejected(release):
    import numpy as np

    payload = np.load(release / "lexicon-indices.npz")
    np.savez_compressed(
        release / "lexicon-indices.npz",
        letters=payload["letters"],
        positions=payload["positions"],
        extra=np.zeros(1, dtype=np.int64),
    )
    _rehash(release, "lexicon-indices.npz")
    _expect_error(release, "lexicon|indices|npz")


def test_words_blank_line_rejected(release):
    (release / "accepted-guesses.txt").write_bytes(
        (release / "accepted-guesses.txt").read_bytes() + b"\n"
    )
    _rehash(release, "accepted-guesses.txt")
    manifest = _manifest(release)
    manifest["identity"]["action_space_sha256"] = hashlib.sha256(
        (release / "accepted-guesses.txt").read_bytes()
    ).hexdigest()
    (release / "payload-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    _expect_error(release, "blank|word|canonical")


def test_verify_release_clean_and_tampered(release):
    from build_fixture import verify_release

    assert verify_release(release) == []
    (release / "weights.safetensors").write_bytes(b"tampered")
    issues = verify_release(release)
    assert len(issues) == 1 and issues[0]


def test_load_never_calls_safetensors_load_file(release, monkeypatch):
    import safetensors.numpy as st_numpy

    def _explode(_path):
        raise AssertionError("load_file must not read the path after verification")

    monkeypatch.setattr(st_numpy, "load_file", _explode)
    from deep_green.release import load_release

    assert load_release(release).width == 64


def test_manifest_rejects_nonfinite_constant(release):
    manifest = _manifest(release)
    manifest["schema"] = float("nan")
    (release / "payload-manifest.json").write_text(json.dumps(manifest))
    _expect_error(release, "non-finite|constant")


def test_config_rejects_nonfinite_constant(release):
    config_path = release / "model-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["width"] = float("nan")
    config_path.write_text(json.dumps(config))
    _rehash(release, "model-config.json")
    _expect_error(release, "non-finite|constant")


def test_declared_payload_over_limit_rejected(release):
    from deep_green.release import MAX_RELEASE_FILE_BYTES

    def mutate(manifest):
        manifest["files"]["weights.safetensors"]["bytes"] = (
            MAX_RELEASE_FILE_BYTES + 1
        )

    _rewrite_manifest(release, mutate)
    _expect_error(release, "limit|bytes")


def test_npz_extra_path_member_rejected(release):
    import io
    import zipfile

    with zipfile.ZipFile(release / "lexicon-indices.npz") as source:
        members = {name: source.read(name) for name in source.namelist()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for name, data in members.items():
            out.writestr(name, data)
        out.writestr("../evil.npy", b"x")
    (release / "lexicon-indices.npz").write_bytes(buf.getvalue())
    _rehash(release, "lexicon-indices.npz")
    _expect_error(release, "members|lexicon|npz")


def test_npz_oversized_uncompressed_member_rejected(release):
    import io
    import struct
    import zipfile

    from deep_green.release import MAX_LEXICON_UNCOMPRESSED_BYTES

    with zipfile.ZipFile(release / "lexicon-indices.npz") as source:
        members = {name: source.read(name) for name in source.namelist()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for name, data in members.items():
            out.writestr(name, data)
    data = bytearray(buf.getvalue())
    cd = data.find(b"PK\x01\x02")
    assert cd > 0
    struct.pack_into("<I", data, cd + 24, MAX_LEXICON_UNCOMPRESSED_BYTES + 1)
    (release / "lexicon-indices.npz").write_bytes(bytes(data))
    _rehash(release, "lexicon-indices.npz")
    _expect_error(release, "decompresses|limit|lexicon")
