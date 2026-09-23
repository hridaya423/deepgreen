"""Release pipeline machinery: candidate build, nine gates, clean repro, seal."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from deep_green.release_pipeline import (
    CANDIDATE_SCHEMA,
    CLEAN_CHECK_NAMES,
    CLEAN_INPUT_KEYS,
    CLEAN_SCHEMA,
    EXPECTED_FILE_PATHS,
    GATE_NAMES,
    SEAL_SCHEMA,
    VERIFY_SCHEMA,
    ReleasePipelineError,
    build_candidate,
    run_clean_reproduction,
    seal_candidate,
    verify_accounting,
    verify_candidate,
)

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "runs/9ddd547e7063ffc3"
PAYLOAD = REPO / "artifacts/releases/champion-2095-certified"
PROOF = REPO / "artifacts/proof/champion-2095-certified/proof.json"
FROZEN = REPO / "benchmarks/openwordle-v1/frozen"
ORACLE_DIR = REPO / "artifacts/oracle/openwordle-v1"
NORMAL_CERT = (
    ORACLE_DIR / "openwordle-v1.uniform.normal.full-compiled@2095/certificate.json"
)
HARD_CERT = (
    ORACLE_DIR / "openwordle-v1.uniform.hard.full-compiled@2095/certificate.json"
)
WEB_CONTRACT = REPO / "reports/web/contract.json"
WEB_DIST = REPO / "web/dist"
DESKTOP = REPO / "reports/web/screens/desktop-assist.png"
MOBILE = REPO / "reports/web/screens/mobile-solve.png"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _build(output: Path) -> dict:
    return build_candidate(
        output=output,
        payload_dir=PAYLOAD,
        proof_path=PROOF,
        campaign_receipt=CAMPAIGN / "receipt.json",
        normal_census=CAMPAIGN / "evidence/width-320/census-normal.json",
        hard_census=CAMPAIGN / "evidence/width-320/census-hard.json",
        firewall_evidence=CAMPAIGN / "evidence/firewall.json",
        web_contract=WEB_CONTRACT,
        web_dist=WEB_DIST,
        desktop_screenshot=DESKTOP,
        mobile_screenshot=MOBILE,
        repo_root=REPO,
    )


@pytest.fixture(scope="module")
def candidate(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("cand") / "candidate"
    _build(output)
    return output


def _redeclare(candidate: Path, relative: str) -> None:
    manifest_path = candidate / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (candidate / relative).read_bytes()
    manifest["files"][relative]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["files"][relative]["bytes"] = len(data)
    manifest_path.write_bytes(_canonical(manifest))


def _mutate(candidate: Path, relative: str, mutate) -> None:
    path = candidate / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_canonical(value))
    _redeclare(candidate, relative)


def _clean_dict(candidate: Path, **overrides) -> dict:
    receipt = {
        "schema": CLEAN_SCHEMA,
        "status": "PASS",
        "source_revision": "0" * 40,
        "candidate_manifest_sha256": _sha(candidate / "candidate-manifest.json"),
        "declared_inputs": {
            key: {"path": f"inputs/{key}", "sha256": "0" * 64}
            for key in CLEAN_INPUT_KEYS
        },
        "checks": [
            {"name": name, "status": "PASS"} for name in CLEAN_CHECK_NAMES
        ],
    }
    receipt.update(overrides)
    return receipt


def _verify(candidate: Path, **kwargs) -> dict:
    options = {
        "candidate": candidate,
        "frozen_root": FROZEN,
        "normal_certificate": NORMAL_CERT,
        "hard_certificate": HARD_CERT,
        "clean_receipt": _clean_dict(candidate),
        "run_live_firewall": False,
    }
    options.update(kwargs)
    return verify_candidate(**options)


def _gate(receipt: dict, name: str) -> dict:
    return next(g for g in receipt["gates"] if g["name"] == name)


class TestBuildCandidate:
    def test_builds_exact_tree(self, candidate):
        manifest = json.loads(
            (candidate / "candidate-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema"] == CANDIDATE_SCHEMA
        assert set(manifest) == {"schema", "bindings", "files"}
        assert set(manifest["files"]) == EXPECTED_FILE_PATHS - {
            "candidate-manifest.json"
        }
        on_disk = {
            str(p.relative_to(candidate))
            for p in candidate.rglob("*")
            if p.is_file()
        }
        assert on_disk == EXPECTED_FILE_PATHS

    def test_payload_and_proof_self_verify(self, candidate):
        from deep_green.proof import load_public_proof
        from deep_green.release import load_release

        payload = load_release(candidate / "payload")
        assert payload.weights_sha256 == (
            "a44c5336d0d36b5c632c0fc0b91aec140e7c08a033f815569d1818fa6c0ec0ae"
        )
        proof = load_public_proof(
            candidate / "evidence/proof.json", candidate / "payload"
        )
        assert proof["status"] == "VERIFIED"

    def test_no_forbidden_material(self, candidate):
        for path in candidate.rglob("*"):
            relative = str(path.relative_to(candidate)).lower()
            for token in ("answer", "oracle", "train", "checkpoint", "cache"):
                assert token not in relative
        runtime = candidate / "runtime/src/deep_green"
        assert not (runtime / "adaptive.py").exists()
        assert not (runtime / "oracle").exists()

    def test_refuses_existing_output(self, candidate):
        with pytest.raises(ReleasePipelineError):
            _build(candidate)

    def test_refuses_missing_source(self, tmp_path):
        with pytest.raises(ReleasePipelineError):
            build_candidate(
                output=tmp_path / "candidate",
                payload_dir=PAYLOAD,
                proof_path=tmp_path / "absent.json",
                campaign_receipt=CAMPAIGN / "receipt.json",
                normal_census=CAMPAIGN / "evidence/width-320/census-normal.json",
                hard_census=CAMPAIGN / "evidence/width-320/census-hard.json",
                firewall_evidence=CAMPAIGN / "evidence/firewall.json",
                web_contract=WEB_CONTRACT,
                web_dist=WEB_DIST,
                desktop_screenshot=DESKTOP,
                mobile_screenshot=MOBILE,
                repo_root=REPO,
            )

    def test_accounting_rejects_undeclared_file(self, candidate):
        extra = candidate / "evidence/extra.json"
        extra.write_text("{}")
        try:
            with pytest.raises(ReleasePipelineError):
                verify_accounting(candidate)
        finally:
            extra.unlink()
        verify_accounting(candidate)


class TestVerifyGates:
    def test_all_nine_pass(self, candidate):
        receipt = _verify(candidate)
        assert receipt["schema"] == VERIFY_SCHEMA
        assert receipt["status"] == "PASS"
        assert [g["name"] for g in receipt["gates"]] == list(GATE_NAMES)
        assert receipt["candidate_manifest_sha256"] == _sha(
            candidate / "candidate-manifest.json"
        )

    def test_live_firewall_gate(self, candidate):
        receipt = _verify(candidate, run_live_firewall=True)
        assert _gate(receipt, "firewall")["status"] == "PASS"
        assert receipt["status"] == "PASS"

    def test_benchmark_gate_fails_on_bad_frozen(self, candidate, tmp_path):
        receipt = _verify(candidate, frozen_root=tmp_path)
        assert _gate(receipt, "benchmark")["status"] == "FAIL"

    def test_oracle_gate_fails_on_swapped_cert(self, candidate):
        receipt = _verify(candidate, normal_certificate=HARD_CERT)
        assert _gate(receipt, "oracle")["status"] == "FAIL"

    def test_perturbed_payload_fails_parity_and_census(self, tmp_path):
        from safetensors.numpy import load_file, save_file

        candidate = tmp_path / "candidate"
        _build(candidate)

        weights = candidate / "payload/weights.safetensors"
        tensors = load_file(str(weights))
        key = min(tensors)
        tensors[key] = tensors[key].copy()
        tensors[key].flat[0] += 1.0
        save_file(tensors, str(weights), metadata={"width": "128"})

        manifest_path = candidate / "payload/payload-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        weights_data = weights.read_bytes()
        manifest["files"]["weights.safetensors"] = {
            "sha256": hashlib.sha256(weights_data).hexdigest(),
            "bytes": len(weights_data),
            "role": "neural-payload",
        }
        manifest_path.write_bytes(
            (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
        )
        _redeclare(candidate, "payload/weights.safetensors")
        _redeclare(candidate, "payload/payload-manifest.json")

        candidate_manifest_path = candidate / "candidate-manifest.json"
        candidate_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
        candidate_manifest["bindings"]["weights_sha256"] = manifest["files"][
            "weights.safetensors"
        ]["sha256"]
        candidate_manifest["bindings"]["manifest_sha256"] = _sha(manifest_path)
        candidate_manifest_path.write_bytes(_canonical(candidate_manifest))

        receipt = _verify(candidate)
        assert _gate(receipt, "payload")["status"] == "PASS"
        assert _gate(receipt, "accounting")["status"] == "PASS"
        assert _gate(receipt, "benchmark")["status"] == "PASS"
        assert _gate(receipt, "model-parity")["status"] == "FAIL"
        assert _gate(receipt, "census")["status"] == "FAIL"
        assert receipt["status"] == "FAIL"

    def test_model_parity_gate_fails_on_mutated_campaign(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        _mutate(
            candidate,
            "evidence/campaign-receipt.json",
            lambda value: value["backend_parity"].__setitem__("states", 1040),
        )
        receipt = _verify(candidate)
        assert _gate(receipt, "model-parity")["status"] == "FAIL"

    def test_census_gate_fails_on_mutated_census(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        _mutate(
            candidate,
            "evidence/census-normal.json",
            lambda value: value["result"].__setitem__(
                "score7_total", value["result"]["score7_total"] + 1
            ),
        )
        receipt = _verify(candidate)
        assert _gate(receipt, "census")["status"] == "FAIL"

    def test_firewall_gate_fails_on_mutated_evidence(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        _mutate(
            candidate,
            "evidence/firewall.json",
            lambda value: value.__setitem__("status", "FAIL"),
        )
        receipt = _verify(candidate)
        assert _gate(receipt, "firewall")["status"] == "FAIL"

    def test_web_gate_fails_on_mutated_dist(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        main = candidate / "web/main.js"
        main.write_bytes(main.read_bytes() + b"//tampered\n")
        _redeclare(candidate, "web/main.js")
        receipt = _verify(candidate)
        assert _gate(receipt, "web")["status"] == "FAIL"

    def test_accounting_gate_fails_on_extra_file(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        (candidate / "stray.txt").write_text("x")
        receipt = _verify(candidate)
        assert _gate(receipt, "accounting")["status"] == "FAIL"

    def test_clean_gate_fails_when_absent(self, candidate):
        receipt = _verify(candidate, clean_receipt=None)
        assert _gate(receipt, "clean-reproduction")["status"] == "FAIL"
        assert receipt["status"] == "FAIL"

    def test_clean_gate_fails_on_unbound_receipt(self, candidate):
        receipt = _verify(
            candidate,
            clean_receipt=_clean_dict(
                candidate, candidate_manifest_sha256="f" * 64
            ),
        )
        assert _gate(receipt, "clean-reproduction")["status"] == "FAIL"

    def test_role_tamper_fails_accounting(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        manifest_path = candidate / "candidate-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["evidence/proof.json"]["role"] = "forged"
        manifest_path.write_bytes(_canonical(manifest))
        with pytest.raises(ReleasePipelineError):
            verify_accounting(candidate)
        receipt = _verify(candidate)
        assert _gate(receipt, "accounting")["status"] == "FAIL"

    def test_binding_shape_and_hash_tamper(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        manifest_path = candidate / "candidate-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bindings"]["weights_sha256"] = "Z" * 64
        manifest_path.write_bytes(_canonical(manifest))
        with pytest.raises(ReleasePipelineError):
            verify_accounting(candidate)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bindings"].pop("weights_sha256")
        manifest_path.write_bytes(_canonical(manifest))
        with pytest.raises(ReleasePipelineError):
            verify_accounting(candidate)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bindings"]["weights_sha256"] = "0" * 64
        manifest["bindings"]["census_sha256s"] = {"normal": "0" * 64}
        manifest_path.write_bytes(_canonical(manifest))
        with pytest.raises(ReleasePipelineError):
            verify_accounting(candidate)

    def test_nan_delta_fails_model_parity(self, tmp_path):
        candidate = tmp_path / "candidate"
        _build(candidate)
        _mutate(
            candidate,
            "evidence/campaign-receipt.json",
            lambda value: value["backend_parity"].__setitem__(
                "max_abs_delta_torch_numpy", float("nan")
            ),
        )
        receipt = _verify(candidate)
        assert _gate(receipt, "model-parity")["status"] == "FAIL"
        assert receipt["status"] == "FAIL"


class TestCleanReproduction:
    def _fake_repo(self, tmp_path: Path) -> Path:
        cache = tmp_path / "artifacts/oracle/build-cache"
        (cache / "src").mkdir(parents=True)
        (cache / "deep-green-built").write_bytes(b"binary")
        (cache / "build-marker.json").write_text("{}")
        (cache / "src/wordle.cpp").write_text("//src")
        for cert in (NORMAL_CERT, HARD_CERT):
            sha = json.loads(cert.read_text(encoding="utf-8"))["binding"][
                "executable_sha256"
            ]
            bound = cache / sha
            bound.mkdir(parents=True, exist_ok=True)
            (bound / "deep-green-built").write_bytes(b"binary")
            (bound / "build-marker.json").write_text("{}")
        corpus = tmp_path / "artifacts/corpus/openwordle-v1-candidate"
        corpus.mkdir(parents=True)
        (corpus / "manifest.json").write_text("{}")
        (corpus / "words").mkdir()
        (corpus / "words/accepted.txt").write_text("stare\n")
        return tmp_path

    def test_dirty_source_fails_closed(self, candidate, tmp_path, monkeypatch):
        repo = self._fake_repo(tmp_path / "repo")
        repo.mkdir(parents=True, exist_ok=True)

        def fake(argv, *, cwd, timeout=3600):
            stdout = " M dirty\n" if "status" in argv else "0" * 40 + "\n"
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        monkeypatch.setattr(
            "deep_green.release_pipeline._run_command", fake
        )
        receipt = run_clean_reproduction(
            repo_root=repo,
            candidate=candidate,
            output=tmp_path / "clean.json",
            frozen_root=FROZEN,
            normal_certificate=NORMAL_CERT,
            hard_certificate=HARD_CERT,
        )
        assert receipt["schema"] == CLEAN_SCHEMA
        assert receipt["status"] == "FAIL"
        by_name = {c["name"]: c["status"] for c in receipt["checks"]}
        assert by_name["source-clean"] == "FAIL"
        assert by_name["census-hard"] == "SKIP"
        assert [c["name"] for c in receipt["checks"]] == list(CLEAN_CHECK_NAMES)

    def test_all_pass_receipt(self, candidate, tmp_path, monkeypatch):
        repo = self._fake_repo(tmp_path / "repo")
        repo.mkdir(parents=True, exist_ok=True)
        removed = []

        def fake(argv, *, cwd, timeout=3600):
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
            if argv[:2] == ["npm", "run"] and argv[2] == "build":
                dist = Path(cwd) / "dist"
                dist.mkdir(parents=True, exist_ok=True)
                for name in ("index.html", "main.js", "styles.css"):
                    (dist / name).write_bytes(
                        (candidate / "web" / name).read_bytes()
                    )
            if "worktree" in argv and "remove" in argv:
                removed.append(True)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(
            "deep_green.release_pipeline._run_command", fake
        )
        output = tmp_path / "clean.json"
        receipt = run_clean_reproduction(
            repo_root=repo,
            candidate=candidate,
            output=output,
            frozen_root=FROZEN,
            normal_certificate=NORMAL_CERT,
            hard_certificate=HARD_CERT,
        )
        assert receipt["status"] == "PASS"
        assert receipt["source_revision"] == "a" * 40
        assert receipt["candidate_manifest_sha256"] == _sha(
            candidate / "candidate-manifest.json"
        )
        assert set(receipt["declared_inputs"]) == set(CLEAN_INPUT_KEYS)
        assert [c["name"] for c in receipt["checks"]] == list(CLEAN_CHECK_NAMES)
        assert all(c["status"] == "PASS" for c in receipt["checks"])
        assert removed == [True]
        assert json.loads(output.read_text()) == receipt

    def test_failed_step_records_fail(self, candidate, tmp_path, monkeypatch):
        repo = self._fake_repo(tmp_path / "repo")
        repo.mkdir(parents=True, exist_ok=True)

        def fake(argv, *, cwd, timeout=3600):
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, "b" * 40 + "\n", "")
            rc = 1 if "ruff" in argv else 0
            return subprocess.CompletedProcess(argv, rc, "", "")

        monkeypatch.setattr(
            "deep_green.release_pipeline._run_command", fake
        )
        receipt = run_clean_reproduction(
            repo_root=repo,
            candidate=candidate,
            output=tmp_path / "clean.json",
            frozen_root=FROZEN,
            normal_certificate=NORMAL_CERT,
            hard_certificate=HARD_CERT,
        )
        by_name = {c["name"]: c["status"] for c in receipt["checks"]}
        assert receipt["status"] == "FAIL"
        assert by_name["ruff"] == "FAIL"
        assert by_name["uv-sync"] == "PASS"
        assert by_name["pytest"] == "SKIP"

    def test_fail_writes_sanitized_bounded_log(
        self, candidate, tmp_path, monkeypatch
    ):
        repo = self._fake_repo(tmp_path / "repo")
        repo.mkdir(parents=True, exist_ok=True)

        def fake(argv, *, cwd, timeout=3600):
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, "c" * 40 + "\n", "")
            if "ruff" in argv:
                return subprocess.CompletedProcess(
                    argv, 1, "x" * 9000 + f"\nE000 failed under {cwd}\n", ""
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(
            "deep_green.release_pipeline._run_command", fake
        )
        output = tmp_path / "clean.json"
        receipt = run_clean_reproduction(
            repo_root=repo,
            candidate=candidate,
            output=output,
            frozen_root=FROZEN,
            normal_certificate=NORMAL_CERT,
            hard_certificate=HARD_CERT,
        )
        assert receipt["status"] == "FAIL"
        log = Path(str(output) + ".log")
        assert log.is_file()
        text = log.read_text()
        assert "== ruff: uv run --no-sync ruff check ." in text
        assert "E000 failed under <TEMP>/checkout" in text
        assert str(tmp_path) not in text
        assert len(text) < 9000

    def test_pass_writes_no_log(self, candidate, tmp_path, monkeypatch):
        repo = self._fake_repo(tmp_path / "repo")
        repo.mkdir(parents=True, exist_ok=True)

        def fake(argv, *, cwd, timeout=3600):
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
            if argv[:2] == ["npm", "run"] and argv[2] == "build":
                dist = Path(cwd) / "dist"
                dist.mkdir(parents=True, exist_ok=True)
                for name in ("index.html", "main.js", "styles.css"):
                    (dist / name).write_bytes(
                        (candidate / "web" / name).read_bytes()
                    )
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(
            "deep_green.release_pipeline._run_command", fake
        )
        output = tmp_path / "clean.json"
        receipt = run_clean_reproduction(
            repo_root=repo,
            candidate=candidate,
            output=output,
            frozen_root=FROZEN,
            normal_certificate=NORMAL_CERT,
            hard_certificate=HARD_CERT,
        )
        assert receipt["status"] == "PASS"
        assert not Path(str(output) + ".log").exists()


@pytest.fixture(scope="module")
def sealed_inputs(candidate):
    clean = _clean_dict(candidate)
    verification = verify_candidate(
        candidate=candidate,
        frozen_root=FROZEN,
        normal_certificate=NORMAL_CERT,
        hard_certificate=HARD_CERT,
        clean_receipt=clean,
        run_live_firewall=False,
    )
    return candidate, verification, clean


class TestSeal:
    def test_seal_deterministic_and_bound(self, sealed_inputs, tmp_path):
        candidate, verification, clean = sealed_inputs
        assert verification["status"] == "PASS"
        out_a = tmp_path / "seal-a"
        out_b = tmp_path / "seal-b"
        seal_a = seal_candidate(
            candidate=candidate,
            verification_receipt=verification,
            clean_receipt=clean,
            output_dir=out_a,
        )
        seal_b = seal_candidate(
            candidate=candidate,
            verification_receipt=verification,
            clean_receipt=clean,
            output_dir=out_b,
        )
        assert seal_a == seal_b
        assert seal_a["schema"] == SEAL_SCHEMA
        assert seal_a["status"] == "SEALED"
        archive_a = (out_a / "release.tar.gz").read_bytes()
        archive_b = (out_b / "release.tar.gz").read_bytes()
        assert archive_a == archive_b
        assert seal_a["archive_sha256"] == hashlib.sha256(archive_a).hexdigest()
        assert seal_a["archive_bytes"] == len(archive_a)
        assert seal_a["release"]["weights_sha256"] == (
            "a44c5336d0d36b5c632c0fc0b91aec140e7c08a033f815569d1818fa6c0ec0ae"
        )
        assert seal_a["proof_sha256"] == _sha(PROOF)
        assert _sha(out_a / "seal.json") == _sha(out_b / "seal.json")

    def test_tar_metadata_normalized(self, sealed_inputs, tmp_path):
        candidate, verification, clean = sealed_inputs
        out = tmp_path / "seal"
        seal_candidate(
            candidate=candidate,
            verification_receipt=verification,
            clean_receipt=clean,
            output_dir=out,
        )
        with tarfile.open(out / "release.tar.gz", "r:gz") as tar:
            members = tar.getmembers()
        names = [m.name for m in members]
        assert names == sorted(names)
        assert names[0] == "deep-green"
        assert members[0].isdir()
        for member in members:
            assert member.uid == 0 and member.gid == 0
            assert member.uname == "" and member.gname == ""
            assert member.mtime == 0
            if member.isdir():
                assert member.mode == 0o755
            else:
                assert member.mode == 0o644
            assert not member.issym() and not member.islnk()
        files = {m.name for m in members if m.isfile()}
        assert "deep-green/clean-reproduction.json" in files
        assert "deep-green/release-verification.json" in files
        assert "deep-green/payload/weights.safetensors" in files

    def test_reseal_refused(self, sealed_inputs, tmp_path):
        candidate, verification, clean = sealed_inputs
        out = tmp_path / "seal"
        seal_candidate(
            candidate=candidate,
            verification_receipt=verification,
            clean_receipt=clean,
            output_dir=out,
        )
        with pytest.raises(ReleasePipelineError):
            seal_candidate(
                candidate=candidate,
                verification_receipt=verification,
                clean_receipt=clean,
                output_dir=out,
            )

    def test_failing_verification_refused(self, sealed_inputs, tmp_path):
        candidate, verification, clean = sealed_inputs
        bad = dict(verification)
        bad["status"] = "FAIL"
        with pytest.raises(ReleasePipelineError):
            seal_candidate(
                candidate=candidate,
                verification_receipt=bad,
                clean_receipt=clean,
                output_dir=tmp_path / "seal",
            )

    def test_verification_extra_keys_refused(self, sealed_inputs, tmp_path):
        candidate, verification, clean = sealed_inputs
        bad = dict(verification, extra="field")
        with pytest.raises(ReleasePipelineError):
            seal_candidate(
                candidate=candidate,
                verification_receipt=bad,
                clean_receipt=clean,
                output_dir=tmp_path / "seal-a",
            )
        bad = dict(verification)
        bad["gates"] = [dict(gate, extra=1) for gate in verification["gates"]]
        with pytest.raises(ReleasePipelineError):
            seal_candidate(
                candidate=candidate,
                verification_receipt=bad,
                clean_receipt=clean,
                output_dir=tmp_path / "seal-b",
            )
