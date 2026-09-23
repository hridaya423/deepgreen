"""Step 8 public proof receipt: generation, determinism, fail-closed loading."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
RUN = REPO_ROOT / "runs/2ddee60886659422"
RELEASE = REPO_ROOT / "artifacts/releases/champion-5022-certified"
PROOF_PATH = REPO_ROOT / "artifacts/proof/champion-5022-certified/proof.json"
CERT_DIR = REPO_ROOT / "artifacts/oracle/openwordle-v1"
FROZEN = REPO_ROOT / "benchmarks/openwordle-v1/frozen-universe"


def _inputs(**overrides):
    inputs = {
        "release_dir": RELEASE,
        "campaign_receipt_path": RUN / "receipt.json",
        "census_paths": {
            "normal": RUN / "evidence/width-320/census-normal.json",
            "hard": RUN / "evidence/width-320/census-hard.json",
        },
        "firewall_path": RUN / "evidence/firewall.json",
        "certificate_paths": {
            "normal": CERT_DIR
            / "openwordle-v1.uniform.normal.full-compiled@5022/certificate.json",
            "hard": CERT_DIR
            / "openwordle-v1.uniform.hard.full-compiled@5022/certificate.json",
        },
        "frozen_root": FROZEN,
    }
    inputs.update(overrides)
    return inputs


def test_generated_receipt_exact_claims():
    from deep_green.proof import generate_proof_receipt
    from deep_green.release import load_release

    receipt = generate_proof_receipt(**_inputs())
    assert receipt["schema"] == "wordle.public-proof/v1"
    assert receipt["status"] == "VERIFIED"
    assert receipt["label"] == "Certified @5022"
    claim = receipt["claim"]
    assert claim["benchmark_id"] == "openwordle-v1"
    assert claim["tier"] == 5022
    assert claim["track"] == "full-compiled"
    assert claim["modes"] == ["normal", "hard"]
    assert claim["exact_tree_action_parity"] is True
    assert claim["objectives"]["normal"] == {
        "losses": 0,
        "score7_total": 18742,
        "worst_score": 6,
    }
    assert claim["objectives"]["hard"] == {
        "losses": 0,
        "score7_total": 20278,
        "worst_score": 6,
    }
    model = receipt["model"]
    assert model["width"] == 320
    assert model["parameter_count"] == 1_550_400
    assert model["weights_bytes"] == 6_203_736
    assert model["score_scale"] == 0.000244140625
    assert 0 < model["max_abs_logit_delta"] <= 1e-5
    assert model["inference_class"] == "numpy-f32-pure-neural"
    payload = load_release(RELEASE)
    assert receipt["release"] == {
        "weights_sha256": payload.weights_sha256,
        "manifest_sha256": payload.manifest_sha256,
    }
    assert receipt["certificates"]["normal"]["winner"] == "tares"
    assert receipt["certificates"]["hard"]["winner"] == "tramp"
    assert receipt["campaign"]["ah_cells_accessed"] is False
    assert receipt["firewall"]["checks_passed"] == 9


def test_proof_receipt_deterministic(tmp_path):
    from deep_green.proof import generate_proof_receipt, write_proof_receipt

    first = generate_proof_receipt(**_inputs())
    second = generate_proof_receipt(**_inputs())
    assert first == second
    out_a = write_proof_receipt(tmp_path / "a/proof.json", first)
    out_b = write_proof_receipt(tmp_path / "b/proof.json", second)
    assert out_a.read_bytes() == out_b.read_bytes()
    digest = hashlib.sha256(out_a.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_on_disk_proof_matches_generated():
    from deep_green.proof import generate_proof_receipt

    on_disk = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    assert on_disk == generate_proof_receipt(**_inputs())


def test_load_public_proof_verified():
    from deep_green.proof import load_public_proof

    proof = load_public_proof(PROOF_PATH, RELEASE)
    assert proof["status"] == "VERIFIED"
    assert proof["label"] == "Certified @5022"
    assert proof["release"]["weights_sha256"]


def test_load_public_proof_fail_closed(tmp_path):
    from deep_green.proof import UNVERIFIED_PROOF, load_public_proof

    assert load_public_proof(tmp_path / "missing.json", RELEASE) == UNVERIFIED_PROOF
    tampered = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    bad = tmp_path / "proof.json"
    tampered["release"]["weights_sha256"] = "0" * 64
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    assert load_public_proof(bad, RELEASE) == UNVERIFIED_PROOF
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"release": "not-a-dict", "status": "VERIFIED"}')
    assert load_public_proof(malformed, RELEASE) == UNVERIFIED_PROOF
    other_release = REPO_ROOT / "artifacts/releases/champion-512"
    if other_release.is_dir():
        assert load_public_proof(PROOF_PATH, other_release) == UNVERIFIED_PROOF


def test_generation_rejects_tampered_inputs(tmp_path):
    from deep_green.proof import ProofError, generate_proof_receipt

    run_copy = tmp_path / "run"
    shutil.copytree(
        RUN / "evidence", run_copy / "evidence", dirs_exist_ok=True
    )
    receipt_copy = run_copy / "receipt.json"
    shutil.copy(RUN / "receipt.json", receipt_copy)

    campaign = json.loads(receipt_copy.read_text(encoding="utf-8"))
    campaign["release"]["weights_sha256"] = "0" * 64
    receipt_copy.write_text(json.dumps(campaign), encoding="utf-8")
    inputs = _inputs(campaign_receipt_path=receipt_copy)
    with pytest.raises(ProofError, match="release hashes"):
        generate_proof_receipt(**inputs)

    census_copy = run_copy / "evidence/width-320/census-normal.json"
    census = json.loads(census_copy.read_text(encoding="utf-8"))
    census["result"]["histogram"][0] += 1
    census_copy.write_text(json.dumps(census), encoding="utf-8")
    inputs = _inputs(
        census_paths={
            "normal": census_copy,
            "hard": RUN / "evidence/width-320/census-hard.json",
        }
    )
    with pytest.raises(ProofError, match="incomplete or illegal"):
        generate_proof_receipt(**inputs)

    firewall_copy = run_copy / "evidence/firewall.json"
    firewall = json.loads(firewall_copy.read_text(encoding="utf-8"))
    firewall["checks"][0]["status"] = "FAIL"
    firewall_copy.write_text(json.dumps(firewall), encoding="utf-8")
    inputs = _inputs(firewall_path=firewall_copy)
    with pytest.raises(ProofError, match="firewall"):
        generate_proof_receipt(**inputs)

    wrong_release = REPO_ROOT / "artifacts/releases/champion-512"
    if wrong_release.is_dir():
        with pytest.raises(ProofError):
            generate_proof_receipt(**_inputs(release_dir=wrong_release))


def test_validate_public_proof_rejects_tampering(tmp_path):
    from deep_green.proof import UNVERIFIED_PROOF, load_public_proof

    original = json.loads(PROOF_PATH.read_text(encoding="utf-8"))

    def tamper(mutate) -> dict:
        receipt = json.loads(json.dumps(original))
        mutate(receipt)
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return load_public_proof(path, RELEASE)

    assert (
        tamper(lambda r: r["claim"].update({"tier": 511})) == UNVERIFIED_PROOF
    )
    assert (
        tamper(lambda r: r["model"].update({"parameter_count": 1550401}))
        == UNVERIFIED_PROOF
    )
    assert (
        tamper(
            lambda r: r["model"].update({"max_abs_logit_delta": 0.5})
        )
        == UNVERIFIED_PROOF
    )
    assert (
        tamper(lambda r: r["firewall"].update({"checks_passed": 8}))
        == UNVERIFIED_PROOF
    )
    assert (
        tamper(
            lambda r: r["claim"]["objectives"]["normal"].update(
                {"score7_total": 1504}
            )
        )
        == UNVERIFIED_PROOF
    )
    assert (
        tamper(
            lambda r: r["censuses"]["normal"].update(
                {"histogram": [1, 91, 362, 57, 1, 0, 0]}
            )
        )
        == UNVERIFIED_PROOF
    )
    assert (
        tamper(lambda r: r.pop("campaign")) == UNVERIFIED_PROOF
    )
    assert (
        tamper(lambda r: r.update({"extra": 1})) == UNVERIFIED_PROOF
    )
