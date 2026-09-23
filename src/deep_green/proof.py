"""Deterministic public proof receipt derived from verified release evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from deep_green.model_spec import parameter_count
from deep_green.release import load_release

PROOF_SCHEMA = "wordle.public-proof/v1"
UNVERIFIED_PROOF = {
    "schema": PROOF_SCHEMA,
    "status": "UNVERIFIED",
    "label": "Unverified build",
}


REQUIRED_FIREWALL_CHECKS = frozenset(
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


class ProofError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofError(f"cannot read proof input {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ProofError(f"proof input {path} is not an object")
    return value


def generate_proof_receipt(
    *,
    release_dir: Path,
    campaign_receipt_path: Path,
    census_paths: dict[str, Path],
    firewall_path: Path,
    certificate_paths: dict[str, Path],
    frozen_root: Path,
) -> dict:
    from deep_green.oracle.verifier import verify_certificate

    modes = ("normal", "hard")
    if set(census_paths) != set(modes) or set(certificate_paths) != set(modes):
        raise ProofError("proof inputs must contain exactly normal and hard")
    release_dir = Path(release_dir)
    payload = load_release(release_dir)
    campaign_receipt_path = Path(campaign_receipt_path)
    campaign = _load_json(campaign_receipt_path)
    if campaign.get("schema") != "wordle.adaptive-campaign/v1":
        raise ProofError("adaptive campaign receipt schema mismatch")
    if campaign.get("status") != "PARITY" or campaign.get("selected_width") != payload.width:
        raise ProofError("adaptive campaign did not select this release")
    if campaign.get("ah_cells_accessed") is not False:
        raise ProofError("adaptive campaign touched held-out cells")
    release_binding = campaign.get("release")
    if not isinstance(release_binding, dict):
        raise ProofError("adaptive campaign has no release binding")
    if (
        release_binding.get("weights_sha256") != payload.weights_sha256
        or release_binding.get("manifest_sha256") != payload.manifest_sha256
    ):
        raise ProofError("adaptive campaign release hashes do not match")
    backend = campaign.get("backend_parity")
    required_backend = {
        "states",
        "tolerance",
        "max_abs_delta_torch_numpy",
        "max_abs_delta_torch_mlx",
        "max_abs_delta_numpy_mlx",
        "torch_disagreements",
        "numpy_disagreements",
        "mlx_disagreements",
    }
    if not isinstance(backend, dict) or set(backend) != required_backend:
        raise ProofError("adaptive campaign backend report is malformed")
    if backend["tolerance"] != 1e-5:
        raise ProofError("adaptive campaign backend coverage is incomplete")
    deltas = (
        backend["max_abs_delta_torch_numpy"],
        backend["max_abs_delta_torch_mlx"],
        backend["max_abs_delta_numpy_mlx"],
    )
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in deltas):
        raise ProofError("adaptive campaign backend deltas are malformed")
    if max(deltas) > backend["tolerance"]:
        raise ProofError("adaptive campaign backend deltas exceed tolerance")
    if any(backend[name] != 0 for name in (
        "torch_disagreements",
        "numpy_disagreements",
        "mlx_disagreements",
    )):
        raise ProofError("adaptive campaign backend actions disagree")

    from deep_green.oracle.tree_stats import count_policy_states

    certificates = {}
    censuses = {}
    objectives = {}
    tier = None
    expected_states = 0
    for mode in modes:
        certificate_path = Path(certificate_paths[mode])
        verify_certificate(certificate_path, frozen_root=Path(frozen_root))
        certificate = _load_json(certificate_path)
        certificate_sha = _sha256(certificate_path)
        certificate_tier = certificate["identity"].get("tier")
        if (
            certificate["identity"].get("mode") != mode
            or not isinstance(certificate_tier, int)
            or isinstance(certificate_tier, bool)
            or certificate_tier <= 0
            or (tier is not None and certificate_tier != tier)
        ):
            raise ProofError(f"{mode} certificate identity is inconsistent")
        tier = certificate_tier
        expected_states += count_policy_states(certificate["policy_tree"])
        if campaign["certificate_sha256s"].get(mode) != certificate_sha:
            raise ProofError(f"{mode} certificate hash disagrees with campaign")
        census_path = Path(census_paths[mode])
        census = _load_json(census_path)
        if census.get("schema") != "wordle.census/v2" or set(census) != {
            "schema",
            "identity",
            "result",
        }:
            raise ProofError(f"{mode} census schema mismatch")
        identity = census["identity"]
        result = census["result"]
        objective = certificate["objective_value"]
        if (
            identity.get("mode") != mode
            or identity.get("tier") != tier
            or identity.get("artifact_sha256") != payload.weights_sha256
            or identity.get("manifest_sha256") != payload.manifest_sha256
            or identity.get("certificate_sha256") != certificate_sha
        ):
            raise ProofError(f"{mode} census identity mismatch")
        if {name: result.get(name) for name in objective} != objective:
            raise ProofError(f"{mode} census objective mismatch")
        if result.get("illegal_actions") != 0 or sum(result.get("histogram", [])) != tier:
            raise ProofError(f"{mode} census is incomplete or illegal")
        certificates[mode] = {
            "sha256": certificate_sha,
            "view_id": certificate["identity"]["view_id"],
            "answers_sha256": certificate["identity"]["answers_sha256"],
            "winner": certificate["first_action_proof"]["winner_word"],
        }
        censuses[mode] = {
            "sha256": _sha256(census_path),
            "trace_sha256": result["trace_digest"],
            "histogram": result["histogram"],
        }
        objectives[mode] = objective

    if tier is None or backend["states"] != expected_states:
        raise ProofError("adaptive campaign backend coverage is incomplete")

    firewall_path = Path(firewall_path)
    firewall = _load_json(firewall_path)
    if firewall.get("schema") != "wordle.firewall/v1" or firewall.get("status") != "PASS":
        raise ProofError("firewall evidence is not PASS")
    checks = firewall.get("checks")
    check_names = (
        [check.get("check") for check in checks]
        if isinstance(checks, list)
        else []
    )
    if (
        firewall.get("weights_sha256") != payload.weights_sha256
        or firewall.get("manifest_sha256") != payload.manifest_sha256
        or not isinstance(checks, list)
        or len(checks) != len(REQUIRED_FIREWALL_CHECKS)
        or set(check_names) != REQUIRED_FIREWALL_CHECKS
        or len(set(check_names)) != len(check_names)
        or any(check.get("status") != "PASS" for check in checks)
    ):
        raise ProofError("firewall evidence does not bind the release")

    return {
        "schema": PROOF_SCHEMA,
        "status": "VERIFIED",
        "label": f"Certified @{tier}",
        "claim": {
            "benchmark_id": payload.identity["benchmark_id"],
            "tier": tier,
            "track": payload.identity["track"],
            "modes": list(modes),
            "exact_tree_action_parity": True,
            "objectives": objectives,
        },
        "model": {
            "policy_spec": campaign["policy_spec"],
            "width": payload.width,
            "parameter_count": parameter_count(payload.width),
            "weights_bytes": (release_dir / "weights.safetensors").stat().st_size,
            "score_scale": campaign["score_scale"],
            "max_abs_logit_delta": max(deltas),
            "inference_class": payload.identity["inference_class"],
        },
        "release": {
            "weights_sha256": payload.weights_sha256,
            "manifest_sha256": payload.manifest_sha256,
        },
        "certificates": certificates,
        "censuses": censuses,
        "firewall": {
            "sha256": _sha256(firewall_path),
            "profile_sha256": firewall["sandbox_profile_sha256"],
            "checks_passed": 9,
        },
        "campaign": {
            "id": campaign["campaign_id"],
            "receipt_sha256": _sha256(campaign_receipt_path),
            "config_sha256": campaign["config_sha256"],
            "ah_cells_accessed": False,
        },
    }


def write_proof_receipt(path: Path, receipt: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
    return path


def _is_sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_public_proof(receipt, payload) -> bool:
    """Structural fail-closed validation of a generated v1 proof receipt."""
    from deep_green.model_spec import TENSOR_SPEC_VERSION, parameter_count

    try:
        modes = ("normal", "hard")
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema",
            "status",
            "label",
            "claim",
            "model",
            "release",
            "certificates",
            "censuses",
            "firewall",
            "campaign",
        }:
            return False
        if (
            receipt["schema"] != PROOF_SCHEMA
            or receipt["status"] != "VERIFIED"
            or not isinstance(receipt["label"], str)
        ):
            return False
        binding = receipt["release"]
        if not isinstance(binding, dict) or set(binding) != {
            "weights_sha256",
            "manifest_sha256",
        }:
            return False
        if (
            binding["weights_sha256"] != payload.weights_sha256
            or binding["manifest_sha256"] != payload.manifest_sha256
        ):
            return False
        claim = receipt["claim"]
        if not isinstance(claim, dict) or set(claim) != {
            "benchmark_id",
            "tier",
            "track",
            "modes",
            "exact_tree_action_parity",
            "objectives",
        }:
            return False
        if (
            claim["benchmark_id"] != payload.identity["benchmark_id"]
            or claim["track"] != payload.identity["track"]
            or not isinstance(claim["tier"], int)
            or isinstance(claim["tier"], bool)
            or claim["tier"] <= 0
            or claim["modes"] != ["normal", "hard"]
            or claim["exact_tree_action_parity"] is not True
            or receipt["label"] != f"Certified @{claim['tier']}"
        ):
            return False
        objectives = claim["objectives"]
        censuses = receipt["censuses"]
        if not isinstance(objectives, dict) or set(objectives) != set(modes):
            return False
        if not isinstance(censuses, dict) or set(censuses) != set(modes):
            return False
        for mode in modes:
            objective = objectives[mode]
            if not isinstance(objective, dict) or set(objective) != {
                "losses",
                "score7_total",
                "worst_score",
            }:
                return False
            if not all(_nonneg_int(value) for value in objective.values()):
                return False
            census = censuses[mode]
            if not isinstance(census, dict) or set(census) != {
                "sha256",
                "trace_sha256",
                "histogram",
            }:
                return False
            if not _is_sha256_hex(census["sha256"]) or not _is_sha256_hex(
                census["trace_sha256"]
            ):
                return False
            histogram = census["histogram"]
            if (
                not isinstance(histogram, list)
                or len(histogram) != 7
                or not all(_nonneg_int(value) for value in histogram)
                or sum(histogram) != claim["tier"]
            ):
                return False
            if objective["score7_total"] != sum(
                (index + 1) * value for index, value in enumerate(histogram)
            ):
                return False
            if objective["losses"] != histogram[6]:
                return False
            worst = max(
                index + 1
                for index, value in enumerate(histogram)
                if value > 0
            )
            if objective["worst_score"] != worst:
                return False
        model = receipt["model"]
        if not isinstance(model, dict) or set(model) != {
            "policy_spec",
            "width",
            "parameter_count",
            "weights_bytes",
            "score_scale",
            "max_abs_logit_delta",
            "inference_class",
        }:
            return False
        if (
            model["policy_spec"] != TENSOR_SPEC_VERSION
            or model["width"] != payload.width
            or model["parameter_count"] != parameter_count(payload.width)
            or model["inference_class"] != payload.identity["inference_class"]
        ):
            return False
        weights_file = Path(payload.release_dir) / "weights.safetensors"
        try:
            weights_bytes = weights_file.stat().st_size
        except OSError:
            return False
        if model["weights_bytes"] != weights_bytes:
            return False
        score_scale = model["score_scale"]
        if (
            isinstance(score_scale, bool)
            or not isinstance(score_scale, (int, float))
            or not 0 < score_scale <= 1
        ):
            return False
        delta = model["max_abs_logit_delta"]
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
            or not 0 <= delta <= 1e-5
        ):
            return False
        certificates = receipt["certificates"]
        if not isinstance(certificates, dict) or set(certificates) != set(modes):
            return False
        for mode in modes:
            record = certificates[mode]
            if not isinstance(record, dict) or set(record) != {
                "sha256",
                "view_id",
                "answers_sha256",
                "winner",
            }:
                return False
            if not _is_sha256_hex(record["sha256"]) or not _is_sha256_hex(
                record["answers_sha256"]
            ):
                return False
            if not isinstance(record["view_id"], str) or not record["view_id"]:
                return False
            winner = record["winner"]
            if (
                not isinstance(winner, str)
                or len(winner) != 5
                or not winner.islower()
                or not winner.isalpha()
            ):
                return False
        if len({certificates[mode]["answers_sha256"] for mode in modes}) != 1:
            return False
        firewall = receipt["firewall"]
        if not isinstance(firewall, dict) or set(firewall) != {
            "sha256",
            "profile_sha256",
            "checks_passed",
        }:
            return False
        if (
            not _is_sha256_hex(firewall["sha256"])
            or not _is_sha256_hex(firewall["profile_sha256"])
            or firewall["checks_passed"] != 9
        ):
            return False
        campaign = receipt["campaign"]
        if not isinstance(campaign, dict) or set(campaign) != {
            "id",
            "receipt_sha256",
            "config_sha256",
            "ah_cells_accessed",
        }:
            return False
        return (
            isinstance(campaign["id"], str)
            and len(campaign["id"]) == 16
            and all(char in "0123456789abcdef" for char in campaign["id"])
            and _is_sha256_hex(campaign["receipt_sha256"])
            and _is_sha256_hex(campaign["config_sha256"])
            and campaign["ah_cells_accessed"] is False
        )
    except (KeyError, TypeError, AttributeError):
        return False


def load_public_proof(path: Path, release_dir: Path) -> dict:
    try:
        receipt = _load_json(Path(path))
        payload = load_release(Path(release_dir))
        if not validate_public_proof(receipt, payload):
            return dict(UNVERIFIED_PROOF)
        return receipt
    except (ProofError, OSError, ValueError):
        return dict(UNVERIFIED_PROOF)
