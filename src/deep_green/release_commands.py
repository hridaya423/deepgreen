"""CLI wiring for the release accounting/verification/clean/seal pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FROZEN_ROOT = "benchmarks/openwordle-v1/frozen-universe"
DEFAULT_CAMPAIGN = "runs/2ddee60886659422"
DEFAULT_ORACLE = "artifacts/oracle/openwordle-v1"
DEFAULT_NORMAL_CERT = (
    f"{DEFAULT_ORACLE}/openwordle-v1.uniform.normal.full-compiled@5022/"
    "certificate.json"
)
DEFAULT_HARD_CERT = (
    f"{DEFAULT_ORACLE}/openwordle-v1.uniform.hard.full-compiled@5022/"
    "certificate.json"
)


def _run_build(args) -> int:
    from deep_green.release_pipeline import (
        ReleasePipelineError,
        build_candidate,
    )

    try:
        result = build_candidate(
            output=Path(args.output),
            payload_dir=Path(args.payload_dir),
            proof_path=Path(args.proof),
            campaign_receipt=Path(args.campaign_receipt),
            normal_census=Path(args.normal_census),
            hard_census=Path(args.hard_census),
            firewall_evidence=Path(args.firewall),
            web_contract=Path(args.web_contract),
            web_dist=Path(args.web_dist),
            desktop_screenshot=Path(args.desktop_screenshot),
            mobile_screenshot=Path(args.mobile_screenshot),
            repo_root=Path(args.repo_root),
        )
    except ReleasePipelineError as exc:
        print(f"CANDIDATE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"CANDIDATE BUILT: {args.output}")
    print(f"  manifest_sha256={result['manifest_sha256']}")
    return 0


def _run_verify(args) -> int:
    from deep_green.release_pipeline import (
        _canonical_bytes,
        _write_atomic,
        verify_candidate,
    )

    receipt = verify_candidate(
        candidate=Path(args.candidate),
        frozen_root=Path(args.frozen_root),
        normal_certificate=Path(args.normal_certificate),
        hard_certificate=Path(args.hard_certificate),
        clean_receipt=Path(args.clean_receipt) if args.clean_receipt else None,
        run_live_firewall=not args.no_live_firewall,
    )
    for gate in receipt["gates"]:
        print(f"  {gate['status']} {gate['name']}: {gate['detail']}")
    print(f"VERIFICATION: {receipt['status']}")
    if args.output:
        _write_atomic(Path(args.output), _canonical_bytes(receipt))
        print(f"  receipt={args.output}")
    return 0 if receipt["status"] == "PASS" else 1


def _run_clean_reproduce(args) -> int:
    from deep_green.release_pipeline import run_clean_reproduction

    receipt = run_clean_reproduction(
        repo_root=Path(args.repo_root),
        candidate=Path(args.candidate),
        output=Path(args.output),
        frozen_root=Path(args.frozen_root),
        normal_certificate=Path(args.normal_certificate),
        hard_certificate=Path(args.hard_certificate),
    )
    for check in receipt["checks"]:
        print(f"  {check['status']} {check['name']}")
    print(f"CLEAN REPRODUCTION: {receipt['status']}")
    print(f"  receipt={args.output}")
    log_path = Path(str(args.output) + ".log")
    if log_path.is_file():
        failed = [c["name"] for c in receipt["checks"] if c["status"] == "FAIL"]
        for name in failed:
            print(f"  failed step: {name}")
        print(f"  log={log_path}")
    return 0 if receipt["status"] == "PASS" else 1


def _run_seal(args) -> int:
    from deep_green.release_pipeline import (
        ReleasePipelineError,
        seal_candidate,
    )

    try:
        seal = seal_candidate(
            candidate=Path(args.candidate),
            verification_receipt=Path(args.verification),
            clean_receipt=Path(args.clean),
            output_dir=Path(args.output),
        )
    except ReleasePipelineError as exc:
        print(f"SEAL FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"SEALED: {args.output}")
    print(f"  archive_sha256={seal['archive_sha256']}")
    return 0


def _add_candidate_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--frozen-root", default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--normal-certificate", default=DEFAULT_NORMAL_CERT)
    parser.add_argument("--hard-certificate", default=DEFAULT_HARD_CERT)


def register_release_subcommands(group_parser: argparse.ArgumentParser) -> None:
    subparsers = group_parser.add_subparsers(
        dest="release_command", metavar="release_command", required=True
    )

    build = subparsers.add_parser(
        "build", help="assemble the allowlist-only release candidate"
    )
    build.add_argument("--output", required=True)
    build.add_argument(
        "--payload-dir", default="artifacts/releases/champion-2095-certified"
    )
    build.add_argument(
        "--proof", default="artifacts/proof/champion-2095-certified/proof.json"
    )
    build.add_argument(
        "--campaign-receipt", default=f"{DEFAULT_CAMPAIGN}/receipt.json"
    )
    build.add_argument(
        "--normal-census",
        default=f"{DEFAULT_CAMPAIGN}/evidence/width-320/census-normal.json",
    )
    build.add_argument(
        "--hard-census",
        default=f"{DEFAULT_CAMPAIGN}/evidence/width-320/census-hard.json",
    )
    build.add_argument(
        "--firewall", default=f"{DEFAULT_CAMPAIGN}/evidence/firewall.json"
    )
    build.add_argument("--web-contract", default="reports/web/contract.json")
    build.add_argument("--web-dist", default="web/dist")
    build.add_argument(
        "--desktop-screenshot",
        default="reports/web/screens/desktop-assist.png",
    )
    build.add_argument(
        "--mobile-screenshot", default="reports/web/screens/mobile-solve.png"
    )
    build.add_argument("--repo-root", default=".")
    build.set_defaults(handler=_run_build)

    verify = subparsers.add_parser(
        "verify", help="run the nine named release gates"
    )
    _add_candidate_common(verify)
    verify.add_argument("--clean-receipt", default=None)
    verify.add_argument("--no-live-firewall", action="store_true")
    verify.add_argument("--output", default=None)
    verify.set_defaults(handler=_run_verify)

    clean = subparsers.add_parser(
        "clean-reproduce", help="verify the candidate in a clean worktree"
    )
    _add_candidate_common(clean)
    clean.add_argument("--output", required=True)
    clean.add_argument("--repo-root", default=".")
    clean.set_defaults(handler=_run_clean_reproduce)

    seal = subparsers.add_parser(
        "seal", help="emit the deterministic sealed release archive"
    )
    seal.add_argument("--candidate", required=True)
    seal.add_argument("--verification", required=True)
    seal.add_argument("--clean", required=True)
    seal.add_argument("--output", required=True)
    seal.set_defaults(handler=_run_seal)
