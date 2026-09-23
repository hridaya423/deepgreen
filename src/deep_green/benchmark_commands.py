"""CLI-facing benchmark commands: preflight, freeze, and verify."""

from __future__ import annotations

import sys
from pathlib import Path


def run_benchmark_preflight(args) -> int:
    from deep_green.benchmark import PreflightError, run_preflight

    try:
        report = run_preflight(Path(args.candidate))
    except PreflightError as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    counts = " ".join(
        f"{Path(rel).stem}={count}" for rel, count in sorted(report.recomputed_counts.items())
    )
    print(f"PREFLIGHT OK: {counts}")
    return 0


def run_benchmark_freeze(args) -> int:
    from deep_green.benchmark import (
        BundleError,
        PreflightError,
        ProtocolError,
        load_protocol,
        publish_freeze,
    )

    try:
        protocol = load_protocol(args.protocol)
    except ProtocolError as error:
        print(f"FREEZE FAILED: {error}", file=sys.stderr)
        return 1
    try:
        publish_freeze(protocol, Path(args.candidate), Path(args.output))
    except (BundleError, PreflightError, OSError) as error:
        print(f"FREEZE FAILED: {error}", file=sys.stderr)
        return 1
    print(f"FREEZE OK: {args.output}")
    return 0


def run_benchmark_verify(args) -> int:
    from deep_green.benchmark import BundleError, verify_frozen_bundle

    try:
        report = verify_frozen_bundle(Path(args.frozen))
    except BundleError as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        return 1
    print(f"VERIFY OK: views={report.view_count}")
    return 0
