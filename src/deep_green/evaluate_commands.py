"""CLI wiring for firewall audit and census evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FROZEN_ROOT = "benchmarks/openwordle-v1/frozen-universe"


def _run_firewall_audit(args) -> int:
    from deep_green.firewall_audit import audit_release

    release_dir = Path(args.release)
    output_path = Path(args.output) if args.output else None
    evidence = audit_release(release_dir, output_path=output_path)
    status = evidence["status"]
    print(f"FIREWALL AUDIT: {status}")
    failed = [c for c in evidence["checks"] if c["status"] != "PASS"]
    if failed:
        for finding in failed:
            print(f"  FAIL {finding['check']}: {finding['detail']}", file=sys.stderr)
        return 1
    for finding in evidence["checks"]:
        print(f"  PASS {finding['check']}")
    return 0


def _run_evaluate(args) -> int:
    from deep_green.evaluate import (
        EvaluationError,
        evaluate_certified,
        save_census_report,
    )

    try:
        result, identity = evaluate_certified(
            release_dir=Path(args.policy),
            certificate_path=Path(args.certificate),
            frozen_root=Path(args.frozen_root),
        )
    except EvaluationError as exc:
        print(f"CENSUS FAILED: {exc}", file=sys.stderr)
        return 1
    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT
        / "artifacts/releases"
        / str(identity["view_id"])
        / "evidence"
        / f"census-{identity['mode']}.json"
    )
    save_census_report(output, result, identity)
    print(
        f"CENSUS: answers={result.answers} losses={result.losses} "
        f"score7_total={result.score7_total} worst={result.worst_score}"
    )
    print(f"  report={output}")
    return 0


def register_evaluate_subcommands(group_parser: argparse.ArgumentParser) -> None:
    subparsers = group_parser.add_subparsers(
        dest="evaluate_command", metavar="evaluate_command", required=True
    )
    run_parser = subparsers.add_parser(
        "run", help="run the exhaustive certified census for a release"
    )
    run_parser.add_argument("--certificate", required=True)
    run_parser.add_argument("--policy", required=True)
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--frozen-root", default=DEFAULT_FROZEN_ROOT)
    run_parser.set_defaults(handler=_run_evaluate)


def _run_play_serve(args) -> int:
    import socket

    import uvicorn

    from deep_green.webapp import create_app

    frozen_root = Path(args.frozen_root)
    tier_default = frozen_root / "corpus/tiers/tier-5022.txt"
    if not tier_default.is_file():
        tier_default = frozen_root / "corpus/tiers/tier-2095.txt"
    tier_answers_path = (
        Path(args.tier_answers) if args.tier_answers else tier_default
    )
    app = create_app(
        release_dir=Path(args.release),
        proof_path=Path(args.proof),
        tier_answers_path=tier_answers_path,
        static_dir=Path(args.static_dir),
    )
    if args.port is not None:
        port = args.port
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((args.host, port))
            probe.listen(1)
        except OSError:
            print(f"PORT UNAVAILABLE: {args.host}:{port}", file=sys.stderr)
            return 1
        finally:
            probe.close()
    else:
        port = None
        for candidate in range(3101, 3200):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind((args.host, candidate))
                probe.listen(1)
            except OSError:
                continue
            finally:
                probe.close()
            port = candidate
            break
        if port is None:
            print("PORT UNAVAILABLE: no bindable port in 3101..3199", file=sys.stderr)
            return 1
    print(f"Serving on http://{args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port)
    return 0


def register_play_subcommands(group_parser: argparse.ArgumentParser) -> None:
    play_parser = group_parser.add_subparsers(
        dest="play_command", metavar="play_command", required=True
    )
    serve_parser = play_parser.add_parser(
        "serve", help="serve the local tournament web app"
    )
    serve_parser.add_argument(
        "--release", default="artifacts/releases/champion-5022-certified"
    )
    serve_parser.add_argument(
        "--proof",
        default="artifacts/proof/champion-5022-certified/proof.json",
    )
    serve_parser.add_argument("--frozen-root", default=DEFAULT_FROZEN_ROOT)
    serve_parser.add_argument(
        "--tier-answers",
        default=None,
        help="certified answer tier file used to label Solve games",
    )
    serve_parser.add_argument("--static-dir", default="web/dist")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.set_defaults(handler=_run_play_serve)


def register_train_subcommands(group_parser: argparse.ArgumentParser) -> None:
    train_parser = group_parser.add_subparsers(
        dest="train_command", metavar="train_command", required=True
    )
    examples_parser = train_parser.add_parser(
        "examples", help="generate root examples from verified oracle certificates"
    )
    examples_parser.add_argument("--certificate", action="append", required=True)
    examples_parser.add_argument("--output", required=True)
    examples_parser.add_argument(
        "--frozen-root", default="benchmarks/openwordle-v1/frozen-universe"
    )
    examples_parser.add_argument("--repo-root", default=".")
    examples_parser.set_defaults(handler=_run_train_examples)

    clone_parser = train_parser.add_parser(
        "clone", help="train an exact-teacher clone policy from oracle examples"
    )
    clone_parser.add_argument("--config", required=True)
    clone_parser.add_argument("--examples", required=True)
    clone_parser.add_argument("--model-width", type=int, default=64)
    clone_parser.add_argument("--device", default="cpu")
    clone_parser.add_argument("--max-updates", type=int, default=8)
    clone_parser.add_argument("--runs-root", default="runs")
    clone_parser.add_argument("--action-words", default=None)
    clone_parser.add_argument("--resume", default=None)
    clone_parser.set_defaults(handler=_run_train_clone)

    adaptive_parser = train_parser.add_parser(
        "adaptive", help="run the certified adaptive ladder campaign"
    )
    adaptive_parser.add_argument("--config", required=True)
    adaptive_parser.add_argument("--runs-root", default="runs")
    adaptive_parser.add_argument("--output-release", required=True)
    adaptive_parser.set_defaults(handler=_run_train_adaptive)


def _run_train_examples(args) -> int:
    from deep_green.examples import generate_root_examples, write_examples

    records = generate_root_examples(
        [Path(path) for path in args.certificate],
        frozen_root=Path(args.frozen_root),
        repo_root=Path(args.repo_root),
    )
    output = write_examples(Path(args.output), records)
    print(f"WROTE {len(records)} verified root examples to {output}")
    return 0


def _run_train_clone(args) -> int:
    from deep_green.train import run_clone

    result = run_clone(Path(args.config), args)
    inventory = result["inventory"]
    print(f"CLONE RUN {result['run_id']} closed at {result['run_dir']}")
    print(f"  updates={inventory['updates']} width={inventory['width']}")
    print(f"  final_argmax_optimal={inventory['final_argmax_optimal']}")
    return 0


def _run_train_adaptive(args) -> int:
    from deep_green.adaptive import run_adaptive

    receipt = run_adaptive(
        Path(args.config),
        runs_root=Path(args.runs_root),
        output_release=Path(args.output_release),
    )
    if receipt["status"] == "PARITY":
        print(f"PARITY width={receipt['selected_width']}")
        return 0
    print("INCOMPLETE")
    return 1
