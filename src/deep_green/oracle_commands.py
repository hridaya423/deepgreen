"""Wire the vendored exact-search oracle behind typed commands."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_MANIFEST = REPO_ROOT / "benchmarks" / "openwordle-v1" / "frozen" / "manifest.json"
FROZEN_ROOT = FROZEN_MANIFEST.parent


def _load_views(frozen_root: Path = FROZEN_ROOT) -> list[dict]:
    import json

    views = json.loads((frozen_root / "views" / "views.json").read_text(encoding="utf-8"))
    items = views["views"] if isinstance(views, dict) and "views" in views else views
    return list(items)


def _resolve_view(view_id: str, frozen_root: Path = FROZEN_ROOT) -> dict:
    matching = [v for v in _load_views(frozen_root) if v["view_id"] == view_id]
    if len(matching) == 1:
        return matching[0]
    raise SystemExit(
        f"error: view {view_id!r} not uniquely present in the frozen bundle "
        f"(candidates: {sorted({v['view_id'] + '@' + str(v['tier']) for v in _load_views(frozen_root)})})"
    )


def _resolve_cell(cell: str, frozen_root: Path = FROZEN_ROOT) -> dict:
    """Resolve 'view_id@tier' or a bare unique view id to one concrete view."""
    if "@" in cell:
        view_id, tier_text = cell.rsplit("@", 1)
        candidates = [
            v
            for v in _load_views(frozen_root)
            if v["view_id"] == view_id and v["tier"] == int(tier_text)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return _resolve_view(cell, frozen_root)


def _cert_output_dir(view: dict) -> Path:
    base = REPO_ROOT / "artifacts" / "oracle" / "openwordle-v1"
    return base / f"{view['view_id']}@{view['tier']}"


def _archive_existing_certificate(cert_path: Path) -> Path | None:
    if not cert_path.is_file():
        return None
    payload = cert_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    archive = cert_path.with_name(f"certificate.legacy-{digest}.json")
    try:
        with archive.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if archive.read_bytes() != payload:
            raise RuntimeError(f"legacy certificate archive collision: {archive}")
    return archive


def _run_smoke(args) -> int:
    from deep_green.oracle.smoke import format_report, run_smoke

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else REPO_ROOT / "artifacts" / "oracle" / "smoke"
    )
    report = run_smoke(repo_root=REPO_ROOT, work_dir=work_dir, mode=args.mode)
    print(format_report(report))
    return 0


def _run_certify(args) -> int:
    import json

    from deep_green.oracle.certifier import build_certificate, fill_objective_value
    from deep_green.oracle.tree_stats import collect_tree_stats
    from deep_green.oracle.verifier import verify_certificate

    frozen_root = Path(args.frozen_root) if args.frozen_root else FROZEN_ROOT
    view = _resolve_cell(args.view, frozen_root)
    out_dir = _cert_output_dir(view)
    work_dir = out_dir / ".work"
    certificate = build_certificate(
        view=view,
        frozen_corpus_dir=frozen_root / "corpus",
        benchmark_id="openwordle-v1",
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        jobs=args.jobs,
        beta=args.beta,
        cache_gb=args.cache_gb,
    )
    tree_stats = collect_tree_stats(
        certificate["policy_tree"],
        frozen_corpus_dir=frozen_root / "corpus",
        answers_file=view["answers_file"],
        horizon_turns=view["horizon_turns"],
    )
    fill_objective_value(certificate, tree_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / "certificate.json"
    payload = json.dumps(certificate, indent=1, sort_keys=True) + "\n"
    tmp_path = out_dir / "certificate.json.tmp"
    tmp_path.write_text(payload, encoding="utf-8")
    replay = verify_certificate(tmp_path, frozen_root=frozen_root)
    _archive_existing_certificate(cert_path)
    tmp_path.replace(cert_path)
    print(f"CERTIFIED: {args.view}")
    print(f"  objective={certificate['objective_value']}")
    print(f"  best_first={certificate['search']['best_first_word']}")
    print(f"  replay: nodes={replay.nodes} answers={replay.answers_reached}")
    return 0


def _run_verify(args) -> int:
    from deep_green.oracle.verifier import verify_certificate

    cert_path = Path(args.certificate)
    frozen_root = Path(args.frozen_root) if args.frozen_root else FROZEN_ROOT
    stats = verify_certificate(cert_path, frozen_root=frozen_root)
    print(f"VERIFY OK: {cert_path}")
    print(f"  nodes={stats.nodes} answers={stats.answers_reached} losses={stats.losses}")
    return 0


def _run_campaign_status(args) -> int:
    import json
    import os

    campaigns_dir = REPO_ROOT / "artifacts" / "oracle" / "campaigns"
    classic_dir = REPO_ROOT / "artifacts" / "oracle" / "classic-replay"
    rows = []
    for base in (campaigns_dir, classic_dir):
        if not base.is_dir():
            continue
        for exit_file in sorted(base.glob("*.exit")):
            stem = exit_file.stem
            events_file = base / f"{stem}.jsonl"
            objective = None
            status = "complete"
            if events_file.is_file():
                for line in events_file.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "result":
                        objective = event.get("objective")
                        break
            else:
                status = "no-events"
            if int(exit_file.read_text().strip()) != 0:
                status = f"solver-exit-{exit_file.read_text().strip()}"
            rows.append((stem, status, objective))
    for stem, status, objective in sorted(rows):
        print(f"{stem}: {status} objective={objective}")
    if not rows:
        print("no campaign results recorded yet")
        return 1
    del os
    return 0


def register_oracle_subcommands(group_parser: argparse.ArgumentParser) -> None:
    oracle_subparsers = group_parser.add_subparsers(
        dest="oracle_command", metavar="oracle_command", required=True
    )
    smoke_parser = oracle_subparsers.add_parser(
        "smoke", help="replay a tiny exhaustive cell and cross-check against python"
    )
    smoke_parser.add_argument("--mode", choices=("normal", "hard"), required=True)
    smoke_parser.add_argument("--work-dir", default=None)
    smoke_parser.set_defaults(handler=_run_smoke)

    certify_parser = oracle_subparsers.add_parser(
        "certify", help="produce an exact policy-tree certificate for a frozen view"
    )
    certify_parser.add_argument("--view", required=True)
    certify_parser.add_argument("--frozen-root", default=None)
    certify_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel first-action sweep shards (merged into one event stream)",
    )
    certify_parser.add_argument(
        "--beta",
        type=int,
        default=None,
        help="initial beta cutoff for the first-action sweep",
    )
    certify_parser.add_argument(
        "--cache-gb",
        type=float,
        default=None,
        help="per-worker solver cache memory limit in GB",
    )
    certify_parser.set_defaults(handler=_run_certify)

    verify_parser = oracle_subparsers.add_parser(
        "verify",
        help="independently replay an oracle certificate against the frozen bundle",
    )
    verify_parser.add_argument("--certificate", required=True)
    verify_parser.add_argument("--frozen-root", default=None)
    verify_parser.set_defaults(handler=_run_verify)

    status_parser = oracle_subparsers.add_parser(
        "campaign-status",
        help="summarize background tier-campaign and historical-replay results",
    )
    status_parser.set_defaults(handler=_run_campaign_status)

    firewall_parser = oracle_subparsers.add_parser(
        "firewall-audit",
        help="run the noninterference battery and emit firewall evidence",
    )
    firewall_parser.add_argument("--release", required=True)
    firewall_parser.add_argument("--output", default=None)
    firewall_parser.set_defaults(handler=_run_firewall_audit)


def _run_firewall_audit(args) -> int:
    from deep_green.firewall_audit import audit_release

    evidence = audit_release(Path(args.release), output_path=Path(args.output) if args.output else None)
    print(f"FIREWALL AUDIT: {evidence['status']}")
    for finding in evidence["checks"]:
        print(f"  {finding['status']} {finding['check']}")
    return 0 if evidence["status"] == "PASS" else 1


def _run_rebuild_index(args) -> int:
    import json
    import sqlite3

    from deep_green.tracking import EventLog

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    db_path = runs_root / "index.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints ("
        "run_id TEXT, path TEXT, update_index INT)"
    )
    connection.execute("DELETE FROM checkpoints")
    scanned = 0
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or (run_dir / "manifest.json").is_file() is False:
            continue
        run_id = run_dir.name
        events_path = run_dir / "events.jsonl"
        if events_path.is_file():
            for event in EventLog(events_path).read_recovered():
                if event.get("type") == "checkpoint":
                    connection.execute(
                        "INSERT INTO checkpoints VALUES (?, ?, ?)",
                        (run_id, event.get("path"), int(event.get("update", 0))),
                    )
        inventory_path = run_dir / "run-inventory.json"
        final = None
        if inventory_path.is_file():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            final = inventory.get("final_checkpoint")
        if final:
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                (run_id, final, 1 << 30),
            )
        scanned += 1
    connection.commit()
    connection.close()
    print(f"REBUILT: {db_path} from {scanned} runs")
    return 0
