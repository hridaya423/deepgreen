"""Build exact policy-tree certificates by driving the vendored solver.

The certificate records, per top-level action: its exact total score (or an
independent lower bound), whether it is optimal, and for the winning action a
full strategy tree replayable by the independent Python verifier.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from deep_green.oracle.adapter import (
    OracleActionValue,
    OracleEventsError,
    OracleRunRecord,
    _parse_action_value,
    _parse_event_line,
    parse_events,
)
from deep_green.oracle.build import ensure_solver, vendor_dir
from deep_green.oracle.upstream import EVENTS_SCHEMA
from deep_green.rules import ActionSpace

INFINITY = 10**9


class CertificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionOutcome:
    action_index: int  # canonical alphabetical ActionSpace index
    word: str
    bound: int
    exact: bool


@dataclass(frozen=True)
class SearchRecord:
    mode: str
    horizon: int
    exhaustive: bool
    best_first_word: str
    objective: int
    outcomes: tuple[ActionOutcome, ...]
    tree_text: str
    solver_events: dict
    executable_sha256: str
    compiler_id: str


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_canonical_winner(
    outcomes: tuple[ActionOutcome, ...] | list[ActionOutcome], objective: int
) -> ActionOutcome:
    """Lowest canonical action index among actions exactly equal to the objective."""
    optimal = [
        outcome
        for outcome in outcomes
        if outcome.exact and outcome.bound == objective
    ]
    if not optimal:
        raise CertificationError(
            f"no exact action record equals the solver objective {objective}"
        )
    return min(optimal, key=lambda outcome: outcome.action_index)


def _validate_reported_winner(
    best_reported: object,
    solver_order: list[str],
    outcomes: list[ActionOutcome],
    objective: int,
) -> None:
    if (
        not isinstance(best_reported, int)
        or isinstance(best_reported, bool)
        or not 0 <= best_reported < len(solver_order)
    ):
        raise CertificationError(f"solver best_first is malformed: {best_reported!r}")
    reported = solver_order[best_reported]
    reported_outcome = next(
        (outcome for outcome in outcomes if outcome.word == reported), None
    )
    if (
        reported_outcome is None
        or not reported_outcome.exact
        or reported_outcome.bound != objective
    ):
        raise CertificationError(
            f"solver best_first {reported!r} is not an exact record equal to "
            f"the objective {objective}"
        )


def _check_solver_identity(
    identity: dict,
    *,
    mode: str,
    horizon: int,
    action_count: int,
    answer_count: int,
) -> None:
    expected_identity = {
        "hardmode": 1 if mode == "hard" else 0,
        "maxguesses": horizon,
        "all_count": action_count,
        "hidden_count": answer_count,
        "depthonly": 0,
    }
    for field, expected in expected_identity.items():
        actual = identity.get(field)
        if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
            raise CertificationError(
                f"solver identity {field} {actual!r} does not match requested {expected!r}"
            )
    if identity.get("exhaustive") is not True:
        raise CertificationError("search was not exhaustive; cannot certify")


def _validated_objective(result: dict) -> int:
    objective = result.get("objective")
    if not isinstance(objective, int) or isinstance(objective, bool) or objective < 0:
        raise CertificationError(f"solver objective is malformed: {objective!r}")
    return objective


def _parse_shard_stream(
    text: str,
) -> tuple[dict | None, list[OracleActionValue], dict | None]:
    """Lenient per-shard parse for work-stealing event files.

    Workers killed mid-sweep leave truncated streams: their ``action_value``
    records are still valid bound proofs, so identity and action values are
    required while ``result``/``completion`` may be absent.
    """
    identity: dict | None = None
    values: list[OracleActionValue] = []
    result: dict | None = None
    seen_indices: set[int] = set()
    seen_words: set[str] = set()
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        event = _parse_event_line(raw_line, number)
        kind = event["type"]
        if kind == "run_identity":
            if identity is not None:
                raise CertificationError("duplicate run_identity in shard stream")
            if number != 1:
                raise CertificationError("run_identity is not the first shard event")
            all_count = event["all_count"]
            if not isinstance(all_count, int) or isinstance(all_count, bool) or all_count < 0:
                raise CertificationError("shard run_identity all_count malformed")
            identity = event
        elif kind == "action_value":
            if identity is None:
                raise CertificationError("shard action_value before run_identity")
            if result is not None:
                raise CertificationError("shard action_value after result")
            value = _parse_action_value(event, identity, number)
            if value.action_index in seen_indices or value.word in seen_words:
                raise CertificationError(
                    f"duplicate shard action {value.word!r}/{value.action_index}"
                )
            seen_indices.add(value.action_index)
            seen_words.add(value.word)
            values.append(value)
        elif kind == "result":
            if identity is None or result is not None:
                raise CertificationError("misplaced or duplicate shard result")
            result = event
    return identity, values, result


def _merge_shard_events(shard_texts: list[str], action_count: int) -> str:
    """Fold per-shard solver event streams into one canonical stream.

    Workers claim each action exactly once via the shared joblist, so the
    action_value sets are disjoint. Streams from interrupted workers lack
    result/completion records and are still merged from their action values.
    """
    parsed = [_parse_shard_stream(text) for text in shard_texts]
    identities = [identity for identity, _, _ in parsed if identity is not None]
    if not identities:
        raise CertificationError("no shard stream carried a run_identity")
    identity = identities[0]
    if any(other != identity for other in identities):
        raise CertificationError("shard run identities diverge")
    values: list[OracleActionValue] = []
    seen_indices: set[int] = set()
    seen_words: set[str] = set()
    for _, shard_values, _ in parsed:
        for value in shard_values:
            if value.action_index in seen_indices or value.word in seen_words:
                raise CertificationError(
                    f"action {value.word!r}/{value.action_index} covered by "
                    "multiple shards"
                )
            seen_indices.add(value.action_index)
            seen_words.add(value.word)
            values.append(value)
    results = [result for _, _, result in parsed if result is not None]
    exact_values = [v for v in values if v.exact]
    if not exact_values and not results:
        raise CertificationError("shard streams contain no objective evidence")
    objective = min(
        [v.bound for v in exact_values]
        + [result["objective"] for result in results]
    )
    winners = [v for v in exact_values if v.bound == objective]
    if not winners:
        raise CertificationError(
            f"no exact action record equals the merged objective {objective}"
        )
    winner = min(winners, key=lambda v: v.action_index)
    base = (
        dict(min(results, key=lambda r: r["objective"]))
        if results
        else {"type": "result", "schema": EVENTS_SCHEMA}
    )
    merged_result = base
    merged_result.update(
        {
            "type": "result",
            "schema": EVENTS_SCHEMA,
            "objective": objective,
            "average": objective / action_count,
            "best_first": winner.action_index,
            "nodes": sum(result["nodes"] for result in results),
            "time_seconds": max(
                (result["time_seconds"] for result in results), default=0
            ),
            "depthonly": False,
            "checkpoints": 0,
        }
    )
    completion = {
        "type": "completion",
        "schema": EVENTS_SCHEMA,
        "status": "completed",
        "covered_actions": action_count,
        "declared_actions": action_count,
    }
    lines = [json.dumps(identity, sort_keys=True, separators=(",", ":"))]
    lines += [
        json.dumps(
            {
                "type": "action_value",
                "schema": EVENTS_SCHEMA,
                "action_index": value.action_index,
                "word": value.word,
                "bound": value.bound,
                "exact": value.exact,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for value in values
    ]
    lines.append(json.dumps(merged_result, sort_keys=True, separators=(",", ":")))
    lines.append(json.dumps(completion, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def run_exact_search(
    *,
    repo_root: Path,
    work_dir: Path,
    action_words: tuple[str, ...],
    answer_words: tuple[str, ...],
    mode: str,
    horizon: int,
    with_tree: bool,
    jobs: int = 1,
    beta: int | None = None,
    cache_gb: float | None = None,
) -> SearchRecord:
    """Exhaustive first-action proof sweep, plus a second forced-winner solver
    invocation when a policy tree is requested."""
    if mode not in ("normal", "hard"):
        raise CertificationError(f"unsupported mode {mode!r}")
    if jobs < 1:
        raise CertificationError(f"jobs must be >= 1, got {jobs}")
    exe_path, executable_sha256, compiler_id = ensure_solver(
        vendor_dir(repo_root), repo_root / "artifacts" / "oracle" / "build-cache"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    all_txt = work_dir / "all.txt"
    hidden_txt = work_dir / "hidden.txt"
    events_jsonl = work_dir / "events.jsonl"
    all_txt.write_text("\n".join(action_words) + "\n", encoding="utf-8")
    hidden_txt.write_text("\n".join(answer_words) + "\n", encoding="utf-8")
    def sweep(*, showtop: bool):
        def command_for(events_path: Path) -> list[str]:
            command = [
                str(exe_path),
                "-a",
                str(all_txt),
                "-h",
                str(hidden_txt),
                "-g",
                str(horizon),
                "-E",
                str(events_path),
            ]
            if showtop:
                command.append("-s")
            if beta is not None:
                command += ["-b", str(beta)]
            if cache_gb is not None:
                command += ["-c", str(cache_gb)]
            if mode == "hard":
                command.append("-H")
            return command

        if jobs <= 1:
            completed = subprocess.run(
                command_for(events_jsonl), capture_output=True, text=True, check=False
            )
            if completed.returncode != 0:
                raise CertificationError(
                    f"solver exited {completed.returncode}: "
                    f"{(completed.stderr or '').strip()[:2000]}"
                )
            if not events_jsonl.is_file():
                raise CertificationError(
                    "solver produced no events file despite exiting zero"
                )
            return parse_events(events_jsonl.read_text(encoding="utf-8"))
        # Work-stealing shard: every worker pulls its next first action from a
        # shared flock-coordinated joblist. Weakest openers go first in the file
        # (pulled last); expensive near-optimal candidates sit at the tail and
        # are pulled first, which keeps the makespan tail short.
        letter_freq = Counter(
            letter for word in action_words for letter in set(word)
        )
        joblist = work_dir / "joblist.txt"
        joblist.write_text(
            "\n".join(
                sorted(
                    action_words,
                    key=lambda word: sum(letter_freq[c] for c in set(word)),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        shard_paths = [work_dir / f"events.shard{i}.jsonl" for i in range(jobs)]
        procs = [
            subprocess.Popen(
                command_for(shard_path) + ["-j", str(joblist)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            for shard, shard_path in enumerate(shard_paths)
        ]
        shard_texts = []
        for shard, (proc, shard_path) in enumerate(zip(procs, shard_paths)):
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise CertificationError(
                    f"solver shard {shard} exited {proc.returncode}: "
                    f"{(stderr or '').strip()[:2000]}"
                )
            if not shard_path.is_file():
                raise CertificationError(
                    f"solver shard {shard} produced no events file"
                )
            shard_texts.append(shard_path.read_text(encoding="utf-8"))
        merged = _merge_shard_events(shard_texts, len(action_words))
        events_jsonl.write_text(merged, encoding="utf-8")
        return parse_events(merged)

    record_events: OracleRunRecord | None = None
    if events_jsonl.is_file():
        # Resume path: a complete merged event stream (e.g. assembled from a
        # detached work-stealing fleet) lets the sweep be skipped entirely.
        try:
            record_events = parse_events(events_jsonl.read_text(encoding="utf-8"))
        except OracleEventsError:
            record_events = None
        if (
            record_events is not None
            and record_events.completion.get("covered_actions") != len(action_words)
        ):
            record_events = None
    if record_events is None:
        record_events = sweep(showtop=False)
    if record_events.completion.get("covered_actions") != len(action_words):
        events_jsonl.unlink(missing_ok=True)
        record_events = sweep(showtop=True)
    identity = record_events.identity
    result = record_events.result
    completion = record_events.completion
    if identity["schema"] != EVENTS_SCHEMA or completion["status"] != "completed":
        raise CertificationError("solver did not complete cleanly")
    _check_solver_identity(
        identity,
        mode=mode,
        horizon=horizon,
        action_count=len(action_words),
        answer_count=len(answer_words),
    )
    if not (
        completion["covered_actions"]
        == completion["declared_actions"]
        == len(action_words)
    ):
        raise CertificationError(
            "first-action proof coverage incomplete: "
            f"covered_actions={completion['covered_actions']} "
            f"declared_actions={completion['declared_actions']} "
            f"action_words={len(action_words)}"
        )
    solver_order = sorted(set(answer_words)) + sorted(set(action_words) - set(answer_words))
    action_space = ActionSpace.parse(action_words)
    canonical_index = {word: index for index, word in enumerate(action_space.words)}
    outcomes: list[ActionOutcome] = []
    for value in record_events.action_values:
        if value.action_index >= len(solver_order):
            raise CertificationError(
                f"action_index {value.action_index} outside the reconstructed "
                f"solver order of {len(solver_order)} words"
            )
        if solver_order[value.action_index] != value.word:
            raise CertificationError(
                f"action_index {value.action_index} maps to solver word "
                f"{solver_order[value.action_index]!r}, not emitted word {value.word!r}"
            )
        canonical = canonical_index.get(value.word)
        if canonical is None:
            raise CertificationError(
                f"emitted action word {value.word!r} is not in the action space"
            )
        outcomes.append(
            ActionOutcome(
                action_index=canonical,
                word=value.word,
                bound=value.bound,
                exact=value.exact,
            )
        )
    if sorted(outcome.action_index for outcome in outcomes) != list(
        range(len(action_space.words))
    ):
        raise CertificationError("action outcomes do not cover the action space once each")
    objective = _validated_objective(result)
    winner = _select_canonical_winner(outcomes, objective)
    _validate_reported_winner(result.get("best_first"), solver_order, outcomes, objective)
    for outcome in outcomes:
        if outcome.bound < objective:
            raise CertificationError(
                f"action {outcome.word!r} bound {outcome.bound} undercuts the "
                f"solver objective {objective}"
            )
    tree_text = ""
    if with_tree:
        tree_path = work_dir / "tree.txt"
        style = "fBGY" if mode == "normal" else "fGBY"
        tree_command = [
            str(exe_path),
            "-a",
            str(all_txt),
            "-h",
            str(hidden_txt),
            "-g",
            str(horizon),
            "-w",
            winner.word,
            "-S",
            style,
            "-p",
            str(tree_path),
        ]
        if mode == "hard":
            tree_command.append("-H")
        tree_run = subprocess.run(
            tree_command, capture_output=True, text=True, check=False
        )
        if tree_run.returncode != 0:
            raise CertificationError(
                f"tree export exited {tree_run.returncode}: "
                f"{(tree_run.stderr or '').strip()[:2000]}"
            )
        if not tree_path.is_file() or not tree_path.read_text(
            encoding="utf-8"
        ).strip():
            raise CertificationError("tree export produced an empty policy tree")
        tree_text = tree_path.read_text(encoding="utf-8")
        if parse_tree_text(tree_text).get("guess") != winner.word:
            raise CertificationError(
                f"exported tree root does not match canonical winner {winner.word!r}"
            )
    return SearchRecord(
        mode=mode,
        horizon=horizon,
        exhaustive=bool(identity["exhaustive"]),
        best_first_word=winner.word,
        objective=objective,
        outcomes=tuple(outcomes),
        tree_text=tree_text,
        solver_events={"identity": identity, "result": result},
        executable_sha256=executable_sha256,
        compiler_id=compiler_id,
    )


_MARK_TO_CODE = {"b": 0, "y": 1, "g": 2}
_CODE_TO_TILE_TEXT = {0: "b", 1: "y", 2: "g"}


def parse_tree_text(tree_text: str) -> dict:
    """Parse hollow-style -p output into a nested policy tree.

    Each line is one root-to-leaf play sequence of ``<guess> <marks><turn>``
    pairs: guess i is played at turn i+1 and its feedback marks key the branch
    to guess i+1. All-green marks end the line at a solved terminal; a final
    non-green pair at the horizon ends in a loss terminal.
    """
    root: dict = {}
    for raw_line in tree_text.splitlines():
        line = raw_line.strip()
        if not line or line == "IMPOSSIBLE":
            continue
        tokens = line.split()
        if len(tokens) % 2 != 0 or not tokens:
            raise CertificationError(f"malformed tree line: {line!r}")
        guesses = tokens[0::2]
        specs = tokens[1::2]
        node = root
        existing_root_guess = node.get("guess")
        if existing_root_guess is None:
            node["guess"] = guesses[0].lower()
        elif node["guess"] != guesses[0].lower():
            raise CertificationError(
                f"contradictory roots: {node['guess']!r} vs {guesses[0]!r}"
            )
        last = len(guesses) - 1
        for index, spec in enumerate(specs):
            marks = spec[:-1].lower()
            turn = int(spec[-1])
            key = "".join(str(_MARK_TO_CODE[mark]) for mark in marks)
            children = node.setdefault("children", {})
            child = children.get(key)
            if child is None:
                child = {}
                children[key] = child
            if set(marks) == {"g"}:
                if child.get("guess"):
                    raise CertificationError(
                        f"solved terminal collides with a branch at {key!r}"
                    )
                child["terminal"] = "solved"
                child["turn"] = turn
                break
            if index == last:
                if child.get("guess"):
                    raise CertificationError(
                        f"loss terminal collides with a branch at {key!r}"
                    )
                child["terminal"] = "loss"
                break
            next_guess = guesses[index + 1].lower()
            if child.get("terminal"):
                raise CertificationError(
                    f"branch extends through a terminal at {key!r}"
                )
            if child.get("guess") is None:
                child["guess"] = next_guess
            elif child["guess"] != next_guess:
                raise CertificationError(
                    f"contradictory subtree guesses at {key!r}: "
                    f"{child['guess']!r} vs {next_guess!r}"
                )
            node = child
    if not root.get("guess"):
        raise CertificationError("parsed policy tree is empty")
    return root


def build_certificate(
    *,
    view: dict,
    frozen_corpus_dir: Path,
    benchmark_id: str,
    repo_root: Path,
    work_dir: Path,
    jobs: int = 1,
    beta: int | None = None,
    cache_gb: float | None = None,
) -> dict:
    """Run both the tree-producing search and assemble the certificate payload."""
    action_words = tuple(
        (frozen_corpus_dir / "words" / "accepted-guesses.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    answer_words = tuple(
        (frozen_corpus_dir / view["answers_file"]).read_text(encoding="utf-8").split()
    )
    record = run_exact_search(
        repo_root=repo_root,
        work_dir=work_dir,
        action_words=action_words,
        answer_words=answer_words,
        mode=view["mode"],
        horizon=view["horizon_turns"],
        with_tree=True,
        jobs=jobs,
        beta=beta,
        cache_gb=cache_gb,
    )

    action_space = ActionSpace.parse(action_words)
    outcomes = sorted(record.outcomes, key=lambda outcome: outcome.action_index)
    winner = next(
        outcome for outcome in outcomes if outcome.word == record.best_first_word
    )
    certificate = {
        "schema": "wordle.oracle-certificate/v1",
        "identity": {
            "view_id": view["view_id"],
            "benchmark_id": benchmark_id,
            "mode": view["mode"],
            "tier": view["tier"],
            "track": view["track"],
            "rules_sha256": view["rules_sha256"],
            "prior_id": view["prior_id"],
            "action_space_sha256": view["action_space_sha256"],
            "answers_sha256": view["answers_sha256"],
            "answers_file": view["answers_file"],
            "horizon_turns": view["horizon_turns"],
            "legality_profile_id": view["legality_profile_id"],
            "tie_order": view["tie_order"],
            "loss_scoring": view["loss_scoring"],
            "objective": view["objective"],
        },
        "binding": {
            "action_space_sha256": _sha_file(
                frozen_corpus_dir / "words" / "accepted-guesses.txt"
            ),
            "answers_sha256": _sha_file(frozen_corpus_dir / view["answers_file"]),
            "action_space_canonical_digest": action_space.digest,
            "answers_canonical_digest": _sha_hex("\n".join(answer_words)),
            "solver_source_sha256": _sha_file(vendor_dir(repo_root) / "wordle.cpp"),
            "patch_sha256": _sha_file(
                vendor_dir(repo_root) / "patches" / "events-v1.patch"
            ),
            "compiler_id": record.compiler_id,
            "executable_sha256": record.executable_sha256,
        },
        "first_action_proof": {
            "declared_actions": len(action_space.words),
            "winner_index": winner.action_index,
            "winner_word": winner.word,
            "winner_score7_total": record.objective,
            "actions": [
                {
                    "action_index": outcome.action_index,
                    "word": outcome.word,
                    "bound": outcome.bound,
                    "exact": outcome.exact,
                }
                for outcome in outcomes
            ],
        },
        "search": {
            "exhaustive": record.exhaustive,
            "best_first_word": record.best_first_word,
            "nodes": record.solver_events["result"]["nodes"],
            "time_seconds": record.solver_events["result"]["time_seconds"],
        },
        "objective_value": {
            "losses": None,
            "score7_total": None,
            "worst_score": None,
        },
        "policy_tree": parse_tree_text(record.tree_text),
    }
    return certificate


def fill_objective_value(certificate: dict, stats) -> None:
    certificate["objective_value"] = {
        "losses": stats.losses,
        "score7_total": stats.total_score + 7 * stats.losses,
        "worst_score": stats.worst_score if stats.losses == 0 else 7,
    }
