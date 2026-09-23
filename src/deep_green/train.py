"""Clone-training driver: exact-teacher batches, checkpoints, closed runs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_green.backends.torch_train import TorchCloneTrainer
from deep_green.checkpoint import (
    LATEST_NAME,
    CheckpointError,
    load_latest_snapshot,
    load_snapshot,
    save_snapshot,
)
from deep_green.examples import INFINITY, load_examples, recompute_legal_mask
from deep_green.model_spec import RegisteredWidths, parameter_count
from deep_green.rules import ActionSpace
from deep_green.tracking import RunSpec, RunStore, canonical_json


class TrainError(RuntimeError):
    pass


def load_train_config(path: Path) -> dict:
    import tomllib

    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != {"identity", "train"}:
        raise TrainError("training config needs exactly [identity] and [train]")
    identity = config["identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "benchmark_id",
        "rules_sha256",
        "prior_id",
        "action_space_sha256",
        "seed",
        "certificates",
    }:
        raise TrainError("training identity is incomplete or has extra fields")
    if not isinstance(identity["benchmark_id"], str) or not identity["benchmark_id"]:
        raise TrainError("identity.benchmark_id must be a nonempty string")
    if not isinstance(identity["prior_id"], str) or not identity["prior_id"]:
        raise TrainError("identity.prior_id must be a nonempty string")
    for field in ("rules_sha256", "action_space_sha256"):
        if not _is_sha256(identity[field]):
            raise TrainError(f"identity.{field} must be a lowercase sha256")
    if not isinstance(identity["seed"], int) or isinstance(identity["seed"], bool):
        raise TrainError("identity.seed must be an integer")
    certificates = identity["certificates"]
    if not isinstance(certificates, dict) or set(certificates) != {"normal", "hard"}:
        raise TrainError("identity.certificates must contain normal and hard")
    for mode, certificate in certificates.items():
        if not isinstance(certificate, dict) or set(certificate) != {
            "path",
            "sha256",
            "view_id",
            "tier",
        }:
            raise TrainError(f"identity.certificates.{mode} is malformed")
        if not isinstance(certificate["path"], str) or not certificate["path"]:
            raise TrainError(f"identity.certificates.{mode}.path must be nonempty")
        if not _is_sha256(certificate["sha256"]):
            raise TrainError(f"identity.certificates.{mode}.sha256 is malformed")
        if not isinstance(certificate["view_id"], str) or not certificate["view_id"]:
            raise TrainError(f"identity.certificates.{mode}.view_id must be nonempty")
        if (
            not isinstance(certificate["tier"], int)
            or isinstance(certificate["tier"], bool)
            or certificate["tier"] <= 0
        ):
            raise TrainError(f"identity.certificates.{mode}.tier must be positive")
    train = config["train"]
    if not isinstance(train, dict) or set(train) != {
        "lambda_rank",
        "learning_rate",
        "batch_examples_per_update",
    }:
        raise TrainError("training controls are incomplete or have extra fields")
    for field in ("lambda_rank", "learning_rate"):
        value = train[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TrainError(f"train.{field} must be finite")
    if train["lambda_rank"] < 0 or train["learning_rate"] <= 0:
        raise TrainError("rank weight must be nonnegative and learning rate positive")
    batch_size = train["batch_examples_per_update"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise TrainError("train.batch_examples_per_update must be positive")
    return config


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _encode_training_batch(
    examples: list[dict], action_words: tuple[str, ...]
) -> list[dict]:
    """Turn validated records into per-example torch tensors."""
    letter_codes = {chr(ord("a") + i): i for i in range(26)}
    mark_codes = {"b": 0, "y": 1, "g": 2}
    mode_vectors = {"normal": [1.0, 0.0], "hard": [0.0, 1.0]}
    encoded = []
    action_space = ActionSpace.parse(action_words)
    if action_space.words != action_words:
        raise TrainError("action words must be in canonical order")
    max_rows = max(len(ex["public_state"]["rows"]) for ex in examples)
    for record in examples:
        state = record["public_state"]
        row_count = len(state["rows"])
        slot_count = max(row_count, 1)
        letters = np.zeros((1, slot_count, 5), dtype=np.int64)
        marks = np.zeros((1, slot_count, 5), dtype=np.int64)
        row_mask = np.zeros((1, slot_count), dtype=bool)
        row_mask[0, :row_count] = True
        for index, (guess_text, marks_text) in enumerate(state["rows"]):
            letters[0, index] = [letter_codes[c] for c in guess_text]
            marks[0, index] = [mark_codes[c] for c in marks_text]
        mask_values = []
        tie_names = set(record["optimal_tie_set"])
        legal = recompute_legal_mask(record, action_space)
        for word, is_legal in zip(action_words, legal, strict=True):
            mask_values.append(bool(is_legal))
            if word in tie_names and not is_legal:
                raise TrainError(f"tie word {word} recomputed illegal")
        tie_indices = torch.tensor(
            [i for i, w in enumerate(action_words) if w in tie_names], dtype=torch.long
        )
        sparse_map = {e["word"]: (e["losses"], e["score7_total"]) for e in record["sparse_action_values"]}
        encoded.append(
            {
                "row_letters": torch.tensor(letters),
                "row_marks": torch.tensor(marks),
                "row_mask": torch.tensor(row_mask),
                "row_positions": torch.tensor(
                    np.broadcast_to(np.arange(5), letters.shape).copy()
                ),
                "turn": torch.full((1,), float(state["turn"])),
                "mode": torch.tensor([mode_vectors[state["mode"]]], dtype=torch.float32),
                "word_letters": None,
                "word_positions": None,
                "legal_mask": torch.tensor([mask_values], dtype=torch.bool),
                "tie_indices": tie_indices,
                "sparse_map": sparse_map,
                "example_id": record.get(
                    "example_id",
                    hashlib.sha256(
                        json.dumps(record, sort_keys=True).encode()
                    ).hexdigest()[:16],
                ),
                "max_rows": max_rows,
            }
        )
    return encoded


def _verify_training_identity(
    config: dict, examples: list[dict], action_words_path: Path
) -> str:
    from deep_green.oracle.verifier import verify_certificate

    repo_root = Path(__file__).resolve().parents[2]
    frozen_root = repo_root / "benchmarks" / "openwordle-v1" / "frozen"
    identity = config["identity"]
    action_digest = hashlib.sha256(action_words_path.read_bytes()).hexdigest()
    if action_digest != identity["action_space_sha256"]:
        raise TrainError("configured action-space digest does not match the action file")
    certificate_hashes = {}
    certificate_identities = {}
    certificates = {}
    for mode, expected in identity["certificates"].items():
        path = repo_root / expected["path"]
        if not path.is_file():
            raise TrainError(f"missing {mode} certificate: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise TrainError(f"{mode} certificate sha256 mismatch")
        verify_certificate(
            path,
            frozen_root=frozen_root,
            vendor_root=repo_root / "vendor" / "alex1770-wordle",
            build_cache=repo_root / "artifacts" / "oracle" / "build-cache",
        )
        certificate = json.loads(path.read_text(encoding="utf-8"))
        certificate_identity = certificate["identity"]
        if (
            certificate_identity["mode"] != mode
            or certificate_identity["view_id"] != expected["view_id"]
            or certificate_identity["tier"] != expected["tier"]
            or certificate_identity["benchmark_id"] != identity["benchmark_id"]
            or certificate_identity["rules_sha256"] != identity["rules_sha256"]
            or certificate_identity["prior_id"] != identity["prior_id"]
            or certificate_identity["action_space_sha256"]
            != identity["action_space_sha256"]
            or certificate_identity["track"] != "full-compiled"
        ):
            raise TrainError(f"{mode} certificate identity disagrees with config")
        certificate_hashes[mode] = actual
        certificate_identities[mode] = certificate_identity
        certificates[mode] = certificate
    seen_modes = set()
    for record in examples:
        example_identity = record["identity"]
        mode = record["public_state"]["mode"]
        expected = identity["certificates"].get(mode)
        certificate_identity = certificate_identities.get(mode)
        if expected is None or certificate_identity is None:
            raise TrainError(f"example uses unconfigured mode {mode!r}")
        required = {
            "benchmark_id": identity["benchmark_id"],
            "view_id": expected["view_id"],
            "tier": expected["tier"],
            "rules_sha256": identity["rules_sha256"],
            "prior_id": identity["prior_id"],
            "certificate_sha256": expected["sha256"],
            "action_space_sha256": identity["action_space_sha256"],
            "exposure_track": "full-compiled",
        }
        if example_identity != required:
            raise TrainError(f"{mode} example identity disagrees with its certificate")
        if record["public_state"] != {"mode": mode, "rows": [], "turn": 1}:
            raise TrainError("clone bootstrap examples must be certified root states")
        proof = certificates[mode]["first_action_proof"]
        actions = proof["actions"]
        if any(
            action["exact"] is not True or action["bound"] >= INFINITY
            for action in actions
        ):
            raise TrainError("clone bootstrap requires finite exact values for every action")
        winner_value = proof["winner_score7_total"]
        expected_ties = sorted(
            action["word"]
            for action in actions
            if action["exact"] is True and action["bound"] == winner_value
        )
        expected_values = [
            {"word": action["word"], "losses": 0, "score7_total": action["bound"]}
            for action in actions
        ]
        if record["optimal_tie_set"] != expected_ties:
            raise TrainError(f"{mode} optimal tie set is not certificate-derived")
        if record["sparse_action_values"] != expected_values:
            raise TrainError(f"{mode} action values are not certificate-derived")
        seen_modes.add(mode)
    if seen_modes != set(identity["certificates"]):
        raise TrainError("training examples must cover every configured certificate mode")
    return hashlib.sha256(canonical_json(certificate_hashes).encode("utf-8")).hexdigest()


def run_clone(config_path: Path, args_namespace: Any) -> dict:
    """Execute the clone smoke: train, checkpoint each update, close the run."""
    config = load_train_config(Path(config_path))
    identity = config["identity"]
    width = int(args_namespace.model_width)
    if width not in RegisteredWidths.default().widths:
        raise TrainError(f"width {width} is off the registered ladder")

    action_words_path = (
        Path(args_namespace.action_words) if args_namespace.action_words else _default_words()
    )
    action_words = tuple(action_words_path.read_text(encoding="utf-8").split())
    action_space = ActionSpace.parse(action_words)
    if action_space.words != action_words:
        raise TrainError("action words must be in canonical order")
    examples_path = Path(args_namespace.examples)
    examples = load_examples(examples_path)
    if len(examples) != config["train"]["batch_examples_per_update"]:
        raise TrainError("every update must consume the configured full example batch")
    certificate_set_sha256 = _verify_training_identity(
        config, examples, action_words_path
    )

    trainer = TorchCloneTrainer(
        width,
        seed=identity["seed"],
        lambda_rank=float(config["train"]["lambda_rank"]),
        learning_rate=float(config["train"]["learning_rate"]),
    )
    letter_codes = {chr(ord("a") + i): i for i in range(26)}
    wl = torch.tensor([[letter_codes[c] for c in w] for w in action_words], dtype=torch.long)
    wp = torch.broadcast_to(torch.arange(5), (len(action_words), 5)).contiguous()

    runs_root = Path(args_namespace.runs_root)
    model_config = {"tensor_width": width, "train": config["train"]}
    spec = RunSpec(
        code_identity=_code_identity(),
        benchmark_id=identity["benchmark_id"],
        rules_sha256=identity["rules_sha256"],
        prior_id=identity["prior_id"],
        oracle_sha256=certificate_set_sha256,
        action_space_sha256=identity["action_space_sha256"],
        model_config_sha256=hashlib.sha256(
            canonical_json(model_config).encode("utf-8")
        ).hexdigest(),
        backend="torch-cpu",
        optimizer=f"adam-lr{float(config['train']['learning_rate']):g}",
        data_sha256=hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        seed=identity["seed"],
        execution_mode="clone-full-batch",
    )
    resume_from = getattr(args_namespace, "resume", None)
    store = RunStore(runs_root, spec)
    store.require_resume()
    if resume_from and (store.run_dir / "CLOSED").exists():
        raise TrainError("refusing to resume a CLOSED run")
    log = store.event_log()

    max_updates = int(args_namespace.max_updates)
    rng = random.Random(spec.seed)
    checkpoint_dir = store.run_dir / "checkpoints"
    checkpoint_config = {
        "batch_examples_per_update": config["train"]["batch_examples_per_update"],
        "certificate_set_sha256": certificate_set_sha256,
        "lambda_rank": trainer.lambda_rank,
        "learning_rate": float(config["train"]["learning_rate"]),
        "width": width,
    }
    backend_version = "torch-" + torch.__version__
    sampler_position = 0
    training_state = {
        "all_argmax_optimal": True,
        "first_loss": None,
        "last_loss": None,
    }
    logged_updates = set()
    if resume_from:
        resume_path = Path(resume_from)
        try:
            if (resume_path / LATEST_NAME).is_file():
                snapshot = load_latest_snapshot(resume_path)
            else:
                snapshot = load_snapshot(resume_path)
        except CheckpointError as exc:
            log.close()
            raise TrainError(f"cannot resume: {exc}") from None
        state = snapshot["state"]
        if state["config"] != checkpoint_config:
            log.close()
            raise TrainError(
                f"resume config {state['config']!r} does not match "
                f"{checkpoint_config!r}"
            )
        if state["backend_version"] != backend_version:
            log.close()
            raise TrainError(
                f"resume backend_version {state['backend_version']!r} does not "
                f"match {backend_version!r}"
            )
        try:
            restored = {
                name: torch.from_numpy(np.array(value))
                for name, value in snapshot["weights"].items()
            }
            trainer.policy.load_exact(restored)
            trainer.restore_optimizer_arrays(snapshot["optimizer_arrays"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            log.close()
            raise TrainError(f"invalid resume tensors: {exc}") from None
        trainer.update_count = int(state["update_count"])
        sampler_position = state["sampler_position"]
        if sampler_position != 0:
            log.close()
            raise TrainError(
                f"full-batch resume sampler_position must be zero, got {sampler_position}"
            )
        try:
            rng.setstate(_restore_random_state(state["rng_state"]))
        except (KeyError, TypeError, ValueError) as exc:
            log.close()
            raise TrainError(f"invalid rng_state in snapshot: {exc}") from None
        try:
            training_state = _restore_training_state(state.get("training_state"))
        except TrainError as exc:
            log.close()
            raise TrainError(f"invalid training_state in snapshot: {exc}") from None
        logged_updates = {
            event.get("update")
            for event in log.read_recovered()
            if event.get("event") == "update"
        }

    try:
        while trainer.update_count < max_updates:
            batches = []
            for record in examples:
                example = _encode_one(record, action_words, wl, wp)
                keyed_sample = [
                    action_words[rng.randrange(len(action_words))] for _ in range(4)
                ]
                sparse_map = {
                    entry["word"]: (entry["losses"], entry["score7_total"])
                    for entry in record["sparse_action_values"]
                }
                example["negative_indices"] = _select_negatives(
                    trainer, example, action_words, keyed_sample, sparse_map
                )
                batches.append(example)
            result = trainer.train_batch(batches)
            if training_state["first_loss"] is None:
                training_state["first_loss"] = result["loss_total"]
            training_state["last_loss"] = result["loss_total"]
            training_state["all_argmax_optimal"] = (
                training_state["all_argmax_optimal"] and result["argmax_optimal"]
            )
            if trainer.update_count not in logged_updates:
                log.append({"event": "update", **result})
            weights_np, optimizer_np = trainer.snapshot_payload()
            save_snapshot(
                checkpoint_dir / f"update-{trainer.update_count:04d}",
                weights=weights_np,
                optimizer_arrays=optimizer_np,
                rng_state=_serialize_random_state(rng.getstate()),
                sampler_position=sampler_position,
                update_count=trainer.update_count,
                config=checkpoint_config,
                backend_version=backend_version,
                training_state=training_state,
            )

        final_actions = _final_actions(trainer, examples, action_words, wl, wp)
        final_argmax_optimal = all(
            final_actions[record["example_id"]] in record["optimal_tie_set"]
            for record in examples
        )

        inventory = {
            "updates": trainer.update_count,
            "examples": len(examples),
            "width": width,
            "parameter_count": parameter_count(width),
            "final_checkpoint": f"checkpoints/update-{trainer.update_count:04d}",
            "losses_first_last": (
                training_state["first_loss"],
                training_state["last_loss"],
            ),
            "all_argmax_optimal": training_state["all_argmax_optimal"],
            "final_actions": final_actions,
            "final_argmax_optimal": final_argmax_optimal,
        }
        store.close(inventory)
    finally:
        log.close()
    return {
        "run_id": store.run_id,
        "run_dir": str(store.run_dir),
        "inventory": inventory,
    }


def _serialize_random_state(state) -> dict:
    algorithm, internal_state, gauss_next = state
    return {
        "algorithm": algorithm,
        "internal_state": list(internal_state),
        "gauss_next": gauss_next,
    }


def _restore_training_state(payload) -> dict:
    if not isinstance(payload, dict) or set(payload) != {
        "all_argmax_optimal",
        "first_loss",
        "last_loss",
    }:
        raise TrainError("training_state is malformed")
    optimal = payload["all_argmax_optimal"]
    if not isinstance(optimal, bool):
        raise TrainError("training_state.all_argmax_optimal must be a bool")
    first, last = payload["first_loss"], payload["last_loss"]
    if (first is None) != (last is None):
        raise TrainError("training_state losses must be both null or both numeric")
    if first is not None:
        for value in (first, last):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise TrainError("training_state losses must be finite numbers")
    return {
        "all_argmax_optimal": optimal,
        "first_loss": first,
        "last_loss": last,
    }


def _restore_random_state(payload) -> tuple:
    if not isinstance(payload, dict) or sorted(payload) != [
        "algorithm",
        "gauss_next",
        "internal_state",
    ]:
        raise TrainError("rng_state is malformed")
    algorithm = payload["algorithm"]
    if not isinstance(algorithm, int) or isinstance(algorithm, bool):
        raise TrainError("rng_state.algorithm must be an integer")
    internal_state = payload["internal_state"]
    if (
        not isinstance(internal_state, list)
        or not internal_state
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in internal_state
        )
    ):
        raise TrainError("rng_state.internal_state must be a nonempty int list")
    gauss_next = payload["gauss_next"]
    if gauss_next is not None and (
        not isinstance(gauss_next, (int, float)) or isinstance(gauss_next, bool)
    ):
        raise TrainError("rng_state.gauss_next must be numeric or null")
    return (algorithm, tuple(internal_state), gauss_next)


def _encode_one(record: dict, action_words, wl, wp) -> dict:
    encoded = _encode_training_batch([record], action_words)[0]
    encoded["word_letters"] = wl
    encoded["word_positions"] = wp
    del encoded["sparse_map"]
    del encoded["example_id"]
    del encoded["max_rows"]
    return encoded


def _final_actions(trainer, examples, action_words, wl, wp) -> dict[str, str]:
    actions = {}
    for record in examples:
        example = _encode_one(record, action_words, wl, wp)
        with torch.no_grad():
            logits = trainer.policy.score_all(**{
                key: value
                for key, value in example.items()
                if key in ("row_letters", "row_marks", "row_positions", "row_mask",
                           "turn", "mode", "word_letters", "word_positions")
            })
        masked = logits.masked_fill(~example["legal_mask"].bool(), float("-inf"))
        actions[record["example_id"]] = action_words[int(torch.argmax(masked[0]))]
    return actions


def _select_negatives(trainer, example, action_words, keyed_sample, sparse_map) -> torch.Tensor:
    with torch.no_grad():
        logits = trainer.policy.score_all(**{
            k: v
            for k, v in example.items()
            if k in ("row_letters", "row_marks", "row_positions", "row_mask",
                     "turn", "mode", "word_letters", "word_positions")
        })
    from deep_green.losses import select_hard_negative_indices

    ranked_words = sorted(
        action_words,
        key=lambda word: (*sparse_map.get(word, (INFINITY, INFINITY)), word),
    )
    ranks = {word: index for index, word in enumerate(ranked_words)}
    return select_hard_negative_indices(
        student_logits_row=logits[0],
        action_words=action_words,
        tie_set=[action_words[int(i)] for i in example["tie_indices"].tolist()],
        teacher_ranks=ranks,
        exact_values=sparse_map,
        keyed_sample=keyed_sample,
        legal_mask=example["legal_mask"][0],
        limit=32,
    )


def _default_words() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks/openwordle-v1/frozen/corpus/words/accepted-guesses.txt"
    )


def _code_identity() -> str:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = "unknown"
    return f"git:{head}"
