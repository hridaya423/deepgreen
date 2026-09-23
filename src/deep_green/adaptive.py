"""Adaptive campaign: certified tree corpus, DAgger loop, certified gates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ADAPTIVE_SCHEMA = "wordle.adaptive-campaign/v1"

_DIGIT_MARK = {"0": "b", "1": "y", "2": "g"}
_MODE_VECTOR = {"normal": (1.0, 0.0), "hard": (0.0, 1.0)}
_MAX_PLAYED_ROWS = 5


class AdaptiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeExample:
    state_key: str
    mode: str
    rows: tuple[tuple[str, str], ...]
    target_word: str
    target_index: int
    legal_mask: tuple[bool, ...]
    visits: int
    child_keys: tuple[str, ...]


@dataclass(frozen=True)
class CertifiedTreeCorpus:
    action_words: tuple[str, ...]
    examples: tuple[TreeExample, ...]
    root_keys: dict[str, str]
    certificate_sha256s: dict[str, str]


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _canonical(payload) -> str:
    from deep_green.tracking import canonical_json

    return canonical_json(payload)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_key(mode: str, rows: tuple[tuple[str, str], ...]) -> str:
    return _sha(
        _canonical(
            {"mode": mode, "rows": [list(row) for row in rows], "turn": len(rows) + 1}
        )
    )


def load_adaptive_config(config_path) -> dict:
    """Strictly parse the adaptive TOML and hash its raw bytes."""
    import tomllib

    from deep_green.model_spec import RegisteredWidths

    config_path = Path(config_path)
    raw = config_path.read_bytes()
    try:
        config = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AdaptiveError(f"adaptive config is not valid TOML: {exc}") from exc
    if set(config) != {"identity", "adaptive"}:
        raise AdaptiveError("adaptive config needs exactly [identity] and [adaptive]")
    identity = config["identity"]
    if not isinstance(identity, dict) or not set(identity) <= {
        "benchmark_id",
        "rules_sha256",
        "prior_id",
        "action_space_sha256",
        "seed",
        "certificates",
        "frozen_root",
    } or not {
        "benchmark_id",
        "rules_sha256",
        "prior_id",
        "action_space_sha256",
        "seed",
        "certificates",
    } <= set(identity):
        raise AdaptiveError("identity is incomplete or has extra fields")
    if "frozen_root" in identity and (
        not isinstance(identity["frozen_root"], str)
        or not identity["frozen_root"]
    ):
        raise AdaptiveError("identity.frozen_root must be a nonempty string")
    if not isinstance(identity["benchmark_id"], str) or not identity["benchmark_id"]:
        raise AdaptiveError("identity.benchmark_id must be a nonempty string")
    if not isinstance(identity["prior_id"], str) or not identity["prior_id"]:
        raise AdaptiveError("identity.prior_id must be a nonempty string")
    for field in ("rules_sha256", "action_space_sha256"):
        if not _is_sha256(identity[field]):
            raise AdaptiveError(f"identity.{field} must be a lowercase sha256")
    if not isinstance(identity["seed"], int) or isinstance(identity["seed"], bool):
        raise AdaptiveError("identity.seed must be an integer")
    certificates = identity["certificates"]
    if not isinstance(certificates, dict) or set(certificates) != {"normal", "hard"}:
        raise AdaptiveError("identity.certificates must contain normal and hard")
    for mode, certificate in certificates.items():
        if not isinstance(certificate, dict) or set(certificate) != {
            "path",
            "sha256",
            "view_id",
            "tier",
        }:
            raise AdaptiveError(f"identity.certificates.{mode} is malformed")
        if not isinstance(certificate["path"], str) or not certificate["path"]:
            raise AdaptiveError(f"identity.certificates.{mode}.path must be nonempty")
        if not _is_sha256(certificate["sha256"]):
            raise AdaptiveError(
                f"identity.certificates.{mode}.sha256 is malformed"
            )
        if not isinstance(certificate["view_id"], str) or not certificate["view_id"]:
            raise AdaptiveError(f"identity.certificates.{mode}.view_id must be nonempty")
        if (
            not isinstance(certificate["tier"], int)
            or isinstance(certificate["tier"], bool)
            or certificate["tier"] <= 0
        ):
            raise AdaptiveError(f"identity.certificates.{mode}.tier must be positive")
    if len({certificate["tier"] for certificate in certificates.values()}) != 1:
        raise AdaptiveError("identity.certificates must use one shared tier")
    adaptive = config["adaptive"]
    if not isinstance(adaptive, dict) or set(adaptive) != {
        "widths",
        "learning_rate",
        "initial_epochs",
        "dagger_epochs",
        "max_dagger_rounds",
        "gradient_clip_norm",
        "stall_patience",
        "fine_tune_learning_rate",
        "fine_tune_threshold",
        "release_score_scale",
        "campaign_revision",
    }:
        raise AdaptiveError("adaptive section is incomplete or has extra fields")
    widths = adaptive["widths"]
    ladder = RegisteredWidths.default().widths
    if (
        not isinstance(widths, list)
        or not widths
        or any(not isinstance(w, int) or isinstance(w, bool) for w in widths)
        or len(set(widths)) != len(widths)
        or tuple(widths) != tuple(sorted(widths))
        or any(w not in ladder for w in widths)
    ):
        raise AdaptiveError(
            "adaptive.widths must be a strictly increasing list of registered widths"
        )
    learning_rate = adaptive["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise AdaptiveError("adaptive.learning_rate must be positive and finite")
    for field in ("initial_epochs", "dagger_epochs", "max_dagger_rounds"):
        value = adaptive[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AdaptiveError(f"adaptive.{field} must be a positive integer")
    clip_norm = adaptive["gradient_clip_norm"]
    if (
        isinstance(clip_norm, bool)
        or not isinstance(clip_norm, (int, float))
        or not math.isfinite(clip_norm)
        or clip_norm <= 0
    ):
        raise AdaptiveError("adaptive.gradient_clip_norm must be positive and finite")
    stall_patience = adaptive["stall_patience"]
    if (
        not isinstance(stall_patience, int)
        or isinstance(stall_patience, bool)
        or stall_patience <= 0
        or stall_patience > adaptive["max_dagger_rounds"]
    ):
        raise AdaptiveError(
            "adaptive.stall_patience must be a positive integer no larger "
            "than max_dagger_rounds"
        )
    fine_rate = adaptive["fine_tune_learning_rate"]
    if (
        isinstance(fine_rate, bool)
        or not isinstance(fine_rate, (int, float))
        or not math.isfinite(fine_rate)
        or fine_rate <= 0
        or fine_rate >= learning_rate
    ):
        raise AdaptiveError(
            "adaptive.fine_tune_learning_rate must be positive, finite, and "
            "strictly below learning_rate"
        )
    threshold = adaptive["fine_tune_threshold"]
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold <= 0
        or threshold > 1024
    ):
        raise AdaptiveError(
            "adaptive.fine_tune_threshold must be a positive integer no larger "
            "than 1024"
        )
    score_scale = adaptive["release_score_scale"]
    if (
        isinstance(score_scale, bool)
        or not isinstance(score_scale, (int, float))
        or not math.isfinite(score_scale)
        or score_scale <= 0
        or score_scale > 1
    ):
        raise AdaptiveError(
            "adaptive.release_score_scale must be positive, finite, and <= 1"
        )
    revision = adaptive["campaign_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise AdaptiveError(
            "adaptive.campaign_revision must be a positive integer"
        )
    return {
        "identity": dict(identity),
        "adaptive": dict(adaptive),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _feedback_key(feedback) -> str:
    from deep_green.rules import Tile

    order = {Tile.ABSENT: "0", Tile.PRESENT: "1", Tile.CORRECT: "2"}
    return "".join(order[tile] for tile in feedback.tiles)


def load_certified_tree_corpus(config_path) -> CertifiedTreeCorpus:
    """Build the example corpus from the two verified certificate trees."""
    from deep_green.oracle.verifier import verify_certificate
    from deep_green.rules import (
        ActionSpace,
        Board,
        Feedback,
        GuessRow,
        Mode,
        Word5,
        legal_actions,
        score_guess,
    )

    config = load_adaptive_config(config_path)
    identity = config["identity"]
    repo_root = Path(__file__).resolve().parents[2]
    frozen_root = repo_root / str(
        identity.get("frozen_root", "benchmarks/openwordle-v1/frozen")
    )
    vendor_root = repo_root / "vendor" / "alex1770-wordle"
    build_cache = repo_root / "artifacts" / "oracle" / "build-cache"

    actions_path = frozen_root / "corpus" / "words" / "accepted-guesses.txt"
    actions_data = actions_path.read_bytes()
    if hashlib.sha256(actions_data).hexdigest() != identity["action_space_sha256"]:
        raise AdaptiveError("frozen action file fails the configured digest")
    action_words = tuple(actions_data.decode("utf-8").split())
    action_space = ActionSpace.parse(action_words)
    if action_space.words != action_words:
        raise AdaptiveError("frozen action words are not canonical")

    examples: dict[str, TreeExample] = {}
    root_keys: dict[str, str] = {}
    certificate_sha256s: dict[str, str] = {}

    def walk(mode: str, node: dict, rows: tuple[tuple[str, str], ...], remaining) -> str | None:
        if "terminal" in node:
            return None
        guess = node.get("guess")
        if not isinstance(guess, str):
            raise AdaptiveError("certificate branch node is missing its guess")
        turn = len(rows) + 1
        board = Board.create(
            Mode(mode),
            tuple(
                GuessRow(Word5.parse(word), Feedback.parse(marks))
                for word, marks in rows
            ),
            turn,
        )
        mask = tuple(bool(v) for v in legal_actions(board, action_space))
        try:
            target_index = action_space.words.index(guess)
        except ValueError as exc:
            raise AdaptiveError(f"tree guess {guess!r} is not in the vocabulary") from exc
        if not mask[target_index]:
            raise AdaptiveError(f"tree guess {guess!r} is illegal under {mode} rules")
        state_key = _state_key(mode, rows)
        children = node.get("children")
        if not isinstance(children, dict):
            raise AdaptiveError("certificate branch node is missing children")
        partition: dict[str, list[str]] = {}
        guess_word = Word5.parse(guess)
        for answer_text in remaining:
            key = _feedback_key(score_guess(guess_word, Word5.parse(answer_text)))
            partition.setdefault(key, []).append(answer_text)
        if set(children) != set(partition):
            raise AdaptiveError(
                "certificate child keys do not equal the replayed feedback partition"
            )
        child_keys: list[tuple[str, str]] = []
        for key in sorted(children):
            child = children[key]
            if not isinstance(child, dict):
                raise AdaptiveError("certificate child is not an object")
            if "terminal" in child:
                continue
            if "guess" not in child:
                raise AdaptiveError("certificate child has neither terminal nor guess")
            child_rows = rows + ((guess, "".join(_DIGIT_MARK[c] for c in key)),)
            child_state = _state_key(mode, child_rows)
            child_keys.append((key, child_state))
            walk(mode, child, child_rows, tuple(partition[key]))
        existing = examples.get(state_key)
        if existing is not None and (
            existing.target_index != target_index or existing.mode != mode
        ):
            raise AdaptiveError(f"conflicting duplicate tree state {state_key}")
        if existing is None:
            examples[state_key] = TreeExample(
                state_key=state_key,
                mode=mode,
                rows=rows,
                target_word=guess,
                target_index=target_index,
                legal_mask=mask,
                visits=len(remaining),
                child_keys=tuple(key for _, key in sorted(child_keys)),
            )
        return state_key

    for mode in ("normal", "hard"):
        expected = identity["certificates"][mode]
        certificate_path = repo_root / expected["path"]
        if not certificate_path.is_file():
            raise AdaptiveError(f"missing {mode} certificate: {certificate_path}")
        certificate_data = certificate_path.read_bytes()
        certificate_sha = hashlib.sha256(certificate_data).hexdigest()
        if certificate_sha != expected["sha256"]:
            raise AdaptiveError(f"{mode} certificate sha256 mismatch")
        verify_certificate(
            certificate_path,
            frozen_root=frozen_root,
            vendor_root=vendor_root,
            build_cache=build_cache,
        )
        certificate = json.loads(certificate_data.decode("utf-8"))
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
            raise AdaptiveError(f"{mode} certificate identity disagrees with config")
        answers_path = frozen_root / "corpus" / certificate_identity["answers_file"]
        answers_data = answers_path.read_bytes()
        if (
            hashlib.sha256(answers_data).hexdigest()
            != certificate_identity["answers_sha256"]
        ):
            raise AdaptiveError(f"{mode} answers file fails the certificate digest")
        answers = tuple(answers_data.decode("utf-8").split())
        tree = certificate["policy_tree"]
        root_keys[mode] = walk(mode, tree, (), answers)
        certificate_sha256s[mode] = certificate_sha

    return CertifiedTreeCorpus(
        action_words=action_words,
        examples=tuple(
            examples[key] for key in sorted(examples)
        ),
        root_keys=root_keys,
        certificate_sha256s=certificate_sha256s,
    )


def _encode_corpus(corpus: CertifiedTreeCorpus) -> dict:
    """Encode the corpus into fixed padded tensors with no hidden inputs."""
    from deep_green.encoding import LETTER_CODES, MARK_CODES
    from deep_green.numeric_policy import encode_lexicon

    count = len(corpus.examples)
    actions = len(corpus.action_words)
    row_letters = torch.zeros((count, _MAX_PLAYED_ROWS, 5), dtype=torch.long)
    row_marks = torch.zeros((count, _MAX_PLAYED_ROWS, 5), dtype=torch.long)
    row_positions = (
        torch.arange(5).view(1, 1, 5).expand(count, _MAX_PLAYED_ROWS, 5).contiguous()
    )
    row_mask = torch.zeros((count, _MAX_PLAYED_ROWS), dtype=torch.bool)
    turn = torch.zeros(count, dtype=torch.float32)
    mode = torch.zeros((count, 2), dtype=torch.float32)
    legal_mask = torch.zeros((count, actions), dtype=torch.bool)
    targets = torch.zeros(count, dtype=torch.long)
    for index, example in enumerate(corpus.examples):
        for row_index, (guess_text, marks_text) in enumerate(example.rows):
            row_letters[index, row_index] = torch.tensor(
                [LETTER_CODES[c] for c in guess_text], dtype=torch.long
            )
            row_marks[index, row_index] = torch.tensor(
                [MARK_CODES[c] for c in marks_text], dtype=torch.long
            )
        row_mask[index, : len(example.rows)] = True
        turn[index] = float(len(example.rows) + 1)
        mode[index] = torch.tensor(_MODE_VECTOR[example.mode], dtype=torch.float32)
        legal_mask[index] = torch.tensor(example.legal_mask, dtype=torch.bool)
        targets[index] = example.target_index
    letters, positions = encode_lexicon(list(corpus.action_words))
    return {
        "row_letters": row_letters,
        "row_marks": row_marks,
        "row_positions": row_positions,
        "row_mask": row_mask,
        "turn": turn,
        "mode": mode,
        "word_letters": torch.tensor(letters, dtype=torch.long),
        "word_positions": torch.tensor(positions, dtype=torch.long),
        "legal_mask": legal_mask,
        "targets": targets,
    }


def train_epochs(
    trainer,
    tensors: dict,
    weights: torch.Tensor,
    epochs: int,
    *,
    gradient_clip_norm: float,
) -> list[float]:
    """One epoch is one true full-corpus weighted optimizer step."""
    losses: list[float] = []
    order = torch.arange(tensors["targets"].shape[0])
    for _ in range(epochs):
        logits = trainer.policy.score_all(
            row_letters=tensors["row_letters"],
            row_marks=tensors["row_marks"],
            row_positions=tensors["row_positions"],
            row_mask=tensors["row_mask"],
            turn=tensors["turn"],
            mode=tensors["mode"],
            word_letters=tensors["word_letters"],
            word_positions=tensors["word_positions"],
        )
        masked = logits.masked_fill(~tensors["legal_mask"], -float("inf"))
        per_state = -torch.log_softmax(masked, dim=1)[order, tensors["targets"]]
        loss = (per_state * weights).sum() / weights.sum()
        trainer.optimizer.zero_grad()
        loss.backward()
        if not math.isfinite(float(loss.detach())):
            raise AdaptiveError("adaptive epoch produced a non-finite loss")
        for parameter in trainer.policy.parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise AdaptiveError("adaptive epoch produced a non-finite gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainer.policy.parameters(), gradient_clip_norm
        )
        if not math.isfinite(float(gradient_norm)):
            raise AdaptiveError("adaptive epoch produced a non-finite gradient norm")
        trainer.optimizer.step()
        trainer.update_count += 1
        losses.append(float(loss.detach()))
    return losses


def predict_indices(trainer, tensors: dict, *, chunk_size: int = 256) -> list[int]:
    """Masked stable argmax per state; the lowest legal index wins ties."""
    predictions: list[int] = []
    count = tensors["targets"].shape[0]
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            logits = trainer.policy.score_all(
                row_letters=tensors["row_letters"][start:stop],
                row_marks=tensors["row_marks"][start:stop],
                row_positions=tensors["row_positions"][start:stop],
                row_mask=tensors["row_mask"][start:stop],
                turn=tensors["turn"][start:stop],
                mode=tensors["mode"][start:stop],
                word_letters=tensors["word_letters"],
                word_positions=tensors["word_positions"],
            )
            masked = logits.masked_fill(
                ~tensors["legal_mask"][start:stop], -float("inf")
            )
            predictions.extend(int(v) for v in torch.argmax(masked, dim=1).tolist())
    return predictions


def first_disagreement_frontier(
    corpus: CertifiedTreeCorpus, predicted_indices
) -> dict[str, int]:
    """Count answer rollouts whose first divergent action happens at each state."""
    predicted = list(predicted_indices)
    if len(predicted) != len(corpus.examples):
        raise AdaptiveError("prediction coverage does not match the corpus")
    actions = len(corpus.action_words)
    for value in predicted:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < actions
        ):
            raise AdaptiveError(f"invalid predicted action index {value!r}")
    index_of = {e.state_key: i for i, e in enumerate(corpus.examples)}
    frontier: dict[str, int] = {}

    def visit(state_key: str) -> None:
        example_index = index_of.get(state_key)
        if example_index is None:
            raise AdaptiveError(f"tree child state {state_key} is missing")
        example = corpus.examples[example_index]
        if predicted[example_index] != example.target_index:
            frontier[state_key] = example.visits
            return
        for child_key in example.child_keys:
            visit(child_key)

    for root in corpus.root_keys.values():
        visit(root)
    return frontier


def promotion_decision(
    *,
    divergent_answers: int,
    previous_divergent_answers: int,
    consecutive_stalls: int,
    stall_patience: int,
    dagger_round: int,
    max_dagger_rounds: int,
    is_last_width: bool,
) -> tuple[str, int]:
    """Pure campaign transition plus the updated stall counter."""
    if divergent_answers == 0:
        return ("candidate", 0)
    stalls = (
        0
        if divergent_answers < previous_divergent_answers
        else consecutive_stalls + 1
    )
    if stalls >= stall_patience:
        return (("incomplete" if is_last_width else "promote"), stalls)
    if dagger_round >= max_dagger_rounds:
        return ("incomplete", stalls)
    return ("continue", stalls)


def phase_learning_rate(
    *,
    base_learning_rate: float,
    fine_tune_learning_rate: float,
    fine_tune_threshold: int,
    previous_divergent_answers: int,
    fine_tuning_started: bool,
) -> tuple[float, bool]:
    """Sticky per-width LR schedule: once fine-tuning starts it never reverts."""
    started = (
        fine_tuning_started
        or previous_divergent_answers <= fine_tune_threshold
    )
    return (
        fine_tune_learning_rate if started else base_learning_rate,
        started,
    )


def scaled_release_tensors(
    tensors: dict[str, np.ndarray], scale: float
) -> dict[str, np.ndarray]:
    """Multiply the scoring head by a positive scale; argmax is unchanged."""
    from deep_green.model_spec import RegisteredWidths, parameter_count
    from deep_green.release import ReleaseError, validate_tensors

    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
        or scale > 1
    ):
        raise AdaptiveError("release score scale must be positive, finite, and <= 1")
    total = sum(int(np.prod(np.asarray(v).shape)) for v in tensors.values())
    width = next(
        (
            w
            for w in RegisteredWidths.default().widths
            if parameter_count(w) == total
        ),
        None,
    )
    if width is None:
        raise AdaptiveError("tensor payload matches no registered width")
    try:
        validate_tensors(tensors, width)
    except ReleaseError as exc:
        raise AdaptiveError(f"release tensors failed the contract: {exc}") from exc
    scaled = {}
    for name, value in tensors.items():
        array = np.array(value, dtype=np.float32)
        if name in ("score.bilinear", "score.bias"):
            array = np.asarray(
                array * np.float32(scale), dtype=np.float32
            )
        scaled[name] = array
    return scaled


def verify_release_backend_parity(
    corpus: CertifiedTreeCorpus,
    tensors: dict[str, np.ndarray],
    width: int,
    *,
    tolerance: float = 1e-5,
    chunk_size: int = 128,
) -> dict:
    """Torch/MLX/NumPy must agree on logits, argmax, and certified targets."""
    try:
        import mlx.core as mx

        from deep_green.backends.mlx_policy import MlxWordlePolicy
    except ImportError as exc:
        raise AdaptiveError("MLX backend is required on this host") from exc
    from deep_green.backends.torch_policy import TorchWordlePolicy
    from deep_green.numeric_policy import score_logits

    np_tensors = {
        name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()
    }
    torch_policy = TorchWordlePolicy(width)
    torch_policy.load_exact(
        {
            name: torch.from_numpy(np.array(value, dtype=np.float32))
            for name, value in np_tensors.items()
        }
    )
    mlx_policy = MlxWordlePolicy(width)
    mlx_policy.load_exact(
        {
            name: mx.array(np.array(value, dtype=np.float32))
            for name, value in np_tensors.items()
        }
    )
    encoded = _encode_corpus(corpus)
    inputs = {name: value.numpy() for name, value in encoded.items()}
    count = len(corpus.examples)
    max_delta = {"torch_numpy": 0.0, "torch_mlx": 0.0, "numpy_mlx": 0.0}
    disagreements = {"torch": 0, "numpy": 0, "mlx": 0}
    argmax_mismatch = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        part = {name: value[start:stop] for name, value in inputs.items()}
        targets = part["targets"]
        torch_logits = (
            torch_policy.score_all(
                row_letters=encoded["row_letters"][start:stop],
                row_marks=encoded["row_marks"][start:stop],
                row_positions=encoded["row_positions"][start:stop],
                row_mask=encoded["row_mask"][start:stop],
                turn=encoded["turn"][start:stop],
                mode=encoded["mode"][start:stop],
                word_letters=encoded["word_letters"],
                word_positions=encoded["word_positions"],
            )
            .detach()
            .numpy()
        )
        numpy_logits = score_logits(
            tensors=np_tensors,
            row_letters=part["row_letters"],
            row_marks=part["row_marks"],
            row_positions=part["row_positions"],
            row_mask=part["row_mask"],
            turn=part["turn"],
            mode=part["mode"],
            word_letters=inputs["word_letters"],
            word_positions=inputs["word_positions"],
        )
        mlx_logits = np.asarray(
            mlx_policy(
                row_letters=mx.array(part["row_letters"]),
                row_marks=mx.array(part["row_marks"]),
                row_positions=mx.array(part["row_positions"]),
                row_mask=mx.array(part["row_mask"]),
                turn=mx.array(part["turn"]),
                mode=mx.array(part["mode"]),
                word_letters=mx.array(inputs["word_letters"]),
                word_positions=mx.array(inputs["word_positions"]),
            ),
            dtype=np.float32,
        )
        max_delta["torch_numpy"] = max(
            max_delta["torch_numpy"],
            float(np.abs(torch_logits - numpy_logits).max()),
        )
        max_delta["torch_mlx"] = max(
            max_delta["torch_mlx"],
            float(np.abs(torch_logits - mlx_logits).max()),
        )
        max_delta["numpy_mlx"] = max(
            max_delta["numpy_mlx"],
            float(np.abs(numpy_logits - mlx_logits).max()),
        )
        mask = part["legal_mask"].astype(bool)
        argmaxes = {
            name: np.argmax(np.where(mask, logits, -np.inf), axis=1)
            for name, logits in (
                ("torch", torch_logits),
                ("numpy", numpy_logits),
                ("mlx", mlx_logits),
            )
        }
        if not (
            np.array_equal(argmaxes["torch"], argmaxes["numpy"])
            and np.array_equal(argmaxes["torch"], argmaxes["mlx"])
        ):
            argmax_mismatch += 1
        for name, argmax in argmaxes.items():
            disagreements[name] += int(np.count_nonzero(argmax != targets))
    report = {
        "states": count,
        "tolerance": tolerance,
        "max_abs_delta_torch_numpy": max_delta["torch_numpy"],
        "max_abs_delta_torch_mlx": max_delta["torch_mlx"],
        "max_abs_delta_numpy_mlx": max_delta["numpy_mlx"],
        "torch_disagreements": disagreements["torch"],
        "numpy_disagreements": disagreements["numpy"],
        "mlx_disagreements": disagreements["mlx"],
    }
    if any(value > tolerance for value in max_delta.values()) or argmax_mismatch:
        raise AdaptiveError(f"backend parity exceeds {tolerance}: {report}")
    if any(disagreements.values()):
        raise AdaptiveError(f"backend argmax diverges from certified targets: {report}")
    return report


def _all_node_disagreements(corpus: CertifiedTreeCorpus, predicted) -> int:
    return sum(
        1
        for i, example in enumerate(corpus.examples)
        if predicted[i] != example.target_index
    )


def _assert_tree_trace(corpus: CertifiedTreeCorpus, predicted, frontier) -> None:
    """In-process certificate-tree replay gate before the sandboxed census."""
    if sum(frontier.values()) != 0:
        raise AdaptiveError("tree frontier is nonempty during candidate gate")
    index_of = {e.state_key: i for i, e in enumerate(corpus.examples)}
    visited = 0

    def visit(state_key: str) -> None:
        nonlocal visited
        example_index = index_of.get(state_key)
        if example_index is None:
            raise AdaptiveError(f"tree state {state_key} is missing from the corpus")
        example = corpus.examples[example_index]
        if predicted[example_index] != example.target_index:
            raise AdaptiveError(
                f"candidate diverges at reachable state {state_key}"
            )
        visited += 1
        for child_key in example.child_keys:
            visit(child_key)

    for root in corpus.root_keys.values():
        visit(root)
    if visited != len(corpus.examples):
        raise AdaptiveError(
            f"candidate replay reached {visited} of {len(corpus.examples)} states"
        )


def _write_atomic(path: Path, text: str) -> None:
    import uuid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(text)
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


def _campaign_id(config, corpus) -> str:
    from deep_green.model_spec import TENSOR_SPEC_VERSION

    return _sha(
        _canonical(
            {
                "identity": config["identity"],
                "adaptive": config["adaptive"],
                "certificate_sha256s": corpus.certificate_sha256s,
                "policy_spec": TENSOR_SPEC_VERSION,
            }
        )
    )[:16]


def run_adaptive(config_path, *, runs_root, output_release) -> dict:
    """Run the adaptive ladder campaign under an exclusive writer lock."""
    config_path = Path(config_path)
    config = load_adaptive_config(config_path)
    corpus = load_certified_tree_corpus(config_path)
    campaign_id = _campaign_id(config, corpus)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_path = runs_root / f".campaign-{campaign_id}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdaptiveError(
                f"campaign {campaign_id} already has an active writer"
            ) from exc
        try:
            return _run_adaptive_locked(
                config,
                corpus,
                campaign_id,
                runs_root=runs_root,
                output_release=output_release,
            )
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _run_adaptive_locked(
    config, corpus, campaign_id, *, runs_root, output_release
) -> dict:
    """Run the campaign body; the caller holds the exclusive writer lock."""
    from deep_green.backends.torch_train import TorchCloneTrainer
    from deep_green.checkpoint import save_snapshot
    from deep_green.evaluate import (
        EvaluationError,
        evaluate_certified,
        save_census_report,
    )
    from deep_green.inference import PolicyRuntimeError
    from deep_green.model_spec import TENSOR_SPEC_VERSION
    from deep_green.oracle.verifier import CertificateError, VerificationError
    from deep_green.release import (
        INFERENCE_CLASS,
        ReleaseError,
        load_release,
        publish_release,
    )
    from deep_green.tracking import EventLog

    identity = config["identity"]
    adaptive = config["adaptive"]

    repo_root = Path(__file__).resolve().parents[2]
    frozen_root = repo_root / str(
        identity.get("frozen_root", "benchmarks/openwordle-v1/frozen")
    )
    actions_path = frozen_root / "corpus" / "words" / "accepted-guesses.txt"

    run_dir = runs_root / campaign_id
    if os.path.lexists(run_dir):
        state = (
            "CLOSED" if (run_dir / "CLOSED").exists() else "incomplete/partial"
        )
        raise AdaptiveError(
            f"campaign run {campaign_id} already exists ({state}); "
            "choose a new hash-bound campaign revision/config"
        )
    manifest = {
        "schema": ADAPTIVE_SCHEMA,
        "campaign_id": campaign_id,
        "config_sha256": config["config_sha256"],
        "policy_spec": TENSOR_SPEC_VERSION,
        "certificate_sha256s": corpus.certificate_sha256s,
        "identity": {
            "benchmark_id": identity["benchmark_id"],
            "rules_sha256": identity["rules_sha256"],
            "prior_id": identity["prior_id"],
            "action_space_sha256": identity["action_space_sha256"],
            "seed": identity["seed"],
        },
    }
    manifest_path = run_dir / "manifest.json"
    run_dir.mkdir()
    _write_atomic(manifest_path, _canonical(manifest) + "\n")

    event_log = EventLog(run_dir / "events.jsonl")
    tensors = _encode_corpus(corpus)
    order = [e.state_key for e in corpus.examples]
    weights = {
        example.state_key: float(example.visits) for example in corpus.examples
    }

    def weight_vector() -> torch.Tensor:
        return torch.tensor([weights[k] for k in order], dtype=torch.float32)

    def weight_digest() -> str:
        return _sha(_canonical({k: weights[k] for k in order}))

    def save_weights_map() -> None:
        _write_atomic(
            run_dir / "weights.json",
            _canonical({k: weights[k] for k in order}) + "\n",
        )

    release_identity = {
        "benchmark_id": identity["benchmark_id"],
        "rules_sha256": identity["rules_sha256"],
        "prior_id": identity["prior_id"],
        "action_space_sha256": identity["action_space_sha256"],
        "track": "full-compiled",
        "inference_class": INFERENCE_CLASS,
    }

    width_attempts: list[dict] = []
    evaluations: list[dict] = []
    status = "INCOMPLETE"
    selected_width = None
    release_info = None
    backend_parity_report = None

    def evaluate_candidate(trainer, width: int, predicted, frontier) -> bool:
        nonlocal release_info, selected_width, backend_parity_report
        _assert_tree_trace(corpus, predicted, frontier)
        candidate_dir = run_dir / "candidates" / f"width-{width}"
        weights_np, _ = trainer.snapshot_payload()
        scaled = scaled_release_tensors(
            weights_np, float(adaptive["release_score_scale"])
        )
        try:
            backend_report = verify_release_backend_parity(corpus, scaled, width)
        except AdaptiveError as exc:
            evaluations.append(
                {
                    "width": width,
                    "mode": "backend-parity",
                    "status": "FAIL",
                    "detail": str(exc)[:200],
                }
            )
            event_log.append(
                {
                    "width": width,
                    "phase": "backend-parity",
                    "status": "FAIL",
                    "detail": str(exc)[:200],
                }
            )
            return False
        publish_release(
            candidate_dir,
            tensors=scaled,
            width=width,
            action_words_path=actions_path,
            identity=release_identity,
        )
        passed = True
        for eval_mode in ("normal", "hard"):
            certificate_path = repo_root / identity["certificates"][eval_mode]["path"]
            evidence_dir = run_dir / "evidence" / f"width-{width}"
            record = {
                "width": width,
                "mode": eval_mode,
                "certificate_sha256": corpus.certificate_sha256s[eval_mode],
            }
            try:
                result, census_identity = evaluate_certified(
                    release_dir=candidate_dir,
                    certificate_path=certificate_path,
                    frozen_root=frozen_root,
                )
                save_census_report(
                    evidence_dir / f"census-{eval_mode}.json",
                    result,
                    census_identity,
                )
                record["status"] = "PASS"
            except (
                EvaluationError,
                PolicyRuntimeError,
                ReleaseError,
                CertificateError,
                VerificationError,
                OSError,
            ) as exc:
                record["status"] = "FAIL"
                record["detail"] = str(exc)[:200]
                passed = False
            evaluations.append(record)
        if not passed:
            return False
        output_path = Path(output_release)
        if os.path.lexists(output_path):
            raise AdaptiveError(f"output release already exists: {output_path}")
        try:
            os.rename(candidate_dir, output_path)
        except OSError as exc:
            raise AdaptiveError(
                f"output release must be on the same filesystem as the run dir: {exc}"
            ) from exc
        final = load_release(output_path)
        backend_parity_report = backend_report
        release_info = {
            "path": str(output_path),
            "weights_sha256": final.weights_sha256,
            "manifest_sha256": final.manifest_sha256,
            "backend_parity": backend_report,
            "score_scale": float(adaptive["release_score_scale"]),
        }
        selected_width = width
        return True

    widths = adaptive["widths"]
    finished = False
    for width_index, width in enumerate(widths):
        is_last_width = width_index == len(widths) - 1
        trainer = TorchCloneTrainer(
            width,
            seed=identity["seed"],
            lambda_rank=0,
            learning_rate=float(adaptive["learning_rate"]),
        )
        attempt = {"width": width, "phases": [], "result": "incomplete"}
        width_attempts.append(attempt)
        fine_tuning_started = False

        def record_phase(
            phase_name: str,
            epochs: int,
            losses,
            frontier,
            predicted,
            *,
            stalls: int = 0,
            learning_rate: float,
            fine_tuning: bool,
            trainer=trainer,
            width=width,
            attempt=attempt,
        ):
            event = {
                "width": width,
                "phase": phase_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "fine_tuning": fine_tuning,
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "divergent_answers": sum(frontier.values()),
                "frontier_states": len(frontier),
                "all_node_disagreements": _all_node_disagreements(corpus, predicted),
                "consecutive_stalls": stalls,
                "weight_digest": weight_digest(),
            }
            best = attempt.get("best_divergent_answers")
            if best is None or event["divergent_answers"] < best:
                attempt["best_divergent_answers"] = event["divergent_answers"]
                attempt["best_all_node_disagreements"] = event[
                    "all_node_disagreements"
                ]
                attempt["best_phase"] = phase_name
            event_log.append(event)
            attempt["phases"].append(event)
            weights_np, opt_np = trainer.snapshot_payload()
            save_snapshot(
                run_dir
                / f"width-{width}"
                / "checkpoints"
                / f"epoch-{trainer.update_count:04d}",
                weights=weights_np,
                optimizer_arrays=opt_np,
                rng_state={"seed": identity["seed"]},
                sampler_position=0,
                update_count=trainer.update_count,
                config={
                    "config_sha256": config["config_sha256"],
                    "width": width,
                    "phase": phase_name,
                },
                backend_version="torch-" + torch.__version__,
                training_state={
                    "divergent_answers": event["divergent_answers"],
                    "learning_rate": event["learning_rate"],
                    "fine_tuning": event["fine_tuning"],
                    "loss_first": event["loss_first"],
                    "loss_last": event["loss_last"],
                    "weight_map_sha256": event["weight_digest"],
                },
            )
            save_weights_map()
            return event

        base_lr = float(adaptive["learning_rate"])
        losses = train_epochs(
            trainer,
            tensors,
            weight_vector(),
            adaptive["initial_epochs"],
            gradient_clip_norm=adaptive["gradient_clip_norm"],
        )
        predicted = predict_indices(trainer, tensors)
        frontier = first_disagreement_frontier(corpus, predicted)
        divergent = sum(frontier.values())
        if divergent <= adaptive["fine_tune_threshold"]:
            fine_tuning_started = True
        record_phase(
            "initial",
            adaptive["initial_epochs"],
            losses,
            frontier,
            predicted,
            learning_rate=base_lr,
            fine_tuning=False,
        )

        if divergent == 0:
            if evaluate_candidate(trainer, width, predicted, frontier):
                attempt["result"] = "parity"
                status = "PARITY"
                finished = True
                break
            attempt["result"] = "evaluation-failed"
            if is_last_width:
                break
            continue

        previous = divergent
        consecutive_stalls = 0
        promoted = False
        for dagger_round in range(1, adaptive["max_dagger_rounds"] + 1):
            for key, count in frontier.items():
                weights[key] += float(count)
            phase_lr, fine_tuning_started = phase_learning_rate(
                base_learning_rate=base_lr,
                fine_tune_learning_rate=float(
                    adaptive["fine_tune_learning_rate"]
                ),
                fine_tune_threshold=adaptive["fine_tune_threshold"],
                previous_divergent_answers=previous,
                fine_tuning_started=fine_tuning_started,
            )
            for group in trainer.optimizer.param_groups:
                group["lr"] = phase_lr
            losses = train_epochs(
                trainer,
                tensors,
                weight_vector(),
                adaptive["dagger_epochs"],
                gradient_clip_norm=adaptive["gradient_clip_norm"],
            )
            predicted = predict_indices(trainer, tensors)
            frontier = first_disagreement_frontier(corpus, predicted)
            divergent = sum(frontier.values())
            decision, consecutive_stalls = promotion_decision(
                divergent_answers=divergent,
                previous_divergent_answers=previous,
                consecutive_stalls=consecutive_stalls,
                stall_patience=adaptive["stall_patience"],
                dagger_round=dagger_round,
                max_dagger_rounds=adaptive["max_dagger_rounds"],
                is_last_width=is_last_width,
            )
            record_phase(
                f"dagger-{dagger_round}",
                adaptive["dagger_epochs"],
                losses,
                frontier,
                predicted,
                stalls=consecutive_stalls,
                learning_rate=phase_lr,
                fine_tuning=fine_tuning_started,
            )
            previous = divergent
            if decision == "candidate":
                if evaluate_candidate(trainer, width, predicted, frontier):
                    attempt["result"] = "parity"
                    status = "PARITY"
                    finished = True
                    break
                decision = "incomplete" if is_last_width else "promote"
                attempt["result"] = "evaluation-failed"
            if decision == "continue":
                continue
            if decision == "promote":
                attempt["result"] = "stalled-promoted"
                promoted = True
            break
        if finished:
            break
        if promoted:
            continue
        break

    receipt = {
        "schema": ADAPTIVE_SCHEMA,
        "status": status,
        "campaign_id": campaign_id,
        "config_sha256": config["config_sha256"],
        "policy_spec": TENSOR_SPEC_VERSION,
        "certificate_sha256s": corpus.certificate_sha256s,
        "ah_cells_accessed": False,
        "width_attempts": width_attempts,
        "selected_width": selected_width,
        "release": release_info,
        "evaluations": evaluations,
        "backend_parity": backend_parity_report,
        "score_scale": float(adaptive["release_score_scale"]),
    }
    _write_atomic(run_dir / "receipt.json", _canonical(receipt) + "\n")
    if status == "PARITY":
        _write_atomic(
            run_dir / "CLOSED",
            _canonical({"closed": "PARITY", "campaign_id": campaign_id}) + "\n",
        )
    event_log.close()
    return receipt
