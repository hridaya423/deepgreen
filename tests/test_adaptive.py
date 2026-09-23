"""Step 7 adaptive campaign: corpus extraction, frontier, promotion, publisher."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tests.benchmark import repo_path

REPO_ROOT = repo_path()
CONFIG = REPO_ROOT / "configs/train/adaptive.toml"
FROZEN = REPO_ROOT / "benchmarks/openwordle-v1/frozen"
ACTIONS_PATH = FROZEN / "corpus/words/accepted-guesses.txt"
CERT_DIR = REPO_ROOT / "artifacts/oracle/openwordle-v1"


@pytest.fixture(scope="module")
def corpus():
    from deep_green.adaptive import load_certified_tree_corpus

    return load_certified_tree_corpus(CONFIG)


@pytest.fixture(scope="module")
def action_words():
    return tuple(ACTIONS_PATH.read_text(encoding="utf-8").split())


def _release_identity():
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    return {
        "benchmark_id": "openwordle-v1",
        "rules_sha256": manifest["files"]["rules.json"],
        "prior_id": "uniform-v1",
        "action_space_sha256": manifest["files"][
            "corpus/words/accepted-guesses.txt"
        ],
        "track": "full-compiled",
        "inference_class": "numpy-f32-pure-neural",
    }


def test_certified_corpus_extraction(corpus, action_words):
    import tomllib

    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert corpus.action_words == action_words
    assert len(corpus.action_words) == 5022
    assert corpus.certificate_sha256s == {
        mode: config["identity"]["certificates"][mode]["sha256"]
        for mode in ("normal", "hard")
    }
    by_mode = {"normal": [], "hard": []}
    for example in corpus.examples:
        by_mode[example.mode].append(example)
    assert len(by_mode["normal"]) == 523
    assert len(by_mode["hard"]) == 518
    roots = {e.state_key: e for e in corpus.examples}
    assert roots[corpus.root_keys["normal"]].target_word == "stare"
    assert roots[corpus.root_keys["hard"]].target_word == "slate"
    assert roots[corpus.root_keys["normal"]].visits == 512
    state_keys = [e.state_key for e in corpus.examples]
    assert len(set(state_keys)) == len(state_keys)
    assert state_keys == sorted(state_keys)
    for example in corpus.examples:
        assert len(example.rows) <= 5
        assert len(example.legal_mask) == 5022
        assert example.legal_mask[example.target_index] is True
        assert action_words[example.target_index] == example.target_word
        assert example.visits >= 1
        for child in example.child_keys:
            assert child in roots
    field_names = {f.name for f in dataclasses.fields(type(corpus.examples[0]))}
    assert field_names == {
        "state_key",
        "mode",
        "rows",
        "target_word",
        "target_index",
        "legal_mask",
        "visits",
        "child_keys",
    }


def test_state_key_is_deterministic(corpus):
    from deep_green.adaptive import _state_key

    example = corpus.examples[0]
    recomputed = hashlib.sha256(
        json.dumps(
            {
                "mode": example.mode,
                "rows": [list(row) for row in example.rows],
                "turn": len(example.rows) + 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert example.state_key == _state_key(example.mode, example.rows)
    assert example.state_key == recomputed


def test_frontier_all_wrong_stops_at_roots(corpus):
    from deep_green.adaptive import first_disagreement_frontier

    predicted = [
        (e.target_index + 1) % len(corpus.action_words) for e in corpus.examples
    ]
    frontier = first_disagreement_frontier(corpus, predicted)
    assert set(frontier) == set(corpus.root_keys.values())
    assert sum(frontier.values()) == 1024


def test_frontier_all_correct_is_empty(corpus):
    from deep_green.adaptive import first_disagreement_frontier

    predicted = [e.target_index for e in corpus.examples]
    assert first_disagreement_frontier(corpus, predicted) == {}


def test_frontier_wrong_child_counts_exact(corpus):
    from deep_green.adaptive import first_disagreement_frontier

    by_key = {e.state_key: e for e in corpus.examples}
    root = by_key[corpus.root_keys["normal"]]
    child_key = root.child_keys[0]
    child = by_key[child_key]
    predicted = [e.target_index for e in corpus.examples]
    index = {e.state_key: i for i, e in enumerate(corpus.examples)}
    predicted[index[child_key]] = (child.target_index + 1) % len(corpus.action_words)
    frontier = first_disagreement_frontier(corpus, predicted)
    assert frontier == {child_key: child.visits}


def test_frontier_rejects_bad_predictions(corpus):
    from deep_green.adaptive import AdaptiveError, first_disagreement_frontier

    with pytest.raises(AdaptiveError):
        first_disagreement_frontier(corpus, [0] * (len(corpus.examples) - 1))
    with pytest.raises(AdaptiveError):
        first_disagreement_frontier(corpus, [True] * len(corpus.examples))
    with pytest.raises(AdaptiveError):
        first_disagreement_frontier(
            corpus, [len(corpus.action_words)] * len(corpus.examples)
        )


def test_encode_corpus_shapes_and_public_inputs(corpus):
    import torch

    from deep_green.adaptive import _encode_corpus

    tensors = _encode_corpus(corpus)
    count = len(corpus.examples)
    actions = len(corpus.action_words)
    assert set(tensors) == {
        "row_letters",
        "row_marks",
        "row_positions",
        "row_mask",
        "turn",
        "mode",
        "word_letters",
        "word_positions",
        "legal_mask",
        "targets",
    }
    assert tuple(tensors["row_letters"].shape) == (count, 5, 5)
    assert tuple(tensors["row_marks"].shape) == (count, 5, 5)
    assert tuple(tensors["row_positions"].shape) == (count, 5, 5)
    assert tuple(tensors["row_mask"].shape) == (count, 5)
    assert tensors["row_mask"].dtype == torch.bool
    assert tuple(tensors["legal_mask"].shape) == (count, actions)
    assert tensors["legal_mask"].dtype == torch.bool
    assert tuple(tensors["targets"].shape) == (count,)
    assert tensors["targets"].dtype == torch.long
    assert tuple(tensors["word_letters"].shape) == (actions, 5)
    assert tuple(tensors["word_positions"].shape) == (actions, 5)
    for index, example in enumerate(corpus.examples[:20]):
        assert int(tensors["row_mask"][index].sum()) == len(example.rows)
        assert float(tensors["turn"][index]) == len(example.rows) + 1
        assert int(tensors["targets"][index]) == example.target_index
        assert bool(tensors["legal_mask"][index][example.target_index])


def _toy_corpus(action_words):
    from deep_green.adaptive import CertifiedTreeCorpus, TreeExample
    from deep_green.rules import (
        ActionSpace,
        Board,
        Mode,
        legal_actions,
    )

    space = ActionSpace.parse(action_words)
    examples = []
    root_keys = {}
    for mode_name, target in (("normal", "stare"), ("hard", "slate")):
        mode = Mode(mode_name)
        board = Board.create(mode, (), 1)
        mask = tuple(legal_actions(board, space))
        key = hashlib.sha256(f"{mode_name}-root".encode()).hexdigest()
        root_keys[mode_name] = key
        examples.append(
            TreeExample(
                state_key=key,
                mode=mode_name,
                rows=(),
                target_word=target,
                target_index=space.words.index(target),
                legal_mask=mask,
                visits=4,
                child_keys=(),
            )
        )
    return CertifiedTreeCorpus(
        action_words=action_words,
        examples=tuple(examples),
        root_keys=root_keys,
        certificate_sha256s={"normal": "a" * 64, "hard": "b" * 64},
    )


def test_full_corpus_step_loss_decreases(action_words):
    import torch

    from deep_green.adaptive import _encode_corpus, train_epochs
    from deep_green.backends.torch_train import TorchCloneTrainer

    corpus = _toy_corpus(action_words)
    tensors = _encode_corpus(corpus)
    trainer = TorchCloneTrainer(64, seed=17, lambda_rank=0, learning_rate=1e-2)
    weights = torch.tensor([4.0, 4.0], dtype=torch.float32)
    losses = train_epochs(
        trainer, tensors, weights, 4, gradient_clip_norm=1.0
    )
    assert len(losses) == 4
    assert trainer.update_count == 4
    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    ("divergent", "previous", "stalls_in", "round_", "is_last", "expected"),
    [
        (0, 100, 2, 1, False, ("candidate", 0)),
        (100, 100, 0, 1, False, ("continue", 1)),
        (100, 100, 1, 2, False, ("continue", 2)),
        (100, 100, 2, 3, False, ("promote", 3)),
        (100, 100, 2, 3, True, ("incomplete", 3)),
        (120, 100, 2, 1, False, ("promote", 3)),
        (50, 100, 2, 3, False, ("continue", 0)),
        (50, 100, 0, 8, False, ("incomplete", 0)),
        (100, 100, 1, 8, False, ("incomplete", 2)),
    ],
)
def test_promotion_decision(
    divergent, previous, stalls_in, round_, is_last, expected
):
    from deep_green.adaptive import promotion_decision

    assert (
        promotion_decision(
            divergent_answers=divergent,
            previous_divergent_answers=previous,
            consecutive_stalls=stalls_in,
            stall_patience=3,
            dagger_round=round_,
            max_dagger_rounds=8,
            is_last_width=is_last,
        )
        == expected
    )


def test_publish_release_is_deterministic_and_strict(tmp_path, action_words):
    import numpy as np

    from deep_green.model_spec import expected_shapes
    from deep_green.release import (
        EXPECTED_FILES,
        ReleaseError,
        load_release,
        publish_release,
    )

    tensors = {
        name: np.full(shape, 0.01, dtype=np.float32)
        for name, shape in expected_shapes(64).items()
    }
    identity = _release_identity()
    first = tmp_path / "release-one"
    second = tmp_path / "release-two"
    payload1 = publish_release(
        first,
        tensors=tensors,
        width=64,
        action_words_path=ACTIONS_PATH,
        identity=identity,
    )
    payload2 = publish_release(
        second,
        tensors=tensors,
        width=64,
        action_words_path=ACTIONS_PATH,
        identity=identity,
    )
    for name in (*EXPECTED_FILES, "payload-manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert payload1.weights_sha256 == payload2.weights_sha256
    assert load_release(first).width == 64
    with pytest.raises(ReleaseError, match="already exists"):
        publish_release(
            first,
            tensors=tensors,
            width=64,
            action_words_path=ACTIONS_PATH,
            identity=identity,
        )


def _patched_campaign(monkeypatch, tmp_path, action_words, predicted_fn, eval_fn):

    import deep_green.backends.torch_train as torch_train_mod
    import deep_green.checkpoint as checkpoint_mod
    import deep_green.evaluate as evaluate_mod
    import deep_green.release as release_mod
    from deep_green import adaptive

    corpus = _toy_corpus(action_words)
    monkeypatch.setattr(
        adaptive, "load_certified_tree_corpus", lambda _path: corpus
    )
    monkeypatch.setattr(adaptive, "_encode_corpus", lambda _corpus: {})
    monkeypatch.setattr(adaptive, "predict_indices", predicted_fn)

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.update_count = 0
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])

        def snapshot_payload(self):
            return {}, {}

    def fake_train_epochs(trainer, tensors, weights, epochs, *, gradient_clip_norm):
        assert gradient_clip_norm > 0
        trainer.update_count += epochs
        return [2.0, 1.0]

    monkeypatch.setattr(adaptive, "train_epochs", fake_train_epochs)
    monkeypatch.setattr(torch_train_mod, "TorchCloneTrainer", FakeTrainer)

    def fake_save_snapshot(directory, **kwargs):
        from pathlib import Path

        Path(directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(checkpoint_mod, "save_snapshot", fake_save_snapshot)

    def fake_publish(destination, **kwargs):
        from pathlib import Path

        Path(destination).mkdir(parents=True)
        return SimpleNamespace(weights_sha256="w" * 64, manifest_sha256="m" * 64)

    monkeypatch.setattr(release_mod, "publish_release", fake_publish)
    monkeypatch.setattr(
        release_mod,
        "load_release",
        lambda _path: SimpleNamespace(
            weights_sha256="w" * 64, manifest_sha256="m" * 64
        ),
    )
    monkeypatch.setattr(evaluate_mod, "evaluate_certified", eval_fn)
    monkeypatch.setattr(
        adaptive, "scaled_release_tensors", lambda tensors, scale: tensors
    )

    def fake_backend_parity(corpus_arg, tensors, width, **kwargs):
        return {
            "states": len(corpus_arg.examples),
            "tolerance": 1e-5,
            "max_abs_delta_torch_numpy": 0.0,
            "max_abs_delta_torch_mlx": 0.0,
            "max_abs_delta_numpy_mlx": 0.0,
            "torch_disagreements": 0,
            "numpy_disagreements": 0,
            "mlx_disagreements": 0,
        }

    monkeypatch.setattr(
        adaptive, "verify_release_backend_parity", fake_backend_parity
    )

    def fake_save_report(path, result, identity):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(evaluate_mod, "save_census_report", fake_save_report)
    return corpus


def test_adaptive_parity_receipt(tmp_path, monkeypatch, action_words):
    from deep_green import adaptive

    def predict(trainer, tensors):
        corpus = adaptive.load_certified_tree_corpus(None)
        return [e.target_index for e in corpus.examples]

    def evaluate(*, release_dir, certificate_path, frozen_root):
        return SimpleNamespace(), {"benchmark_id": "openwordle-v1"}

    _patched_campaign(monkeypatch, tmp_path, action_words, predict, evaluate)
    output_release = tmp_path / "champion-release"
    receipt = adaptive.run_adaptive(
        CONFIG, runs_root=tmp_path / "runs", output_release=output_release
    )
    assert receipt["status"] == "PARITY"
    assert receipt["schema"] == "wordle.adaptive-campaign/v1"
    assert receipt["selected_width"] == 64
    assert receipt["ah_cells_accessed"] is False
    assert len(receipt["evaluations"]) == 2
    assert all(e["status"] == "PASS" for e in receipt["evaluations"])
    assert receipt["release"]["weights_sha256"] == "w" * 64
    assert receipt["backend_parity"]["states"] == len(
        adaptive.load_certified_tree_corpus(None).examples
    )
    assert receipt["backend_parity"]["mlx_disagreements"] == 0
    assert receipt["score_scale"] == 0.000244140625
    assert receipt["release"]["backend_parity"] == receipt["backend_parity"]
    assert receipt["release"]["score_scale"] == 0.000244140625
    assert output_release.is_dir()
    run_dir = tmp_path / "runs" / receipt["campaign_id"]
    assert (run_dir / "CLOSED").is_file()
    saved = json.loads((run_dir / "receipt.json").read_text())
    assert saved == receipt


def test_adaptive_incomplete_on_stall(tmp_path, monkeypatch, action_words):
    from deep_green import adaptive

    def predict(trainer, tensors):
        corpus = adaptive.load_certified_tree_corpus(None)
        return [
            (e.target_index + 1) % len(corpus.action_words)
            for e in corpus.examples
        ]

    def evaluate(*, release_dir, certificate_path, frozen_root):
        raise AssertionError("evaluator must not run without zero divergence")

    _patched_campaign(monkeypatch, tmp_path, action_words, predict, evaluate)
    receipt = adaptive.run_adaptive(
        CONFIG,
        runs_root=tmp_path / "runs",
        output_release=tmp_path / "champion-release",
    )
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["selected_width"] is None
    assert receipt["release"] is None
    assert receipt["ah_cells_accessed"] is False
    assert receipt["backend_parity"] is None
    assert receipt["score_scale"] == 0.000244140625
    run_dir = tmp_path / "runs" / receipt["campaign_id"]
    assert not (run_dir / "CLOSED").exists()
    assert (run_dir / "receipt.json").is_file()
    for attempt in receipt["width_attempts"][:-1]:
        phases = [p["phase"] for p in attempt["phases"]]
        assert phases == ["initial", "dagger-1", "dagger-2", "dagger-3"]
        assert [p["consecutive_stalls"] for p in attempt["phases"]] == [0, 1, 2, 3]
        assert attempt["phases"][0]["learning_rate"] == 3e-4
        assert attempt["phases"][0]["fine_tuning"] is False
        for phase in attempt["phases"][1:]:
            assert phase["learning_rate"] == 3e-5
            assert phase["fine_tuning"] is True
        assert attempt["best_phase"] == "initial"
        assert attempt["best_divergent_answers"] == 8
        assert attempt["result"] == "stalled-promoted"


def test_adaptive_never_parity_when_one_gate_fails(
    tmp_path, monkeypatch, action_words
):
    from deep_green import adaptive
    from deep_green.evaluate import EvaluationError

    def predict(trainer, tensors):
        corpus = adaptive.load_certified_tree_corpus(None)
        return [e.target_index for e in corpus.examples]

    calls = []

    def evaluate(*, release_dir, certificate_path, frozen_root):
        calls.append(certificate_path)
        if len(calls) % 2 == 0:
            raise EvaluationError("parity gate failed")
        return SimpleNamespace(), {"benchmark_id": "openwordle-v1"}

    _patched_campaign(monkeypatch, tmp_path, action_words, predict, evaluate)
    receipt = adaptive.run_adaptive(
        CONFIG,
        runs_root=tmp_path / "runs",
        output_release=tmp_path / "champion-release",
    )
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["release"] is None
    assert all(
        e["status"] == "FAIL" for e in receipt["evaluations"] if e["mode"] == "hard"
    )
    run_dir = tmp_path / "runs" / receipt["campaign_id"]
    assert not (run_dir / "CLOSED").exists()


def test_adaptive_config_strictness(tmp_path):
    from deep_green.adaptive import AdaptiveError, load_adaptive_config

    config = load_adaptive_config(CONFIG)
    assert config["adaptive"]["widths"][:1] == [64]
    assert config["adaptive"]["gradient_clip_norm"] == 1.0
    assert config["adaptive"]["stall_patience"] == 3
    assert len(config["config_sha256"]) == 64

    text = CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("widths = [64, 96", "widths = [96, 64"))
    with pytest.raises(AdaptiveError, match="widths"):
        load_adaptive_config(bad)
    bad.write_text(text.replace("[adaptive]", "[adaptive]\nextra = 1", 1))
    with pytest.raises(AdaptiveError, match="extra|incomplete"):
        load_adaptive_config(bad)
    bad.write_text(
        text.replace("gradient_clip_norm = 1.0", "gradient_clip_norm = -1.0")
    )
    with pytest.raises(AdaptiveError, match="gradient_clip_norm"):
        load_adaptive_config(bad)
    bad.write_text(text.replace("stall_patience = 3", "stall_patience = 999"))
    with pytest.raises(AdaptiveError, match="stall_patience"):
        load_adaptive_config(bad)
    bad.write_text(text.replace("stall_patience = 3", ""))
    with pytest.raises(AdaptiveError, match="extra|incomplete"):
        load_adaptive_config(bad)
    bad.write_text(text.replace("tier = 512", "tier = 2095", 1))
    with pytest.raises(AdaptiveError, match="shared tier"):
        load_adaptive_config(bad)


def test_adaptive_config_allows_registered_suffix(tmp_path):
    from deep_green.adaptive import load_adaptive_config

    config = tmp_path / "adaptive-2095.toml"
    config.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "widths = [64, 96, 128, 192, 256, 320, 384]",
            "widths = [128, 192, 256, 320, 384]",
        )
    )
    loaded = load_adaptive_config(config)
    assert loaded["adaptive"]["widths"] == [128, 192, 256, 320, 384]


def test_phase_learning_rate_threshold_and_sticky():
    from deep_green.adaptive import phase_learning_rate

    rate, started = phase_learning_rate(
        base_learning_rate=3e-4,
        fine_tune_learning_rate=3e-5,
        fine_tune_threshold=64,
        previous_divergent_answers=100,
        fine_tuning_started=False,
    )
    assert (rate, started) == (3e-4, False)
    rate, started = phase_learning_rate(
        base_learning_rate=3e-4,
        fine_tune_learning_rate=3e-5,
        fine_tune_threshold=64,
        previous_divergent_answers=64,
        fine_tuning_started=False,
    )
    assert (rate, started) == (3e-5, True)
    rate, started = phase_learning_rate(
        base_learning_rate=3e-4,
        fine_tune_learning_rate=3e-5,
        fine_tune_threshold=64,
        previous_divergent_answers=500,
        fine_tuning_started=True,
    )
    assert (rate, started) == (3e-5, True)


def test_adaptive_fine_tune_and_best_phase(tmp_path, monkeypatch, action_words):
    from deep_green import adaptive

    def predict(trainer, tensors):
        corpus = adaptive.load_certified_tree_corpus(None)
        return [e.target_index for e in corpus.examples]

    frontier_calls = iter([100, 60, 90, 90, 90])
    monkeypatch.setattr(
        adaptive,
        "first_disagreement_frontier",
        lambda corpus, predicted: {
            corpus.root_keys["normal"]: next(frontier_calls, 200)
        },
    )

    def evaluate(*, release_dir, certificate_path, frozen_root):
        raise AssertionError("evaluator must not run without zero divergence")

    _patched_campaign(monkeypatch, tmp_path, action_words, predict, evaluate)
    receipt = adaptive.run_adaptive(
        CONFIG,
        runs_root=tmp_path / "runs",
        output_release=tmp_path / "champion-release",
    )
    assert receipt["status"] == "INCOMPLETE"
    attempt64 = receipt["width_attempts"][0]
    assert attempt64["best_phase"] == "dagger-1"
    assert attempt64["best_divergent_answers"] == 60
    assert attempt64["result"] == "stalled-promoted"
    phases = attempt64["phases"]
    assert [p["phase"] for p in phases] == [
        "initial",
        "dagger-1",
        "dagger-2",
        "dagger-3",
        "dagger-4",
    ]
    assert [p["learning_rate"] for p in phases] == [
        3e-4,
        3e-4,
        3e-5,
        3e-5,
        3e-5,
    ]
    assert [p["fine_tuning"] for p in phases] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [p["consecutive_stalls"] for p in phases] == [0, 0, 1, 2, 3]


def test_scaled_release_tensors_exact_two_tensors():
    from deep_green import adaptive
    from deep_green.model_spec import expected_shapes

    rng = np.random.default_rng(7)
    tensors = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in expected_shapes(64).items()
    }
    scale = 0.000244140625
    scaled = adaptive.scaled_release_tensors(tensors, scale)
    assert set(scaled) == set(tensors)
    for name, original in tensors.items():
        assert scaled[name].dtype == np.float32
        assert not np.may_share_memory(scaled[name], original)
        if name in ("score.bilinear", "score.bias"):
            np.testing.assert_array_equal(
                scaled[name],
                np.asarray(original * np.float32(scale), dtype=np.float32),
            )
        else:
            np.testing.assert_array_equal(scaled[name], original)
    scaled["board.project"][0, 0] = 12345.0
    assert tensors["board.project"][0, 0] != 12345.0


def test_scaled_release_tensors_preserves_argmax(corpus):
    import torch

    from deep_green import adaptive
    from deep_green.backends.torch_policy import TorchWordlePolicy
    from deep_green.model_spec import expected_shapes

    rng = np.random.default_rng(11)
    np_tensors = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in expected_shapes(64).items()
    }
    policy = TorchWordlePolicy(64)
    policy.load_exact(
        {n: torch.from_numpy(v) for n, v in np_tensors.items()}
    )
    scaled = adaptive.scaled_release_tensors(np_tensors, 0.000244140625)
    scaled_policy = TorchWordlePolicy(64)
    scaled_policy.load_exact(
        {n: torch.from_numpy(v) for n, v in scaled.items()}
    )
    encoded = adaptive._encode_corpus(corpus)
    count = len(corpus.examples)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        kwargs = {
            k: (v[start:stop] if v.shape[0] == count else v)
            for k, v in encoded.items()
            if k not in ("legal_mask", "targets")
        }
        mask = encoded["legal_mask"][start:stop]
        base = policy.score_all(**kwargs).masked_fill(~mask, -torch.inf)
        fine = scaled_policy.score_all(**kwargs).masked_fill(~mask, -torch.inf)
        assert torch.equal(base.argmax(dim=1), fine.argmax(dim=1))


def test_scaled_release_tensors_rejects_bad_inputs():
    from deep_green import adaptive
    from deep_green.model_spec import expected_shapes

    tensors = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in expected_shapes(64).items()
    }
    for bad in (0.0, -1.0, 2.0, True, float("nan"), float("inf")):
        with pytest.raises(adaptive.AdaptiveError):
            adaptive.scaled_release_tensors(tensors, bad)
    del tensors["score.bias"]
    with pytest.raises(adaptive.AdaptiveError):
        adaptive.scaled_release_tensors(tensors, 0.5)


@pytest.fixture(scope="module")
def champion_release():
    from deep_green.release import load_release

    release_dir = REPO_ROOT / "artifacts/releases/champion-512"
    if not release_dir.is_dir():
        pytest.skip("preserved pre-final release artifact absent")
    return load_release(release_dir)


def test_backend_parity_rejects_unscaled_artifact(corpus, champion_release):
    from deep_green import adaptive

    with pytest.raises(adaptive.AdaptiveError, match="backend parity"):
        adaptive.verify_release_backend_parity(
            corpus, champion_release.tensors, champion_release.width
        )


def test_backend_parity_scaled_artifact(corpus, champion_release):
    from deep_green import adaptive

    scaled = adaptive.scaled_release_tensors(
        champion_release.tensors, 0.000244140625
    )
    report = adaptive.verify_release_backend_parity(
        corpus, scaled, champion_release.width
    )
    assert report["states"] == len(corpus.examples) == 1041
    assert report["tolerance"] == 1e-5
    for key in (
        "max_abs_delta_torch_numpy",
        "max_abs_delta_torch_mlx",
        "max_abs_delta_numpy_mlx",
    ):
        assert 0 <= report[key] <= 1e-5
    assert report["torch_disagreements"] == 0
    assert report["numpy_disagreements"] == 0
    assert report["mlx_disagreements"] == 0


def test_backend_gate_failure_blocks_publish(
    tmp_path, monkeypatch, action_words
):
    import deep_green.release as release_mod
    from deep_green import adaptive

    def predict(trainer, tensors):
        corpus = adaptive.load_certified_tree_corpus(None)
        return [e.target_index for e in corpus.examples]

    def evaluate(**kwargs):
        raise AssertionError("evaluator must not run when backend gate fails")

    _patched_campaign(monkeypatch, tmp_path, action_words, predict, evaluate)
    publish_calls = []

    def forbid_publish(destination, **kwargs):
        publish_calls.append(destination)
        raise AssertionError("publish must not be reached")

    monkeypatch.setattr(release_mod, "publish_release", forbid_publish)

    def failing_parity(*args, **kwargs):
        raise adaptive.AdaptiveError("backend parity exceeds 1e-5: {}")

    monkeypatch.setattr(
        adaptive, "verify_release_backend_parity", failing_parity
    )
    receipt = adaptive.run_adaptive(
        CONFIG,
        runs_root=tmp_path / "runs",
        output_release=tmp_path / "champion-release",
    )
    assert not publish_calls
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["release"] is None
    assert receipt["backend_parity"] is None
    assert any(
        e["mode"] == "backend-parity" and e["status"] == "FAIL"
        for e in receipt["evaluations"]
    )
    assert all(
        e["mode"] == "backend-parity" for e in receipt["evaluations"]
    )


def _campaign_lock_setup(monkeypatch, tmp_path, action_words):
    from deep_green import adaptive

    corpus = _toy_corpus(action_words)
    monkeypatch.setattr(
        adaptive, "load_certified_tree_corpus", lambda _path: corpus
    )
    config = adaptive.load_adaptive_config(CONFIG)
    campaign_id = adaptive._campaign_id(config, corpus)
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    return adaptive, campaign_id, runs_root


def test_campaign_lock_rejects_active_writer(
    tmp_path, monkeypatch, action_words
):
    adaptive, campaign_id, runs_root = _campaign_lock_setup(
        monkeypatch, tmp_path, action_words
    )
    lock_path = runs_root / f".campaign-{campaign_id}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(adaptive.AdaptiveError, match="active writer"):
            adaptive.run_adaptive(
                CONFIG,
                runs_root=runs_root,
                output_release=tmp_path / "out",
            )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert not (runs_root / campaign_id).exists()


def test_campaign_refuses_partial_run_dir(
    tmp_path, monkeypatch, action_words
):
    adaptive, campaign_id, runs_root = _campaign_lock_setup(
        monkeypatch, tmp_path, action_words
    )
    run_dir = runs_root / campaign_id
    run_dir.mkdir()
    marker = run_dir / "partial.txt"
    marker.write_bytes(b"partial")
    with pytest.raises(
        adaptive.AdaptiveError, match="already exists.*incomplete/partial"
    ):
        adaptive.run_adaptive(
            CONFIG, runs_root=runs_root, output_release=tmp_path / "out"
        )
    assert marker.read_bytes() == b"partial"
    assert sorted(p.name for p in run_dir.iterdir()) == ["partial.txt"]


def test_campaign_refuses_receipted_and_closed_run_dir(
    tmp_path, monkeypatch, action_words
):
    adaptive, campaign_id, runs_root = _campaign_lock_setup(
        monkeypatch, tmp_path, action_words
    )
    run_dir = runs_root / campaign_id
    run_dir.mkdir()
    (run_dir / "receipt.json").write_text('{"status":"INCOMPLETE"}')
    with pytest.raises(
        adaptive.AdaptiveError, match="already exists.*incomplete/partial"
    ):
        adaptive.run_adaptive(
            CONFIG, runs_root=runs_root, output_release=tmp_path / "out"
        )
    (run_dir / "CLOSED").touch()
    with pytest.raises(adaptive.AdaptiveError, match="already exists.*CLOSED"):
        adaptive.run_adaptive(
            CONFIG, runs_root=runs_root, output_release=tmp_path / "out"
        )
    assert json.loads((run_dir / "receipt.json").read_text())[
        "status"
    ] == "INCOMPLETE"


def test_campaign_lock_released_after_failure(
    tmp_path, monkeypatch, action_words
):
    adaptive, campaign_id, runs_root = _campaign_lock_setup(
        monkeypatch, tmp_path, action_words
    )
    monkeypatch.setattr(
        adaptive,
        "_encode_corpus",
        lambda _corpus: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        adaptive.run_adaptive(
            CONFIG, runs_root=runs_root, output_release=tmp_path / "out"
        )
    lock_path = runs_root / f".campaign-{campaign_id}.lock"
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
