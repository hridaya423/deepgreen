"""Training tests: CPU overfit, deterministic resume, closed runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.benchmark import repo_path
from deep_green.checkpoint import load_snapshot
from deep_green.examples import load_examples
from deep_green.train import run_clone

FIXTURE = repo_path("tests/fixtures/oracle_examples/tiny.jsonl")
WORDS = repo_path("benchmarks/openwordle-v1/frozen/corpus/words/accepted-guesses.txt")
REPO_ROOT = Path(__file__).resolve().parents[1]


class _Args:
    config = str(REPO_ROOT / "configs/train/clone.toml")
    examples = str(FIXTURE)
    model_width = 64
    device = "cpu"
    max_updates = 8
    runs_root = str(Path("/tmp/wordle-clone-test-runs"))
    action_words = str(WORDS)
    seed = 17
    resume = None


@pytest.fixture(scope="module")
def examples():
    return load_examples(FIXTURE)


def test_cpu_overfit_eight_updates(tmp_path):
    args = _Args()
    args.runs_root = str(tmp_path / "runs")
    result = run_clone(Path(args.config), args)
    inventory = result["inventory"]
    assert inventory["updates"] == 8
    assert inventory["losses_first_last"][1] < inventory["losses_first_last"][0]
    assert inventory["final_argmax_optimal"], (
        "a final selected action left the optimal set"
    )
    records = load_examples(FIXTURE)
    for record in records:
        assert inventory["final_actions"][record["example_id"]] in record[
            "optimal_tie_set"
        ]
    events = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 8
    assert all(event["batch_examples"] == len(records) for event in events)


def _clone_child(runs_root: str, resume: str | None = None) -> str:
    resume_literal = repr(resume) if resume else "None"
    return f"""
from pathlib import Path
from deep_green.train import run_clone

class _Args:
    config = {str(REPO_ROOT / "configs/train/clone.toml")!r}
    examples = {str(FIXTURE)!r}
    model_width = 64
    device = "cpu"
    max_updates = 8
    runs_root = {runs_root!r}
    action_words = {str(WORDS)!r}
    seed = 17
    resume = {resume_literal}

run_clone(Path(_Args.config), _Args())
"""


@pytest.fixture(scope="module")
def uninterrupted_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("uninterrupted")
    args = _Args()
    args.runs_root = str(root)
    return run_clone(Path(args.config), args)


@pytest.mark.parametrize(
    "kill_generation,kill_boundary,expected_latest",
    [
        ("update-0004", "generation-renamed", "update-0003"),
        ("update-0008", "generation-renamed", "update-0007"),
        ("update-0008", "latest-renamed", "update-0008"),
    ],
)
def test_interrupted_run_resumes_byte_identically(
    tmp_path, uninterrupted_run, kill_generation, kill_boundary, expected_latest
):
    from deep_green.checkpoint import latest_snapshot_path, snapshot_digest

    full_run_dir = Path(uninterrupted_run["run_dir"])
    full_generation = full_run_dir / "checkpoints" / "update-0008"

    interrupted_root = tmp_path / "interrupted"
    env = dict(os.environ)
    env["WORDLE_CHECKPOINT_KILL_GENERATION"] = kill_generation
    env["WORDLE_CHECKPOINT_KILL_BOUNDARY"] = kill_boundary
    killed = subprocess.run(
        [sys.executable, "-c", _clone_child(str(interrupted_root))],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    run_dirs = [
        p
        for p in interrupted_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert not (run_dir / "CLOSED").exists()
    checkpoint_root = run_dir / "checkpoints"
    assert latest_snapshot_path(checkpoint_root).name == expected_latest

    resumed = subprocess.run(
        [
            sys.executable,
            "-c",
            _clone_child(str(interrupted_root), resume=str(checkpoint_root)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr

    resumed_generation = checkpoint_root / "update-0008"
    for name in ("weights.safetensors", "optimizer.safetensors", "state.json", "COMPLETE"):
        assert (resumed_generation / name).read_bytes() == (
            full_generation / name
        ).read_bytes(), name
    resumed_snapshot = load_snapshot(resumed_generation)
    full_snapshot = load_snapshot(full_generation)
    assert snapshot_digest(resumed_snapshot) == snapshot_digest(full_snapshot)
    resumed_state = resumed_snapshot["state"]
    full_state = full_snapshot["state"]
    assert resumed_state["sampler_position"] == full_state["sampler_position"]
    assert resumed_state["update_count"] == full_state["update_count"] == 8
    assert resumed_state["rng_state"] == full_state["rng_state"]
    assert resumed_state["config"] == full_state["config"]
    assert resumed_state["backend_version"] == full_state["backend_version"]
    assert resumed_state["training_state"] == full_state["training_state"]

    for name in ("manifest.json", "events.jsonl", "run-inventory.json", "CLOSED"):
        assert (run_dir / name).read_bytes() == (full_run_dir / name).read_bytes(), name
    assert (checkpoint_root / "LATEST").read_bytes() == (
        full_run_dir / "checkpoints" / "LATEST"
    ).read_bytes()
    inventory = json.loads((run_dir / "run-inventory.json").read_text())
    assert inventory["final_checkpoint"] == "checkpoints/update-0008"


def test_resume_rejects_closed_run(tmp_path):
    from deep_green.train import TrainError

    args = _Args()
    args.runs_root = str(tmp_path / "runs")
    result = run_clone(Path(args.config), args)
    resume_args = _Args()
    resume_args.runs_root = args.runs_root
    resume_args.resume = str(Path(result["run_dir"]) / "checkpoints")
    with pytest.raises(TrainError, match="CLOSED"):
        run_clone(Path(resume_args.config), resume_args)


def test_resume_rejects_checkpoint_config_drift(tmp_path):
    import numpy as np
    import torch

    from deep_green.checkpoint import save_snapshot
    from deep_green.train import TrainError

    generation = tmp_path / "checkpoints" / "update-0099"
    save_snapshot(
        generation,
        weights={"w": np.zeros((4, 4), dtype=np.float32)},
        optimizer_arrays={},
        rng_state={"draws": 0},
        sampler_position=0,
        update_count=4,
        config={"lambda_rank": 0.25, "width": 128},
        backend_version="torch-" + torch.__version__,
    )
    resume_args = _Args()
    resume_args.runs_root = str(tmp_path / "runs")
    resume_args.resume = str(generation)
    with pytest.raises(TrainError, match="config"):
        run_clone(Path(resume_args.config), resume_args)


def test_resume_rejects_backend_drift(tmp_path, uninterrupted_run):
    import numpy as np

    from deep_green.checkpoint import load_snapshot, save_snapshot
    from deep_green.train import TrainError

    source = load_snapshot(
        Path(uninterrupted_run["run_dir"]) / "checkpoints" / "update-0008"
    )
    source_state = source["state"]
    generation = tmp_path / "checkpoints" / "update-0099"
    save_snapshot(
        generation,
        weights={"w": np.zeros((4, 4), dtype=np.float32)},
        optimizer_arrays={},
        rng_state=source_state["rng_state"],
        sampler_position=0,
        update_count=4,
        config=source_state["config"],
        backend_version="torch-0.0.0-foreign",
        training_state=source_state["training_state"],
    )
    resume_args = _Args()
    resume_args.runs_root = str(tmp_path / "runs")
    resume_args.resume = str(generation)
    with pytest.raises(TrainError, match="backend"):
        run_clone(Path(resume_args.config), resume_args)


@pytest.mark.parametrize("tamper", ["missing", "shape"])
def test_resume_rejects_incomplete_optimizer_state(
    tmp_path, uninterrupted_run, tamper
):
    import numpy as np
    import torch

    from deep_green.checkpoint import load_snapshot, save_snapshot
    from deep_green.train import TrainError

    generation = Path(uninterrupted_run["run_dir"]) / "checkpoints" / "update-0008"
    snapshot = load_snapshot(generation)
    optimizer = dict(snapshot["optimizer_arrays"])
    if tamper == "missing":
        del optimizer[min(optimizer)]
    else:
        key = min(k for k in optimizer if k.endswith("exp_avg"))
        optimizer[key] = np.zeros((2, 2), dtype=np.float32)
    state = snapshot["state"]
    tampered = tmp_path / "checkpoints" / "update-0099"
    save_snapshot(
        tampered,
        weights=snapshot["weights"],
        optimizer_arrays=optimizer,
        rng_state=state["rng_state"],
        sampler_position=state["sampler_position"],
        update_count=state["update_count"],
        config=state["config"],
        backend_version="torch-" + torch.__version__,
        training_state=state["training_state"],
    )
    resume_args = _Args()
    resume_args.runs_root = str(tmp_path / "runs")
    resume_args.resume = str(tampered)
    with pytest.raises(TrainError, match="invalid resume tensors"):
        run_clone(Path(resume_args.config), resume_args)


def test_restore_optimizer_arrays_validates_before_mutating():
    import numpy as np
    import torch

    from deep_green.backends.torch_train import TorchCloneTrainer

    trainer = TorchCloneTrainer(64, seed=17)
    trainer.train_example(
        row_letters=torch.zeros(1, 1, 5, dtype=torch.long),
        row_marks=torch.zeros(1, 1, 5, dtype=torch.long),
        row_positions=torch.broadcast_to(torch.arange(5), (1, 1, 5)).contiguous(),
        row_mask=torch.ones(1, 1, dtype=torch.bool),
        turn=torch.tensor([0.0]),
        mode=torch.tensor([[1.0, 0.0]]),
        word_letters=torch.zeros(4, 5, dtype=torch.long),
        word_positions=torch.broadcast_to(torch.arange(5), (4, 5)).contiguous(),
        legal_mask=torch.ones(1, 4, dtype=torch.bool),
        tie_indices=torch.tensor([0]),
        negative_indices=torch.tensor([1, 2]),
    )
    _, base = trainer.snapshot_payload()
    moment_key = min(k for k in base if k.endswith("exp_avg"))

    missing = dict(base)
    del missing[min(missing)]
    extra = dict(base)
    extra["p999.step"] = np.zeros(1, dtype=np.float32)
    wrong_shape = dict(base)
    wrong_shape[moment_key] = np.zeros((2, 2), dtype=np.float32)
    wrong_dtype = dict(base)
    wrong_dtype[moment_key] = base[moment_key].astype(np.float64)
    nonfinite = dict(base)
    poisoned = base[moment_key].copy()
    poisoned.flat[0] = np.inf
    nonfinite[moment_key] = poisoned

    for bad in (missing, extra, wrong_shape, wrong_dtype, nonfinite):
        fresh = TorchCloneTrainer(64, seed=17)
        with pytest.raises(ValueError):
            fresh.restore_optimizer_arrays(bad)
        assert not fresh.optimizer.state

    fresh = TorchCloneTrainer(64, seed=17)
    fresh.restore_optimizer_arrays(dict(base))
    assert fresh.optimizer.state


def test_closed_run_writes_inventory_and_marker(tmp_path):
    args = _Args()
    args.runs_root = str(tmp_path / "runs")
    result = run_clone(Path(args.config), args)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "CLOSED").is_file()
    inventory = json.loads((run_dir / "run-inventory.json").read_text())
    assert inventory["updates"] == 8
    assert inventory["final_argmax_optimal"] is True


def test_resume_rejects_identity_drift(tmp_path):
    args = _Args()
    args.runs_root = str(tmp_path / "runs-a")
    first = run_clone(Path(args.config), args)

    from deep_green.tracking import RunSpec, RunStore, RunStoreError

    manifest_path = Path(first["run_dir"]) / "manifest.json"
    stored = json.loads(manifest_path.read_text())
    tampered_spec = dict(stored["spec"])
    tampered_spec["seed"] = 999
    manifest_path.write_text(
        json.dumps({"spec": tampered_spec, "run_id": stored["run_id"]}, sort_keys=True) + "\n"
    )
    spec = RunSpec.from_dict(stored["spec"])
    with pytest.raises(RunStoreError, match="identity drift"):
        RunStore(Path(tmp_path / "runs-a"), spec)


def test_config_certificate_hash_must_match_verified_artifact(tmp_path):
    from deep_green.train import TrainError

    config = tmp_path / "clone.toml"
    text = Path(_Args.config).read_text()
    text = text.replace(
        "ab763d900306a83e9407885d210186db6239778054b1a5290f7f0d8b84abccfe",
        "0" * 64,
    )
    config.write_text(text)
    args = _Args()
    args.runs_root = str(tmp_path / "runs")
    with pytest.raises(TrainError, match="certificate sha256 mismatch"):
        run_clone(config, args)


def test_example_certificate_identity_must_match_config(tmp_path):
    from deep_green.examples import canonical_example_bytes, example_id_for
    from deep_green.train import TrainError

    records = load_examples(FIXTURE)
    records[0]["identity"]["certificate_sha256"] = "0" * 64
    records[0]["example_id"] = example_id_for(records[0])
    path = tmp_path / "examples.jsonl"
    path.write_bytes(b"".join(canonical_example_bytes(record) + b"\n" for record in records))
    args = _Args()
    args.examples = str(path)
    args.runs_root = str(tmp_path / "runs")
    with pytest.raises(TrainError, match="example identity"):
        run_clone(Path(args.config), args)


def test_teacher_labels_must_equal_certificate_proof(tmp_path):
    from deep_green.examples import canonical_example_bytes, example_id_for
    from deep_green.train import TrainError

    records = load_examples(FIXTURE)
    hard = next(record for record in records if record["public_state"]["mode"] == "hard")
    hard["optimal_tie_set"] = ["stare"]
    hard["example_id"] = example_id_for(hard)
    path = tmp_path / "examples.jsonl"
    path.write_bytes(b"".join(canonical_example_bytes(record) + b"\n" for record in records))
    args = _Args()
    args.examples = str(path)
    args.runs_root = str(tmp_path / "runs")
    with pytest.raises(TrainError, match="tie set is not certificate-derived"):
        run_clone(Path(args.config), args)


def test_refuses_unproven_answer_fold_claim(tmp_path):
    bad_examples = tmp_path / "bad.jsonl"
    lines = FIXTURE.read_text().splitlines()
    record = json.loads(lines[0])
    record["identity"]["answer_fold"] = "train"
    bad_examples.write_text(json.dumps(record, sort_keys=True) + "\n")

    args = _Args()
    args.examples = str(bad_examples)
    args.runs_root = str(tmp_path / "runs-bad")
    from deep_green.examples import ExampleError

    with pytest.raises(ExampleError, match="identity block"):
        run_clone(Path(args.config), args)
