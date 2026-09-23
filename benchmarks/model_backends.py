"""Backend parity benchmark: score the real action vocabulary on both stacks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np


def _load_action_words() -> list[str]:
    return (
        (REPO_ROOT / "benchmarks/openwordle-v1/frozen/corpus/words/accepted-guesses.txt")
        .read_text(encoding="utf-8")
        .split()
    )


def _random_boards(rng: np.random.Generator, rows_count: int, samples: int):
    letter_codes = {chr(ord("a") + i): i for i in range(26)}
    words = _load_action_words()
    rl = np.zeros((samples, rows_count, 5), dtype=np.int64)
    rm = np.zeros((samples, rows_count, 5), dtype=np.int64)
    rp = np.broadcast_to(np.arange(5), (samples, rows_count, 5)).copy()
    for b in range(samples):
        for r in range(rows_count):
            word = words[rng.integers(len(words))]
            for slot in range(5):
                rl[b, r, slot] = letter_codes[word[slot]]
            marks = rng.integers(0, 3, size=5)
            rm[b, r] = marks
    turn = np.full((samples,), rows_count + 1, dtype=np.float32)
    mode = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (samples, 1))
    wl = np.array([[letter_codes[c] for c in w] for w in words], dtype=np.int64)
    wp = np.broadcast_to(np.arange(5), (len(words), 5)).copy()
    return rl, rm, rp, turn, mode, wl, wp


def run(backend: str, batch_size: int, action_count: int, samples: int) -> float:
    rng = np.random.default_rng(20260825)
    if backend == "torch":
        import torch

        from deep_green.backends.torch_policy import TorchWordlePolicy
        from deep_green.model_spec import parameter_count

        policy = TorchWordlePolicy(64)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.uniform_(-0.08, 0.08)
        assert sum(p.numel() for p in policy.parameters()) == parameter_count(64)
        torch.set_grad_enabled(False)

    rng_state = np.random.default_rng(97)
    warmup_rl, warmup_rm, warmup_rp, _, _, wl, wp = _random_boards(rng_state, 1, 1)
    if backend == "torch":
        policy.score_all(
            row_letters=torch.tensor(warmup_rl), row_marks=torch.tensor(warmup_rm),
            row_positions=torch.tensor(warmup_rp),
            row_mask=torch.ones((1, 1), dtype=torch.bool),
            turn=torch.full((1,), 2.0),
            mode=torch.tensor(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (1, 1))),
            word_letters=torch.tensor(wl[:action_count]),
            word_positions=torch.tensor(wp[:action_count]),
        )
    else:
        import mlx.core as mx

        from deep_green.backends.mlx_policy import MlxWordlePolicy
        from deep_green.model_spec import parameter_count

        policy = MlxWordlePolicy(64)
        mx.random.seed(17)
        assert policy.parameter_report()["stored_scalars"] == parameter_count(64)

    total_delta_max = 0.0
    started = time.perf_counter()
    processed = 0
    while processed < samples:
        take = min(batch_size, samples - processed)
        rl, rm, rp, _turn, mode, wl, wp = _random_boards(rng, 1 + int(processed % 3), take)
        if backend == "torch":
            logits = policy.score_all(
                row_letters=torch.tensor(rl[:, :1]),
                row_marks=torch.tensor(rm[:, :1]),
                row_positions=torch.tensor(rp[:, :1]),
                row_mask=torch.ones((take, 1), dtype=torch.bool),
                turn=torch.full((take,), 2.0),
                mode=torch.tensor(mode),
                word_letters=torch.tensor(wl[:action_count]),
                word_positions=torch.tensor(wp[:action_count]),
            ).numpy()
        else:
            logits = np.asarray(
                policy(
                    row_letters=mx.array(rl[:, :1]),
                    row_marks=mx.array(rm[:, :1]),
                    row_positions=mx.array(rp[:, :1]),
                    row_mask=mx.array(np.ones((take, 1), dtype=bool)),
                    turn=mx.array(np.full((take,), 2.0, dtype=np.float32)),
                    mode=mx.array(mode),
                    word_letters=mx.array(wl[:action_count]),
                    word_positions=mx.array(wp[:action_count]),
                )
            )
        assert logits.shape == (take, action_count)
        assert np.isfinite(logits).all()
        total_delta_max = max(total_delta_max, float(np.abs(logits).max()))
        processed += take
    elapsed = time.perf_counter() - started
    print(
        f"backend={backend} width=64 actions={action_count} samples={processed} "
        f"max|logit|={total_delta_max:.4f} elapsed={elapsed:.2f}s OK"
    )
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-count", type=int, default=5022)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    run(args.backend, args.batch_size, args.action_count, args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
