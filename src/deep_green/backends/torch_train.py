"""PyTorch training backend: exact clone updates with serializable state."""

from __future__ import annotations

import torch

from deep_green.backends.torch_policy import TorchWordlePolicy
from deep_green.losses import combined_loss
from deep_green.model_spec import tensor_names


class TorchCloneTrainer:
    """Deterministic CPU trainer over the frozen DeepSets policy."""

    def __init__(
        self,
        width: int,
        *,
        seed: int,
        lambda_rank: float = 0.25,
        learning_rate: float = 1e-2,
    ) -> None:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.policy = TorchWordlePolicy(width)
        with torch.no_grad():
            for parameter in self.policy.parameters():
                parameter.uniform_(-0.08, 0.08)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.lambda_rank = lambda_rank
        self.update_count = 0
        self.rng_state = {"python_seed": seed, "draws": 0}

    def train_example(
        self,
        *,
        row_letters,
        row_marks,
        row_positions,
        row_mask,
        turn,
        mode,
        word_letters,
        word_positions,
        legal_mask,
        tie_indices,
        negative_indices,
    ) -> dict:
        return self.train_batch(
            [
                {
                    "row_letters": row_letters,
                    "row_marks": row_marks,
                    "row_positions": row_positions,
                    "row_mask": row_mask,
                    "turn": turn,
                    "mode": mode,
                    "word_letters": word_letters,
                    "word_positions": word_positions,
                    "legal_mask": legal_mask,
                    "tie_indices": tie_indices,
                    "negative_indices": negative_indices,
                }
            ]
        )

    def train_batch(self, batches: list[dict]) -> dict:
        if not batches:
            raise ValueError("a full-batch update requires at least one example")
        totals = []
        breakdowns = []
        self.optimizer.zero_grad()
        for batch in batches:
            logits = self.policy.score_all(
                row_letters=batch["row_letters"],
                row_marks=batch["row_marks"],
                row_positions=batch["row_positions"],
                row_mask=batch["row_mask"],
                turn=batch["turn"],
                mode=batch["mode"],
                word_letters=batch["word_letters"],
                word_positions=batch["word_positions"],
            )
            total, breakdown = combined_loss(
                logits=logits,
                legal_mask=batch["legal_mask"],
                tie_indices=batch["tie_indices"],
                negative_indices=batch["negative_indices"],
                lambda_rank=self.lambda_rank,
            )
            totals.append(total)
            breakdowns.append(breakdown)
        mean_total = torch.stack(totals).mean()
        mean_total.backward()
        self.optimizer.step()
        self.update_count += 1
        count = len(breakdowns)
        return {
            "update": self.update_count,
            "batch_examples": count,
            "loss_total": float(mean_total.detach()),
            "loss_bc": sum(item.bc for item in breakdowns) / count,
            "loss_rank": sum(item.rank for item in breakdowns) / count,
            "tie_probability": sum(item.tie_probability for item in breakdowns) / count,
            "argmax_optimal": all(item.argmax_legal_optimal for item in breakdowns),
        }

    def weights(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.policy.tensors.items()}

    def optimizer_arrays(self) -> dict[str, torch.Tensor]:
        arrays: dict[str, torch.Tensor] = {}
        for parameter_index, parameter in enumerate(self.policy.parameters()):
            state = self.optimizer.state.get(parameter)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                arrays[f"p{parameter_index}.{key}"] = state[key].detach().clone()
            step_value = state["step"]
            arrays[f"p{parameter_index}.step"] = torch.tensor(
                [float(step_value)], dtype=torch.float32
            )
        return arrays

    def restore_optimizer_arrays(self, arrays: dict) -> None:
        parameters = list(self.policy.parameters())
        expected = {
            f"p{index}.{key}"
            for index in range(len(parameters))
            for key in ("exp_avg", "exp_avg_sq", "step")
        }
        missing = sorted(expected - set(arrays))
        extra = sorted(set(arrays) - expected)
        if missing or extra:
            raise ValueError(
                f"optimizer arrays mismatch: missing {missing} extra {extra}"
            )
        staged = []
        for name in sorted(arrays):
            parameter_index, key = name.split(".", 1)
            parameter = parameters[int(parameter_index[1:])]
            stored = arrays[name]
            value = (
                torch.from_numpy(stored.copy())
                if not isinstance(stored, torch.Tensor)
                else stored.clone()
            )
            if value.dtype != torch.float32:
                raise ValueError(f"optimizer array {name} is not float32")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"optimizer array {name} is not finite")
            expected_shape = (1,) if key == "step" else tuple(parameter.shape)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"optimizer array {name} has shape {tuple(value.shape)}, "
                    f"expected {expected_shape}"
                )
            staged.append((parameter, key, value))
        for parameter, key, value in staged:
            state = self.optimizer.state.setdefault(
                parameter,
                {
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                    "step": torch.zeros((), dtype=torch.float32),
                },
            )
            if key == "step":
                state[key] = value[0].clone()
            else:
                state[key] = value

    def snapshot_payload(self):
        from deep_green.model_spec import parameter_count

        assert sum(p.numel() for p in self.policy.parameters()) == parameter_count(
            self.policy.width
        )
        return (
            {name: t.numpy() for name, t in self.weights().items()},
            {name: t.numpy() for name, t in self.optimizer_arrays().items()},
        )

    def select_action(self, logits_row: torch.Tensor, legal_mask: torch.Tensor) -> int:
        masked = logits_row.masked_fill(~legal_mask.bool(), float("-inf"))
        return int(torch.argmax(masked))


def tensor_name_check() -> tuple[str, ...]:
    return tensor_names()
