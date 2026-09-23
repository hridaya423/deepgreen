"""MLX training backend mirroring torch_train for accelerator parity."""

from __future__ import annotations

import mlx.core as mx

from deep_green.backends.mlx_policy import MlxWordlePolicy


class MlxCloneTrainer:
    """Mirror of TorchCloneTrainer using mx.eval and value-and-grad."""

    def __init__(self, width: int, *, seed: int, lambda_rank: float = 0.25) -> None:
        self.policy = MlxWordlePolicy(width)
        self.lambda_rank = lambda_rank
        self.update_count = 0
        del seed

    def loss_value_and_grad(self, *, logits, legal_mask, tie_indices, negative_indices):
        masked = mx.where(legal_mask.astype(mx.bool_), logits, mx.array(-mx.inf))
        log_probs = mx.log_softmax(masked, axis=-1)
        bc = -mx.mean(log_probs[0, tie_indices])
        if negative_indices.size > 0:
            target = (
                mx.logsumexp(logits[0, tie_indices]) - mx.log(float(tie_indices.size))
            )
            penalties = mx.logaddexp(
                mx.array(0.0), 1.0 - target + logits[0, negative_indices]
            )
            rank = mx.mean(penalties)
        else:
            rank = mx.array(0.0)
        return bc + self.lambda_rank * rank

    def train_step(self, **batch) -> float:
        def objective(parameters):
            saved = {k: getattr(self.policy, k) for k in parameters}
            for k, v in parameters.items():
                setattr(self.policy, k, v)
            try:
                logits = self.policy(
                    row_letters=batch["row_letters"],
                    row_marks=batch["row_marks"],
                    row_positions=batch["row_positions"],
                    row_mask=batch["row_mask"],
                    turn=batch["turn"],
                    mode=batch["mode"],
                    word_letters=batch["word_letters"],
                    word_positions=batch["word_positions"],
                )
                total = self.loss_value_and_grad(
                    logits=logits,
                    legal_mask=batch["legal_mask"],
                    tie_indices=batch["tie_indices"],
                    negative_indices=batch["negative_indices"],
                )
            finally:
                for k, v in saved.items():
                    setattr(self.policy, k, v)
            return total

        value, gradients = mx.value_and_grad(objective)(self.policy.parameters_dict())
        updates = {}
        for name, gradient in gradients.items():
            current = getattr(self.policy, name)
            setattr(self.policy, name, current - 1e-2 * gradient)
            updates[name] = True
        mx.eval(self.policy.parameters())
        self.update_count += 1
        return float(value.item())
