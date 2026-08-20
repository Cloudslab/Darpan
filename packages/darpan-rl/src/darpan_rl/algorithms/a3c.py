from __future__ import annotations

import torch
from torch.distributions import Categorical

from ..types import Trajectory
from .a2c import A2C


class A3C(A2C):
    """A3C learner supporting local-worker gradient computation."""

    def compute_gradients(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[torch.Tensor | None], dict[str, float]]:
        batch = self._batch(trajectories)
        if batch is None:
            return [], {}
        obs, mask, action, _, ret, adv, weight, _ = batch
        self.model.train()
        logits, values = self.model(obs, mask)
        distribution = Categorical(logits=logits)
        wsum = weight.sum().clamp_min(1e-8)
        policy_loss = -(distribution.log_prob(action) * adv.detach() * weight).sum() / wsum
        value_loss = (((values - ret) ** 2) * weight).sum() / wsum
        entropy = (distribution.entropy() * weight).sum() / wsum
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        gradients = [
            None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for parameter in self.model.parameters()
        ]
        return gradients, {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "gradient_norm": float(gradient_norm.item()),
        }

    def apply_gradients(self, gradients: list[torch.Tensor | None]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(self.model.parameters(), gradients, strict=True):
            if gradient is not None:
                parameter.grad = gradient.to(self.device)
        self.optimizer.step()
        self.policy_version += 1

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        gradients, metrics = self.compute_gradients(trajectories)
        if not gradients:
            return metrics
        self.apply_gradients(gradients)
        self.episodes += len(trajectories)
        metrics["policy_version"] = float(self.policy_version)
        return metrics
