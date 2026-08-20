from __future__ import annotations

import torch
from torch.distributions import Categorical

from ..types import Trajectory
from .actor_critic import ActorCriticAlgorithm


class A2C(ActorCriticAlgorithm):
    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        batch = self._batch(trajectories)
        if batch is None:
            return {}
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.policy_version += 1
        self.episodes += len(trajectories)
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "policy_version": float(self.policy_version),
        }
