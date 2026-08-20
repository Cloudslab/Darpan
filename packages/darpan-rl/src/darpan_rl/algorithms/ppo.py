from __future__ import annotations

import torch
from torch.distributions import Categorical

from ..types import Trajectory
from .actor_critic import ActorCriticAlgorithm


class PPO(ActorCriticAlgorithm):
    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        batch = self._batch(trajectories)
        if batch is None:
            return {}
        obs, mask, action, old_log, ret, adv, weight, _ = batch
        total_loss = policy_loss = value_loss = entropy_value = 0.0
        count = 0
        size = obs.shape[0]
        minibatch = max(1, min(self.config.minibatch_size, size))
        self.model.train()
        for _ in range(self.config.update_epochs):
            order = torch.randperm(size, device=self.device)
            for start in range(0, size, minibatch):
                index = order[start : start + minibatch]
                logits, values = self.model(obs[index], mask[index])
                distribution = Categorical(logits=logits)
                log_prob = distribution.log_prob(action[index])
                ratio = torch.exp(log_prob - old_log[index])
                unclipped = ratio * adv[index]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * adv[index]
                w = weight[index]
                p_loss = -(torch.minimum(unclipped, clipped) * w).sum() / w.sum().clamp_min(1e-8)
                v_loss = (((values - ret[index]) ** 2) * w).sum() / w.sum().clamp_min(1e-8)
                entropy = (distribution.entropy() * w).sum() / w.sum().clamp_min(1e-8)
                loss = p_loss + self.config.value_coef * v_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.item())
                policy_loss += float(p_loss.item())
                value_loss += float(v_loss.item())
                entropy_value += float(entropy.item())
                count += 1
        self.policy_version += 1
        self.episodes += len(trajectories)
        denom = max(1, count)
        return {
            "loss": total_loss / denom,
            "policy_loss": policy_loss / denom,
            "value_loss": value_loss / denom,
            "entropy": entropy_value / denom,
            "policy_version": float(self.policy_version),
        }
