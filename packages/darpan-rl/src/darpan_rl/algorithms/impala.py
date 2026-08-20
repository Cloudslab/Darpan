from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from ..types import Trajectory
from .actor_critic import ActorCriticAlgorithm


class IMPALA(ActorCriticAlgorithm):
    """V-trace actor-critic update for off-policy actor trajectories."""

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        self.model.train()
        for trajectory in trajectories:
            if not trajectory.steps:
                continue
            obs = torch.as_tensor(
                np.stack([step.observation for step in trajectory.steps]),
                dtype=torch.float32,
                device=self.device,
            )
            mask = torch.as_tensor(
                np.stack([step.action_mask for step in trajectory.steps]),
                dtype=torch.bool,
                device=self.device,
            )
            actions = torch.as_tensor(
                [step.action for step in trajectory.steps],
                dtype=torch.long,
                device=self.device,
            )
            rewards = torch.as_tensor(
                [step.reward for step in trajectory.steps],
                dtype=torch.float32,
                device=self.device,
            )
            behavior_log = torch.as_tensor(
                [step.log_prob for step in trajectory.steps],
                dtype=torch.float32,
                device=self.device,
            )
            done = torch.as_tensor(
                [step.done for step in trajectory.steps],
                dtype=torch.float32,
                device=self.device,
            )
            logits, values = self.model(obs, mask)
            distribution = Categorical(logits=logits)
            target_log = distribution.log_prob(actions)
            ratios = torch.exp(target_log.detach() - behavior_log)
            rho = torch.clamp(ratios, max=self.config.vtrace_rho_clip)
            c = torch.clamp(ratios, max=self.config.vtrace_c_clip)

            with torch.no_grad():
                bootstrap = torch.tensor(0.0, device=self.device)
                vs = torch.zeros_like(values)
                next_vs = bootstrap
                next_value = bootstrap
                for index in range(len(trajectory.steps) - 1, -1, -1):
                    discount = self.config.gamma * (1.0 - done[index])
                    delta = rho[index] * (
                        rewards[index] + discount * next_value - values[index]
                    )
                    vs[index] = values[index] + delta + discount * c[index] * (
                        next_vs - next_value
                    )
                    next_vs = vs[index]
                    next_value = values[index]
                next_v = torch.cat([vs[1:], bootstrap.unsqueeze(0)])
                pg_advantage = rho * (
                    rewards + self.config.gamma * (1.0 - done) * next_v - values
                )

            policy_loss = -(target_log * pg_advantage.detach()).mean()
            value_loss = ((values - vs.detach()) ** 2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            ) * trajectory.sample_weight
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
        if not losses:
            return {}
        self.policy_version += 1
        self.episodes += len(trajectories)
        return {
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "policy_version": float(self.policy_version),
        }
