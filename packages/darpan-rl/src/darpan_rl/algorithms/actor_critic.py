from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.distributions import Categorical

from ..config import RLConfig
from ..networks import ActorCriticNetwork
from ..types import ActionOutput, Trajectory
from .base import Algorithm


class ActorCriticAlgorithm(Algorithm):
    def __init__(self, observation_dim: int, action_dim: int, config: RLConfig) -> None:
        super().__init__(observation_dim, action_dim, config)
        self.model = ActorCriticNetwork(
            observation_dim, action_dim, config.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    def _distribution(self, observation, action_mask):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        logits, value = self.model(obs, mask)
        return Categorical(logits=logits), value

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        self.model.eval()
        with torch.no_grad():
            distribution, value = self._distribution(observation, action_mask)
            if deterministic:
                action = int(torch.argmax(distribution.logits).item())
            else:
                action = int(distribution.sample().item())
            log_prob = float(
                distribution.log_prob(torch.tensor(action, device=self.device)).item()
            )
        return ActionOutput(action, log_prob, float(value.item()), self.policy_version)

    def _returns_advantages(self, trajectory: Trajectory):
        rewards = [step.reward for step in trajectory.steps]
        values = [step.value for step in trajectory.steps]
        advantages = [0.0] * len(rewards)
        gae = 0.0
        next_value = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            nonterminal = 0.0 if trajectory.steps[index].done else 1.0
            delta = rewards[index] + self.config.gamma * next_value * nonterminal - values[index]
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * nonterminal
                * gae
            )
            advantages[index] = gae
            next_value = values[index]
        returns = [adv + value for adv, value in zip(advantages, values, strict=True)]
        return returns, advantages

    def _batch(self, trajectories: Iterable[Trajectory]):
        observations = []
        masks = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        weights = []
        policy_versions = []
        for trajectory in trajectories:
            episode_returns, episode_advantages = self._returns_advantages(trajectory)
            for step, ret, adv in zip(
                trajectory.steps, episode_returns, episode_advantages, strict=True
            ):
                observations.append(step.observation)
                masks.append(step.action_mask)
                actions.append(step.action)
                old_log_probs.append(step.log_prob)
                returns.append(ret)
                advantages.append(adv)
                weights.append(trajectory.sample_weight)
                policy_versions.append(step.policy_version)
        if not observations:
            return None
        obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.device)
        action = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        weight = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        version = torch.as_tensor(policy_versions, dtype=torch.long, device=self.device)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return obs, mask, action, old_log, ret, adv, weight, version

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_version": self.policy_version,
            "episodes": self.episodes,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.policy_version = int(state.get("policy_version", 0))
        self.episodes = int(state.get("episodes", 0))
