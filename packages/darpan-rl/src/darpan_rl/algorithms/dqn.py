from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..config import RLConfig
from ..networks import QNetwork
from ..types import ActionOutput, Trajectory
from .base import Algorithm


@dataclass(slots=True)
class _Replay:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray | None
    next_mask: np.ndarray | None
    done: bool
    weight: float


class DQN(Algorithm):
    def __init__(self, observation_dim: int, action_dim: int, config: RLConfig) -> None:
        super().__init__(observation_dim, action_dim, config)
        self.online = QNetwork(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.target = QNetwork(observation_dim, action_dim, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque[_Replay] = deque(maxlen=config.replay_capacity)
        self.updates = 0
        self.random = random.Random(config.seed)

    def epsilon(self) -> float:
        progress = min(1.0, self.episodes / max(1, self.config.epsilon_decay_episodes))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, observation, action_mask, *, deterministic=False) -> ActionOutput:
        valid = np.flatnonzero(action_mask)
        if len(valid) == 0:
            raise RuntimeError("DQN received an action mask with no valid action")
        if not deterministic and self.random.random() < self.epsilon():
            action = int(self.random.choice(valid.tolist()))
        else:
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                q = self.online(obs, mask)
                action = int(torch.argmax(q).item())
        return ActionOutput(action, 0.0, 0.0, self.policy_version)

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        for trajectory in trajectories:
            for step in trajectory.steps:
                self.replay.append(
                    _Replay(
                        step.observation,
                        step.action_mask,
                        step.action,
                        step.reward,
                        step.next_observation,
                        step.next_action_mask,
                        step.done,
                        trajectory.sample_weight,
                    )
                )
            self.episodes += 1
        if len(self.replay) < min(self.config.replay_batch_size, 8):
            return {"replay_size": float(len(self.replay)), "epsilon": self.epsilon()}
        batch = self.random.sample(
            list(self.replay), min(self.config.replay_batch_size, len(self.replay))
        )
        obs = torch.as_tensor(
            np.stack([item.observation for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.as_tensor(
            np.stack([item.mask for item in batch]), dtype=torch.bool, device=self.device
        )
        action = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        reward = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        weight = torch.as_tensor(
            [item.weight for item in batch], dtype=torch.float32, device=self.device
        )
        current = self.online(obs, mask).gather(1, action.unsqueeze(1)).squeeze(1)
        targets = reward.clone()
        nonterminal_indices = [
            i
            for i, item in enumerate(batch)
            if not item.done and item.next_observation is not None
        ]
        if nonterminal_indices:
            next_obs = torch.as_tensor(
                np.stack([batch[i].next_observation for i in nonterminal_indices]),
                dtype=torch.float32,
                device=self.device,
            )
            next_mask = torch.as_tensor(
                np.stack([batch[i].next_mask for i in nonterminal_indices]),
                dtype=torch.bool,
                device=self.device,
            )
            with torch.no_grad():
                next_q = self.target(next_obs, next_mask).max(dim=1).values
            index_tensor = torch.as_tensor(
                nonterminal_indices, dtype=torch.long, device=self.device
            )
            targets[index_tensor] += self.config.gamma * next_q
        loss = (((current - targets) ** 2) * weight).sum() / weight.sum().clamp_min(1e-8)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.policy_version += 1
        return {
            "loss": float(loss.item()),
            "replay_size": float(len(self.replay)),
            "epsilon": self.epsilon(),
            "policy_version": float(self.policy_version),
        }

    def state_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_version": self.policy_version,
            "episodes": self.episodes,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state.get("target", state["online"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.policy_version = int(state.get("policy_version", 0))
        self.episodes = int(state.get("episodes", 0))
        self.updates = int(state.get("updates", 0))
