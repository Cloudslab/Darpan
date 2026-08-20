from __future__ import annotations

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor, mask: torch.Tensor | None = None):
        embedding = self.body(observation)
        logits = self.policy(embedding)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        value = self.value(embedding).squeeze(-1)
        return logits, value


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor, mask: torch.Tensor | None = None):
        q_values = self.net(observation)
        if mask is not None:
            q_values = q_values.masked_fill(~mask, -1e9)
        return q_values
