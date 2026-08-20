from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from ..config import RLConfig
from ..types import ActionOutput, Trajectory


class Algorithm(ABC):
    def __init__(self, observation_dim: int, action_dim: int, config: RLConfig) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device)
        self.policy_version = 0
        self.episodes = 0
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> ActionOutput: ...

    @abstractmethod
    def update(self, trajectories: list[Trajectory]) -> dict[str, float]: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
