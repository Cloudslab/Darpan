from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ActionOutput:
    action: int
    log_prob: float
    value: float
    policy_version: int


@dataclass(slots=True)
class Step:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray | None
    next_action_mask: np.ndarray | None
    done: bool
    log_prob: float
    value: float
    policy_version: int


@dataclass(slots=True)
class Trajectory:
    steps: list[Step]
    source: str = "real"
    sample_weight: float = 1.0
    model_uncertainty: float = 0.0

    @property
    def reward(self) -> float:
        return sum(step.reward for step in self.steps)
