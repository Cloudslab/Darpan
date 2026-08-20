"""Evaluation objectives remain separate from RL rewards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Objective:
    metric: str
    direction: str = "minimize"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("objective direction must be minimize or maximize")

    def value(self, metrics: Mapping[str, float]) -> float:
        value = float(metrics[self.metric])
        signed = value if self.direction == "minimize" else -value
        return self.weight * signed
