from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Constraint:
    metric: str
    operator: str
    threshold: float

    def satisfied(self, metrics: Mapping[str, float]) -> bool:
        value = float(metrics[self.metric])
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == ">":
            return value > self.threshold
        raise ValueError(f"unsupported constraint operator: {self.operator}")
