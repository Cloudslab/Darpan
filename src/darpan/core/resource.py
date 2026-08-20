"""Extensible resource model.

Resources are named rather than hard-coded so research can introduce GPUs,
NPUs, battery budgets, accelerators, or domain-specific capacities without
changing Darpan Core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    capacity: float
    unit: str = "count"
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource name cannot be empty")
        if self.capacity < 0:
            raise ValueError("resource capacity cannot be negative")
        scheduling = self.attributes.get("scheduling", "reserve")
        if scheduling not in {"reserve", "fair"}:
            raise ValueError("resource scheduling must be 'reserve' or 'fair'")


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    name: str
    amount: float
    unit: str = "count"

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("resource request cannot be negative")


@dataclass(frozen=True, slots=True)
class ResourceState:
    name: str
    capacity: float
    allocated: float = 0.0
    unit: str = "count"
    attributes: Mapping[str, str] = field(default_factory=dict)

    @property
    def available(self) -> float:
        return max(0.0, self.capacity - self.allocated)

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, max(0.0, self.allocated / self.capacity))

    @property
    def scheduling(self) -> str:
        return self.attributes.get("scheduling", "reserve")

    @property
    def is_shareable(self) -> bool:
        return self.scheduling == "fair"

    def can_allocate(self, amount: float) -> bool:
        return amount >= 0 and self.available + 1e-12 >= amount

    def can_admit(self, amount: float) -> bool:
        if amount < 0 or amount > self.capacity + 1e-12:
            return False
        return self.is_shareable or self.can_allocate(amount)
