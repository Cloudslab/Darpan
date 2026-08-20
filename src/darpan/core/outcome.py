"""Execution and action outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    action_id: str
    accepted: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    application_instance_id: str
    started_at: float
    completed_at: float
    success: bool
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.completed_at - self.started_at)
