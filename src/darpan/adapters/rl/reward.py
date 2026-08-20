"""RL reward is deliberately separate from experiment objectives/metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from darpan.core.action import Action
from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState


@dataclass(frozen=True, slots=True)
class Transition:
    before: ContinuumState
    action: Action
    after: ContinuumState
    events: tuple[Event, ...]
    terminated: bool


class RewardFunction(Protocol):
    def compute(self, transition: Transition) -> float: ...


class CompletionTimeReward:
    """Sparse default: negative elapsed Twin/real time when application completes."""

    def compute(self, transition: Transition) -> float:
        for event in transition.events:
            if event.kind == EventKind.APPLICATION_COMPLETED:
                return -max(0.0, transition.after.time - transition.before.time)
        return 0.0
