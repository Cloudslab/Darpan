from __future__ import annotations

from dataclasses import dataclass

from darpan.core.event import Event
from darpan.core.state import ContinuumState


@dataclass(frozen=True, slots=True)
class SimulationResult:
    state: ContinuumState
    events: tuple[Event, ...]
