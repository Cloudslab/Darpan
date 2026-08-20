"""Physical-to-Twin event mirror and calibration bridge."""

from __future__ import annotations

from darpan.core.event import Event
from darpan.core.state import ContinuumState

from .calibration import TwinCalibrator
from .models.registry import ModelRegistry


class TwinMirror:
    def __init__(self, models: ModelRegistry) -> None:
        self.calibrator = TwinCalibrator(models)
        self.last_state = ContinuumState()
        self.events_seen = 0

    def observe(self, event: Event, state: ContinuumState) -> None:
        self.last_state = state
        self.events_seen += 1
        self.calibrator.observe(event, state)
