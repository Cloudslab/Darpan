"""Calibration bridge: physical events update the Twin model registry."""

from __future__ import annotations

from darpan.core.event import Event
from darpan.core.state import ContinuumState

from .models.registry import ModelRegistry


class TwinCalibrator:
    def __init__(self, models: ModelRegistry) -> None:
        self.models = models
        self.observations = 0

    def observe(self, event: Event, state: ContinuumState) -> None:
        self.models.observe(event, state)
        self.observations += 1
