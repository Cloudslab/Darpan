from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ..measurement import Measurement
from ..state import ContinuumState


class TelemetryProvider(Protocol):
    name: str

    def collect(self, state: ContinuumState) -> Iterable[Measurement]: ...
