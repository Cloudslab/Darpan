from __future__ import annotations

from typing import Any, Protocol

from ..event import Event
from ..state import ContinuumState


class Metric(Protocol):
    name: str

    def reset(self) -> None: ...

    def observe(self, event: Event, state: ContinuumState) -> None: ...

    def result(self) -> Any: ...
