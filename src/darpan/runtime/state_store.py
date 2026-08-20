from __future__ import annotations

from darpan.core.event import Event
from darpan.core.state import ContinuumState

from .reducer import reduce_event


class StateStore:
    def __init__(self, initial: ContinuumState | None = None) -> None:
        self._state = initial if initial is not None else ContinuumState()

    @property
    def state(self) -> ContinuumState:
        return self._state

    def apply(self, event: Event) -> ContinuumState:
        self._state = reduce_event(self._state, event)
        return self._state
