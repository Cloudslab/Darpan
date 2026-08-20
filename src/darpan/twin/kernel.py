"""Default discrete-event queue used by the Digital Twin backend."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from darpan.core.event import Event


@dataclass(order=True, slots=True)
class _Scheduled:
    at: float
    sequence: int
    event: Event = field(compare=False)


class DiscreteEventQueue:
    def __init__(self) -> None:
        self._queue: list[_Scheduled] = []
        self._sequence = 0

    def schedule(self, event: Event, at: float) -> None:
        if at < 0:
            raise ValueError("scheduled time cannot be negative")
        self._sequence += 1
        heapq.heappush(self._queue, _Scheduled(at, self._sequence, event))

    def pop(self) -> tuple[float, Event]:
        scheduled = heapq.heappop(self._queue)
        return scheduled.at, scheduled.event

    def __bool__(self) -> bool:
        return bool(self._queue)

    def clear(self) -> None:
        self._queue.clear()
