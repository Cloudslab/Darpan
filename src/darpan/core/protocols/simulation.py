from __future__ import annotations

from typing import Protocol

from ..event import Event


class SimulationKernel(Protocol):
    """Scheduling kernel used by a Twin runtime.

    The kernel owns event ordering only.  Twin semantics, state transitions and
    model predictions remain in ``TwinBackend`` so a researcher can replace the
    event queue implementation without forking the Digital Twin runtime.
    """

    def schedule(self, event: Event, at: float) -> None: ...

    def pop(self) -> tuple[float, Event]: ...

    def __bool__(self) -> bool: ...

    def clear(self) -> None: ...
