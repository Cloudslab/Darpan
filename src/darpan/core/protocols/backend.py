from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from ..action import Action
from ..event import Event
from ..state import ContinuumState


@dataclass(frozen=True, slots=True)
class BackendContext:
    emit: Callable[[Event], Awaitable[None]]
    state: Callable[[], ContinuumState]
    now: Callable[[], float]


class RuntimeBackend(Protocol):
    async def start(self, context: BackendContext) -> None: ...

    async def apply(self, action: Action) -> None: ...

    async def close(self) -> None: ...
