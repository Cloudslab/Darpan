"""Explicit clock abstractions for real time, virtual time, and replay."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class WallClock:
    def __init__(self) -> None:
        self._origin = perf_counter()

    def now(self) -> float:
        return perf_counter() - self._origin

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


@dataclass(slots=True)
class VirtualClock:
    _time: float = 0.0

    def now(self) -> float:
        return self._time

    async def sleep(self, seconds: float) -> None:
        self.advance_to(self._time + max(0.0, seconds))

    def advance_to(self, value: float) -> None:
        if value < self._time:
            raise ValueError("virtual clock cannot move backwards")
        self._time = value

    def set(self, value: float) -> None:
        if value < 0:
            raise ValueError("virtual time cannot be negative")
        self._time = value
