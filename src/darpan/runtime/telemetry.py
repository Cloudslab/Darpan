"""Periodic physical telemetry sampling over public TelemetryProvider contracts."""

from __future__ import annotations

import asyncio

from darpan.core.protocols.telemetry import TelemetryProvider

from .real.telemetry import measurement_event


class TelemetrySampler:
    def __init__(
        self,
        session,
        provider: TelemetryProvider,
        *,
        interval_s: float = 1.0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("telemetry interval must be positive")
        self.session = session
        self.provider = provider
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> TelemetrySampler:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return self

    async def _run(self) -> None:
        while not self._stop.is_set():
            for measurement in self.provider.collect(self.session.state):
                await self.session.emit(
                    measurement_event(measurement, source=self.provider.name)
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
