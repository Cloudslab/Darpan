"""Managed physical-cluster health and telemetry monitoring."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from darpan.core.event import Event, EventKind
from darpan.core.measurement import Measurement
from darpan.runtime.real.telemetry import measurement_event

from ..transport import AgentClient


class ClusterMonitor:
    """Emit canonical node availability and optional agent telemetry events."""

    def __init__(
        self,
        session,
        clients: Mapping[str, AgentClient],
        *,
        interval_s: float = 1.0,
        failure_threshold: int = 1,
        emit_telemetry: bool = True,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("cluster monitor interval must be positive")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.session = session
        self.clients = dict(clients)
        self.interval_s = interval_s
        self.failure_threshold = failure_threshold
        self.emit_telemetry = emit_telemetry
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._subscribed = False
        self._online: dict[str, bool] = {}
        self._failures: dict[str, int] = {}

    def start(self) -> ClusterMonitor:
        if not self._subscribed:
            self.session.subscribe(self._observe)
            self._subscribed = True
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return self

    async def _observe(self, event, state) -> None:
        del state
        if event.kind != EventKind.NODE_REGISTERED:
            return
        node_id = str(event.payload["node"]["id"])
        client = self.clients.get(node_id)
        if client is not None:
            await self._check(node_id, client)

    async def _run(self) -> None:
        while not self._stop.is_set():
            await self.poll_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    async def poll_once(self) -> None:
        await asyncio.gather(
            *(self._check(node_id, client) for node_id, client in self.clients.items())
        )

    async def _check(self, node_id: str, client: AgentClient) -> None:
        # Avoid emitting lifecycle events before the experiment has registered
        # the node's canonical SystemSpec.
        if node_id not in self.session.state.nodes:
            return
        try:
            await client.ping()
            telemetry = await client.telemetry() if self.emit_telemetry else None
        except Exception as exc:
            failures = self._failures.get(node_id, 0) + 1
            self._failures[node_id] = failures
            if failures >= self.failure_threshold and self._online.get(node_id, True):
                self._online[node_id] = False
                reason = str(exc)
                await self.session.emit(
                    Event(
                        kind=EventKind.NODE_OFFLINE,
                        event_time=self.session.clock.now(),
                        source="runtime.cluster.monitor",
                        subject=node_id,
                        payload={"node_id": node_id, "reason": reason},
                    )
                )
                handler = getattr(self.session.backend, "node_unavailable", None)
                if handler is not None:
                    await handler(node_id, reason=reason)
            return

        self._failures[node_id] = 0
        was_online = self._online.get(node_id)
        self._online[node_id] = True
        if was_online is False:
            await self.session.emit(
                Event(
                    kind=EventKind.NODE_RECOVERED,
                    event_time=self.session.clock.now(),
                    source="runtime.cluster.monitor",
                    subject=node_id,
                    payload={"node_id": node_id},
                )
            )
        if telemetry is not None:
            await self._emit_telemetry(node_id, telemetry)

    async def _emit_telemetry(self, node_id: str, telemetry: Mapping[str, Any]) -> None:
        timestamp = self.session.clock.now()
        for key, value in telemetry.items():
            if key == "node_id" or not isinstance(value, (int, float)):
                continue
            if key == "cpu_capacity":
                name = "compute.cpu_capacity"
                unit = "cpu"
            else:
                name = f"agent.{key}"
                unit = "bytes" if key.endswith("_bytes") else "1"
            measurement = Measurement(
                name=name,
                value=float(value),
                unit=unit,
                target=node_id,
                timestamp=timestamp,
                source="runtime.cluster.monitor",
            )
            await self.session.emit(
                measurement_event(measurement, source="runtime.cluster.monitor")
            )
            if key == "cpu_capacity":
                handler = getattr(self.session.backend, "resource_capacity_changed", None)
                if handler is not None:
                    await handler(node_id)

    async def close(self) -> None:
        self._stop.set()
        if self._subscribed:
            self.session.unsubscribe(self._observe)
            self._subscribed = False
        if self._task is not None:
            await self._task
            self._task = None
