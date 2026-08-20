"""Physical link probes that emit canonical network measurements."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from statistics import fmean
from typing import Any

from darpan.core.event import EventKind
from darpan.core.measurement import Measurement
from darpan.runtime.real.telemetry import measurement_event

from ..transport import AgentClient
from .inventory import ClusterInventory


class LinkProbeService:
    """Measure agent-to-agent links and feed the canonical Continuum state.

    Probes are explicitly opt-in on the *source* agent.  The controller asks
    that source agent to contact the target agent, which means the measured
    path is source-node -> target-node rather than controller -> node.
    """

    def __init__(
        self,
        session,
        inventory: ClusterInventory,
        clients: Mapping[str, AgentClient] | None = None,
        *,
        interval_s: float = 10.0,
        samples: int = 3,
        payload_bytes: int = 256 * 1024,
        timeout_s: float = 5.0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("link probe interval must be positive")
        if not 1 <= samples <= 20:
            raise ValueError("link probe samples must be in [1, 20]")
        if not 0 <= payload_bytes <= 4 * 1024 * 1024:
            raise ValueError("link probe payload_bytes must be in [0, 4194304]")
        if not 0 < timeout_s <= 30:
            raise ValueError("link probe timeout_s must be in (0, 30]")
        self.session = session
        self.inventory = inventory
        self.nodes = {node.id: node for node in inventory.nodes}
        default_clients = {node.id: inventory.client(node) for node in inventory.nodes}
        self.clients = dict(default_clients if clients is None else clients)
        self.interval_s = interval_s
        self.samples = samples
        self.payload_bytes = payload_bytes
        self.timeout_s = timeout_s
        self._task: asyncio.Task | None = None
        self._probe_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._subscribed = False

    def start(self) -> LinkProbeService:
        if not self._subscribed:
            self.session.subscribe(self._observe)
            self._subscribed = True
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return self

    def _observe(self, event, state) -> None:
        if event.kind != EventKind.LINK_REGISTERED:
            return
        link = state.links.get(str(event.subject))
        if link is None:
            return
        task = asyncio.create_task(self._probe_link(link.spec))
        self._probe_tasks.add(task)
        task.add_done_callback(self._probe_tasks.discard)

    async def _run(self) -> None:
        while not self._stop.is_set():
            await self.poll_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    async def poll_once(self) -> None:
        links = tuple(self.session.state.links.values())
        await asyncio.gather(*(self._probe_link(link.spec) for link in links))

    async def _probe_link(self, link) -> None:
        source = self.nodes.get(link.source)
        target = self.nodes.get(link.target)
        if source is None or target is None or not source.network_probe:
            return
        client = self.clients.get(source.id)
        if client is None:
            return
        try:
            result = await client.probe_agent(
                target.host,
                target.port,
                token=target.token(),
                ca_pem=self.inventory.tls_ca_pem(target),
                server_hostname=target.tls_server_name or target.host,
                samples=self.samples,
                payload_bytes=self.payload_bytes,
                timeout=self.timeout_s,
            )
        except Exception:
            # Health monitoring owns node availability. A transient link probe
            # failure should not turn a measurement service into a session
            # failure or duplicate node-offline semantics.
            return
        await self._emit(link, result)

    async def _emit(self, link, result: Mapping[str, Any]) -> None:
        timestamp = self.session.clock.now()
        rtt_samples = tuple(float(item) for item in result.get("rtt_samples_ms", ()))
        rtt_ms = float(result["rtt_ms"])
        one_way_latency = max(0.0, rtt_ms / 2.0)
        latency_uncertainty = (
            float(result.get("rtt_std_ms", 0.0)) / 2.0 if rtt_samples else None
        )
        metadata = {
            "source_node_id": link.source,
            "target_node_id": link.target,
            "method": "agent_rpc_rtt",
            "rtt_ms": rtt_ms,
            "samples": int(result.get("samples", len(rtt_samples))),
            "payload_bytes": int(result.get("payload_bytes", 0)),
        }
        await self.session.emit(
            measurement_event(
                Measurement(
                    name="network.latency_ms",
                    value=one_way_latency,
                    unit="ms",
                    target=link.id,
                    timestamp=timestamp,
                    source="runtime.cluster.link_probe",
                    uncertainty=latency_uncertainty,
                    metadata=metadata,
                ),
                source="runtime.cluster.link_probe",
            )
        )
        bandwidth = result.get("bandwidth_mbps")
        if bandwidth is None:
            return
        bandwidth_samples = tuple(
            float(item) for item in result.get("bandwidth_samples_mbps", ())
        )
        uncertainty = None
        if len(bandwidth_samples) > 1:
            mean = fmean(bandwidth_samples)
            variance = fmean((item - mean) ** 2 for item in bandwidth_samples)
            uncertainty = variance**0.5
        await self.session.emit(
            measurement_event(
                Measurement(
                    name="network.bandwidth_mbps",
                    value=max(1e-9, float(bandwidth)),
                    unit="Mbit/s",
                    target=link.id,
                    timestamp=timestamp,
                    source="runtime.cluster.link_probe",
                    uncertainty=uncertainty,
                    metadata=metadata,
                ),
                source="runtime.cluster.link_probe",
            )
        )

    async def close(self) -> None:
        self._stop.set()
        if self._subscribed:
            self.session.unsubscribe(self._observe)
            self._subscribed = False
        if self._task is not None:
            await self._task
            self._task = None
        if self._probe_tasks:
            await asyncio.gather(*self._probe_tasks, return_exceptions=True)
            self._probe_tasks.clear()
