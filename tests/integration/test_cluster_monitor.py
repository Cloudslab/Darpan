from __future__ import annotations

import asyncio

from darpan import Darpan
from darpan.core.event import EventKind
from darpan.runtime.real.cluster.monitor import ClusterMonitor


class FlappingClient:
    def __init__(self) -> None:
        self.up = True

    async def ping(self):
        if not self.up:
            raise ConnectionError("offline")
        return {"node_id": "edge-1"}

    async def telemetry(self):
        if not self.up:
            raise ConnectionError("offline")
        return {
            "node_id": "edge-1",
            "cpu_capacity": 1.5,
            "load_1m": 0.5,
            "workspace_free_bytes": 10,
        }


def test_cluster_monitor_emits_offline_recovery_and_telemetry(small_system):
    async def run():
        session = Darpan.real()
        client = FlappingClient()
        monitor = ClusterMonitor(
            session,
            {"edge-1": client},
            interval_s=60,
            emit_telemetry=True,
        )
        session.add_service(monitor)
        await session.start()
        await session.register_system(small_system)

        # NODE_REGISTERED triggers an immediate physical telemetry refresh, so
        # the first placement decision does not wait for the periodic poll.
        assert session.state.nodes["edge-1"].measurement("agent.load_1m").value == 0.5
        assert (
            session.state.nodes["edge-1"].effective_resource_capacity("cpu")
            == 1.5
        )
        await monitor.poll_once()
        assert session.state.nodes["edge-1"].measurement("agent.load_1m").value == 0.5
        assert (
            session.state.nodes["edge-1"].measurement("compute.cpu_capacity").value
            == 1.5
        )

        client.up = False
        await monitor.poll_once()
        assert session.state.nodes["edge-1"].status == "offline"

        client.up = True
        await monitor.poll_once()
        assert session.state.nodes["edge-1"].status == "online"
        kinds = [event.kind for event in session.event_log]
        assert EventKind.NODE_OFFLINE in kinds
        assert EventKind.NODE_RECOVERED in kinds
        await session.close()

    asyncio.run(run())
