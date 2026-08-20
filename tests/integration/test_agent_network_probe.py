from __future__ import annotations

import asyncio

import pytest

from darpan import Darpan
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode
from darpan.runtime.real.cluster.probe import LinkProbeService
from darpan.runtime.real.transport import AgentClient


def test_agent_to_agent_probe_is_opt_in_and_measures_remote_path():
    async def run():
        target = AgentServer("target", host="127.0.0.1", port=0, token="target-secret")
        source = AgentServer(
            "source",
            host="127.0.0.1",
            port=0,
            token="source-secret",
            allow_network_probe=True,
        )
        disabled = AgentServer("disabled", host="127.0.0.1", port=0)
        await target.start()
        await source.start()
        await disabled.start()
        try:
            source_client = AgentClient(
                "127.0.0.1",
                source.port,
                token="source-secret",
            )
            result = await source_client.probe_agent(
                "127.0.0.1",
                target.port,
                token="target-secret",
                samples=2,
                payload_bytes=4096,
            )
            assert result["source_node_id"] == "source"
            assert result["samples"] == 2
            assert result["rtt_ms"] > 0
            assert result["bandwidth_mbps"] > 0
            assert len(result["rtt_samples_ms"]) == 2

            disabled_client = AgentClient("127.0.0.1", disabled.port)
            with pytest.raises(RuntimeError, match="network probing is disabled"):
                await disabled_client.probe_agent(
                    "127.0.0.1",
                    target.port,
                    token="target-secret",
                    samples=1,
                    payload_bytes=0,
                )
        finally:
            await disabled.close()
            await source.close()
            await target.close()

    asyncio.run(run())


def test_link_probe_service_emits_canonical_link_measurements():
    class SourceClient:
        async def probe_agent(self, host, port, **kwargs):
            assert host == "cloud.local"
            assert port == 9000
            assert kwargs["samples"] == 3
            return {
                "rtt_ms": 8.0,
                "rtt_std_ms": 2.0,
                "rtt_samples_ms": [6.0, 8.0, 10.0],
                "bandwidth_mbps": 40.0,
                "bandwidth_samples_mbps": [36.0, 40.0, 44.0],
                "samples": 3,
                "payload_bytes": 1024,
            }

    async def run():
        inventory = ClusterInventory(
            (
                ClusterNode("edge", "edge.local", 8000, network_probe=True),
                ClusterNode("cloud", "cloud.local", 9000),
            )
        )
        session = Darpan.real()
        await session.start()
        try:
            await session.register_system(
                SystemSpec(
                    nodes=(NodeSpec("edge"), NodeSpec("cloud")),
                    links=(LinkSpec("uplink", "edge", "cloud"),),
                )
            )
            service = LinkProbeService(
                session,
                inventory,
                clients={"edge": SourceClient()},
                samples=3,
                payload_bytes=1024,
            )
            await service.poll_once()
            link = session.state.links["uplink"]
            latency = link.measurements["network.latency_ms"]
            bandwidth = link.measurements["network.bandwidth_mbps"]
            assert latency.value == pytest.approx(4.0)
            assert latency.uncertainty == pytest.approx(1.0)
            assert bandwidth.value == pytest.approx(40.0)
            assert bandwidth.uncertainty > 0
            assert latency.metadata["method"] == "agent_rpc_rtt"
        finally:
            await session.close()

    asyncio.run(run())
