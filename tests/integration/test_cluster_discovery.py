from __future__ import annotations

import asyncio

from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.discovery import (
    discover_cluster,
    discovered_inventory_dict,
    discovered_system_dict,
)
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


def test_cluster_discovery_generates_hardware_backed_candidates():
    async def run() -> None:
        server = AgentServer("edge", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (
                    ClusterNode(
                        "edge",
                        "127.0.0.1",
                        server.port,
                        labels={"tier": "edge"},
                    ),
                )
            )
            discovery = await discover_cluster(inventory)
            assert len(discovery.environment_fingerprint) == 64
            assert discovery.nodes[0].ping["environment"]["machine"]
            candidate = discovered_inventory_dict(inventory, discovery)
            assert candidate["nodes"][0]["labels"]["tier"] == "edge"
            system = discovered_system_dict(inventory, discovery)
            resources = system["nodes"][0]["resources"]
            assert resources["cpu"] > 0
            assert resources["memory"] > 0
            assert system["links"] == []
        finally:
            await server.close()

    asyncio.run(run())
