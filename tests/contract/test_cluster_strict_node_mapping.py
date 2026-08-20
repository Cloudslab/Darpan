from __future__ import annotations

import asyncio

from darpan.core.action import Action
from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.resource import ResourceRequest
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode
from darpan.runtime.real.cluster.session import session_from_inventory


def test_cluster_session_rejects_system_node_without_physical_agent():
    async def run() -> None:
        server = AgentServer("edge", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (ClusterNode("edge", "127.0.0.1", server.port),)
            )
            session = session_from_inventory(inventory, monitor=False, link_probes=False)
            await session.start()
            await session.register_system(
                SystemSpec(nodes=(NodeSpec("edge"), NodeSpec("ghost")))
            )
            app = ApplicationSpec(
                id="app",
                components=(
                    ComponentSpec(
                        id="task",
                        resources=(ResourceRequest("cpu", 0.1),),
                    ),
                ),
            )
            instance = await session.submit_application(app, instance_id="run")
            component_id = f"{instance}:task"
            accepted = await session.apply(
                Action.place(component_id, "ghost", source="test")
            )
            assert accepted is False
            assert session.state.components[component_id].status == "ready"
            rejected = [
                event
                for event in session.event_log
                if event.kind == "action.rejected"
            ]
            assert rejected
            assert "no executor for node: ghost" in rejected[-1].payload["reason"]
            await session.close()
        finally:
            await server.close()

    asyncio.run(run())
