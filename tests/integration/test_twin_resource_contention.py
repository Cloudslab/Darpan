from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec


def test_twin_uses_node_capacity_instead_of_serializing_entire_node():
    async def run():
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 2),)),)
        )
        app = ApplicationSpec(
            "parallel",
            components=tuple(
                ComponentSpec(
                    name,
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=2.0,
                )
                for name in ("a", "b", "c")
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply_many(
            [
                Action.place(f"{instance}:{name}", "edge")
                for name in ("a", "b", "c")
            ]
        )
        await session.advance()
        starts = {
            event.payload["component_id"]: event.event_time
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        }
        assert starts["a"] == 0.0
        assert starts["b"] == 0.0
        assert starts["c"] == 2.0
        await session.close()

    asyncio.run(run())
