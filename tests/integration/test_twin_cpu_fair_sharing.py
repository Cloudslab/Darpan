from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec


def test_twin_fair_cpu_runs_contenders_concurrently_with_max_min_sharing():
    async def run():
        system = SystemSpec(
            nodes=(
                NodeSpec(
                    "edge",
                    resources=(
                        ResourceSpec(
                            "cpu",
                            1,
                            attributes={"scheduling": "fair"},
                        ),
                    ),
                ),
            )
        )
        app = ApplicationSpec(
            "shared",
            components=tuple(
                ComponentSpec(
                    name,
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=1.0,
                )
                for name in ("a", "b")
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await session.advance()

        starts = [
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        ]
        completed = [
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.payload.get("component_id") in {"a", "b"}
        ]
        assert [event.event_time for event in starts] == [0.0, 0.0]
        assert [event.event_time for event in completed] == [2.0, 2.0]
        assert all(event.payload["compute_sharing_policy"] == "max-min" for event in completed)
        assert all(event.payload["predicted_solo_duration_s"] == 1.0 for event in completed)
        assert all(event.payload["compute_sharing_delay_s"] == 1.0 for event in completed)
        assert session.state.nodes["edge"].resources["cpu"].allocated == 0.0
        await session.close()

    asyncio.run(run())
