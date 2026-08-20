from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec


def _app() -> ApplicationSpec:
    return ApplicationSpec(
        "mixed",
        components=tuple(
            ComponentSpec(
                name,
                resources=(
                    ResourceRequest("cpu", 1),
                    ResourceRequest("memory", 1),
                ),
                work_units=1.0,
            )
            for name in ("a", "b")
        ),
    )


def test_fair_cpu_still_respects_strict_secondary_resource_capacity():
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(
            SystemSpec(
                nodes=(
                    NodeSpec(
                        "edge",
                        resources=(
                            ResourceSpec(
                                "cpu",
                                1,
                                attributes={"scheduling": "fair"},
                            ),
                            ResourceSpec("memory", 1),
                        ),
                    ),
                )
            )
        )
        instance = await session.submit_application(_app())
        await session.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await session.advance()
        starts = [
            event.event_time
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        ]
        assert starts == [0.0, 1.0]
        await session.close()

    asyncio.run(run())


def test_fair_cpu_shares_when_secondary_resource_has_room():
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(
            SystemSpec(
                nodes=(
                    NodeSpec(
                        "edge",
                        resources=(
                            ResourceSpec(
                                "cpu",
                                1,
                                attributes={"scheduling": "fair"},
                            ),
                            ResourceSpec("memory", 2),
                        ),
                    ),
                )
            )
        )
        instance = await session.submit_application(_app())
        await session.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await session.advance()
        starts = [
            event.event_time
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        ]
        completed = [
            event.event_time
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.payload.get("component_id") in {"a", "b"}
        ]
        assert starts == [0.0, 0.0]
        assert completed == [2.0, 2.0]
        await session.close()

    asyncio.run(run())
