from __future__ import annotations

import asyncio
import sys

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec


def test_real_runtime_queues_simultaneous_placements_at_resource_capacity():
    async def run():
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),)
        )
        command = (sys.executable, "-c", "import time; time.sleep(0.06)")
        app = ApplicationSpec(
            "serialized",
            components=(
                ComponentSpec(
                    "a",
                    command=command,
                    resources=(ResourceRequest("cpu", 1),),
                ),
                ComponentSpec(
                    "b",
                    command=command,
                    resources=(ResourceRequest("cpu", 1),),
                ),
            ),
        )
        session = Darpan.real()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3.0,
        )
        starts = [
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        ]
        assert len(starts) == 2
        assert starts[1].event_time > starts[0].event_time
        assert float(starts[1].payload["queue_delay_s"]) > 0.0
        allocations = [
            event
            for event in session.event_log
            if event.kind == EventKind.RESOURCE_ALLOCATED
        ]
        releases = [
            event
            for event in session.event_log
            if event.kind == EventKind.RESOURCE_RELEASED
        ]
        assert len(allocations) == len(releases) == 2
        assert session.state.nodes["edge"].resources["cpu"].allocated == 0.0
        await session.close()

    asyncio.run(run())


def test_real_runtime_fair_cpu_admits_contenders_without_capacity_queueing():
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
        command = (sys.executable, "-c", "import time; time.sleep(0.08)")
        app = ApplicationSpec(
            "shared",
            components=tuple(
                ComponentSpec(
                    name,
                    command=command,
                    resources=(ResourceRequest("cpu", 1),),
                )
                for name in ("a", "b")
            ),
        )
        session = Darpan.real()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3.0,
        )
        starts = [
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
        ]
        assert len(starts) == 2
        assert abs(starts[1].event_time - starts[0].event_time) < 0.05
        assert all(float(event.payload["queue_delay_s"]) < 0.05 for event in starts)
        completed = [
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.payload.get("component_id") in {"a", "b"}
        ]
        assert len(completed) == 2
        assert all(event.payload["compute_contention_observed"] for event in completed)
        assert all(
            event.payload["execution_calibration_eligible"] is False
            for event in completed
        )
        assert session.state.nodes["edge"].resources["cpu"].allocated == 0.0
        await session.close()

    asyncio.run(run())
