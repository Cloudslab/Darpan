from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.measurement import Measurement
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.runtime.real.telemetry import measurement_event


def test_twin_rebalances_active_fair_cpu_job_when_capacity_changes():
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
            "dynamic-cpu",
            components=(
                ComponentSpec(
                    "work",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=1.0,
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        # Keep a second ready decision so placement does not auto-advance to completion.
        blocker_app = ApplicationSpec("blocker", components=(ComponentSpec("hold"),))
        await session.submit_application(blocker_app, instance_id="blocker-1")
        await session.apply(Action.place(f"{instance}:work", "edge"))
        session.backend.schedule_event(
            measurement_event(
                Measurement(
                    "compute.cpu_capacity",
                    0.5,
                    unit="cpu",
                    target="edge",
                    timestamp=0.25,
                    source="test",
                ),
                source="test",
            ),
            0.25,
        )
        await session.advance()

        completed = next(
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.payload.get("component_id") == "work"
        )
        # 0.25 CPU-seconds at 1 CPU, then 0.75 CPU-seconds at 0.5 CPU.
        assert completed.event_time == 1.75
        assert completed.payload["duration_s"] == 1.75
        await session.close()

    asyncio.run(run())
