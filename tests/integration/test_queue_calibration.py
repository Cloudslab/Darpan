from __future__ import annotations

import asyncio
import sys

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, DigitalTwin
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec


def test_real_queue_delay_calibrates_digital_twin_model():
    async def run():
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),)
        )
        command = (sys.executable, "-c", "import time; time.sleep(0.05)")
        app = ApplicationSpec(
            "queue-calibration",
            components=tuple(
                ComponentSpec(
                    name,
                    command=command,
                    resources=(ResourceRequest("cpu", 1),),
                )
                for name in ("a", "b")
            ),
        )
        real = Darpan.real()
        twin = DigitalTwin().attach(real)
        await real.start()
        await real.register_system(system)
        instance = await real.submit_application(app)
        await real.apply_many(
            [
                Action.place(f"{instance}:a", "edge"),
                Action.place(f"{instance}:b", "edge"),
            ]
        )
        await real.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3.0,
        )
        prediction = twin.models.get("queue").predict(
            {
                "node_id": "edge",
                "earliest_start": real.state.time,
                "duration_s": 0.1,
                "requests": {"cpu": 1.0},
                "reservations": (),
            },
            real.state,
        )
        assert prediction.metadata["samples"] >= 2
        assert prediction.estimate > 0.0
        await real.close()

    asyncio.run(run())
