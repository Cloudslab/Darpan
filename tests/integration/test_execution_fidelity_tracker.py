from __future__ import annotations

import asyncio
import sys

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.experiment.fidelity import ExecutionFidelityTracker
from darpan.twin.models.execution import ExecutionTimeModel
from darpan.twin.models.registry import ModelRegistry


def test_execution_fidelity_tracker_predicts_before_calibrating_current_sample():
    async def run():
        models = ModelRegistry([ExecutionTimeModel(calibration_rate=1.0)])
        tracker = ExecutionFidelityTracker(models)
        session = Darpan.real()
        session.subscribe(tracker.observe)
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),)
        )
        app = ApplicationSpec(
            "fidelity",
            components=(
                ComponentSpec(
                    "task",
                    command=(sys.executable, "-c", "import time; time.sleep(0.03)"),
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.5,
                ),
            ),
        )
        await session.start()
        await session.register_system(system)
        for index in range(2):
            instance = await session.submit_application(app, instance_id=f"run-{index}")
            await session.apply(Action.place(f"{instance}:task", "edge"))
            await session.wait_for(
                lambda event, state, expected=instance: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == expected
                ),
                timeout=2.0,
            )
        assert len(tracker.samples) == 2
        assert tracker.samples[0].metadata["samples"] == 0
        assert tracker.samples[1].metadata["samples"] == 1
        assert tracker.summary().samples == 2
        await session.close()

    asyncio.run(run())
