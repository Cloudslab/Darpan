from __future__ import annotations

import asyncio

from darpan import Action, DigitalTwin
from darpan.core.event import EventKind
from darpan.core.protocols.executor import ExecutionResult
from darpan.runtime.clock import WallClock
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.session import Session


class FixedExecutor:
    async def execute(self, component):
        return ExecutionResult(return_code=0, duration_s=2.5)


def test_real_execution_calibrates_twin_model(small_system, small_app):
    async def run():
        real = Session(
            RealBackend(default_executor=FixedExecutor()),
            clock=WallClock(),
        )
        twin = DigitalTwin().attach(real)
        await real.start()
        await real.register_system(small_system)
        instance = await real.submit_application(small_app)
        await real.apply(Action.place(f"{instance}:a", "edge-1"))
        await real.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{instance}:b",
            timeout=2,
        )
        prediction = twin.models.get("execution").predict(
            {
                "application_id": small_app.id,
                "component_id": "a",
                "node_id": "edge-1",
                "work_units": 0.2,
                "cpu_request": 1,
            },
            real.state,
        )
        assert prediction.metadata["calibrated"] is True
        assert prediction.estimate == 2.5
        await real.close()

    asyncio.run(run())
