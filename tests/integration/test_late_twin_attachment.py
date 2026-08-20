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
        return ExecutionResult(
            return_code=0,
            duration_s=3.0,
            output_bytes=1234,
        )


def test_late_attach_replays_real_history_into_twin_models(small_system, small_app):
    async def run():
        real = Session(RealBackend(default_executor=FixedExecutor()), clock=WallClock())
        await real.start()
        await real.register_system(small_system)
        instance = await real.submit_application(small_app)
        await real.apply(Action.place(f"{instance}:a", "edge-1"))
        await real.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{instance}:b",
            timeout=2,
        )

        twin = DigitalTwin().attach(real)
        execution = twin.models.get("execution").predict(
            {
                "application_id": small_app.id,
                "component_id": "a",
                "node_id": "edge-1",
                "work_units": 0.2,
                "cpu_request": 1,
            },
            real.state,
        )
        artifact = twin.models.get("artifact_size").predict(
            {
                "application_id": small_app.id,
                "component_id": "a",
                "hint_bytes": 1,
            },
            real.state,
        )
        assert execution.estimate == 3.0
        assert execution.metadata["calibrated"] is True
        assert artifact.estimate == 1234
        assert twin.mirror.last_state == real.state
        await real.close()

    asyncio.run(run())
