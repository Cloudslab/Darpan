from __future__ import annotations

import asyncio

from darpan import Action, Darpan, DigitalTwin
from darpan.core.event import EventKind
from darpan.core.protocols.executor import ExecutionResult
from darpan.runtime.clock import WallClock
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.session import Session


class SizedExecutor:
    async def execute(self, component):
        return ExecutionResult(
            return_code=0,
            duration_s=0.01,
            output_bytes=250_000 if component.id == "a" else 7,
        )


def test_real_output_size_calibrates_twin_artifact_model(small_system, small_app):
    async def run():
        real = Session(RealBackend(default_executor=SizedExecutor()), clock=WallClock())
        twin = DigitalTwin().attach(real)
        await real.start()
        await real.register_system(small_system)
        instance = await real.submit_application(small_app)
        await real.apply(Action.place(f"{instance}:a", "edge-1"))
        await real.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:b"
            ),
            timeout=2,
        )

        prediction = twin.models.get("artifact_size").predict(
            {
                "application_id": small_app.id,
                "component_id": "a",
                "hint_bytes": 1024,
            },
            real.state,
        )
        assert prediction.metadata["calibrated"] is True
        assert prediction.estimate == 250_000
        await real.close()

    asyncio.run(run())


def test_twin_completion_uses_calibrated_output_size(small_system, small_app):
    async def run():
        real = Session(RealBackend(default_executor=SizedExecutor()), clock=WallClock())
        twin = DigitalTwin().attach(real)
        await real.start()
        await real.register_system(small_system)
        real_instance = await real.submit_application(small_app)
        await real.apply(Action.place(f"{real_instance}:a", "edge-1"))
        await real.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{real_instance}:b",
            timeout=2,
        )

        session = Darpan.twin(models=twin.models)
        await session.start()
        await session.register_system(small_system)
        twin_instance = await session.submit_application(small_app, instance_id="simulated")
        await session.apply(Action.place(f"{twin_instance}:a", "edge-1"))
        completed = next(
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.subject == f"{twin_instance}:a"
        )
        assert completed.payload["output_bytes"] == 250_000
        assert completed.payload["model_uncertainty"] is not None
        await session.close()
        await real.close()

    asyncio.run(run())
