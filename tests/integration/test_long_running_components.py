from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import EventKind


def _stream_app():
    return ApplicationSpec(
        "stream-pipeline",
        components=(
            ComponentSpec("ingest", kind="service", work_units=0),
            ComponentSpec("processor", kind="stream", work_units=0),
        ),
        flows=(FlowSpec("ingest", "processor", kind="stream"),),
    )


def test_twin_stream_dependency_becomes_ready_when_service_starts(small_system):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(_stream_app())

        await session.apply(Action.place(f"{instance}:ingest", "edge-1"))
        assert session.state.components[f"{instance}:ingest"].status == "running"
        assert session.state.components[f"{instance}:processor"].status == "ready"

        await session.apply(Action.place(f"{instance}:processor", "fog-1"))
        assert session.state.components[f"{instance}:processor"].status == "running"
        assert not any(
            event.kind == EventKind.APPLICATION_COMPLETED
            for event in session.event_log
        )

        assert await session.apply(Action.stop(f"{instance}:ingest"))
        assert session.state.components[f"{instance}:ingest"].status == "completed"
        assert await session.apply(Action.stop(f"{instance}:processor"))
        assert session.state.components[f"{instance}:processor"].status == "completed"
        assert any(
            event.kind == EventKind.APPLICATION_COMPLETED
            for event in session.event_log
        )
        await session.close()

    asyncio.run(run())


def test_real_long_running_component_stays_alive_until_stop(small_system):
    async def run():
        app = ApplicationSpec(
            "service-app",
            components=(ComponentSpec("api", kind="service", work_units=0),),
        )
        session = Darpan.real()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(app)
        assert await session.apply(Action.place(f"{instance}:api", "edge-1"))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == f"{instance}:api"
            ),
            timeout=2,
        )

        assert session.state.components[f"{instance}:api"].status == "running"
        assert await session.apply(Action.stop(f"{instance}:api"))
        completed = await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=2,
        )
        assert completed.event.payload["success"] is True
        component = session.state.components[f"{instance}:api"]
        assert component.status == "completed"
        stop_event = next(
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.subject == f"{instance}:api"
        )
        assert stop_event.payload["stopped"] is True
        await session.close()

    asyncio.run(run())


def test_stop_is_rejected_for_finite_task(small_system, small_app):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        await session.apply(Action.place(f"{instance}:a", "edge-1"))
        stopped = await session.apply(Action.stop(f"{instance}:a"))
        assert not stopped
        rejection = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == "component.stop"
        ][-1]
        assert "not long-running" in rejection.payload["reason"]
        await session.close()

    asyncio.run(run())
