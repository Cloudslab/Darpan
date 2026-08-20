from __future__ import annotations

import asyncio

import pytest

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, ResourceRequest
from darpan.core.action import ActionKind
from darpan.core.event import EventKind


def _service_app() -> ApplicationSpec:
    return ApplicationSpec(
        "restart-service",
        components=(
            ComponentSpec(
                "api",
                kind="service",
                resources=(ResourceRequest("cpu", 1),),
                work_units=0,
            ),
        ),
    )


@pytest.mark.parametrize("factory", [Darpan.real, Darpan.twin])
def test_running_service_restarts_in_place_without_terminal_completion(
    factory,
    small_system,
):
    async def run() -> None:
        session = factory()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(_service_app())
        component_id = f"{instance}:api"

        assert await session.apply(Action.place(component_id, "edge-1"))
        await session.wait_for(
            lambda event, _state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == component_id
            ),
            timeout=2,
        )
        after = session.event_count
        restart = Action.restart(component_id, metadata={"reason": "config-refresh"})
        assert await session.apply(restart)
        await session.wait_for(
            lambda event, _state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == component_id
                and event.causation_id == restart.id
            ),
            after=after,
            timeout=2,
        )

        component = session.state.components[component_id]
        assert component.status == "running"
        assert component.node_id == "edge-1"
        assert component.attempt == 1
        assert session.state.nodes["edge-1"].resources["cpu"].allocated == 1
        events = session.events_since(after)
        restarting = next(
            event for event in events if event.kind == EventKind.COMPONENT_RESTARTING
        )
        assert restarting.payload["attempt"] == 1
        assert restarting.payload["mode"] == "process"
        assert not any(
            event.kind in {EventKind.COMPONENT_COMPLETED, EventKind.APPLICATION_COMPLETED}
            for event in events
        )
        requested = next(
            event
            for event in events
            if event.kind == EventKind.ACTION_REQUESTED
            and event.payload.get("action_id") == restart.id
        )
        assert requested.payload["metadata"]["reason"] == "config-refresh"

        assert await session.apply(Action.stop(component_id))
        await session.wait_for(
            lambda event, _state: (
                event.kind == EventKind.APPLICATION_COMPLETED
                and event.payload.get("instance_id") == instance
            ),
            timeout=2,
        )
        await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("factory", [Darpan.real, Darpan.twin])
def test_restart_rejects_component_that_is_not_running(factory, small_system):
    async def run() -> None:
        session = factory()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(_service_app())
        component_id = f"{instance}:api"
        assert not await session.apply(Action.restart(component_id))
        rejection = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.RESTART
        ][-1]
        assert "not running" in rejection.payload["reason"]
        await session.close()

    asyncio.run(run())


def test_twin_restart_preserves_running_stream_successor(small_system):
    async def run() -> None:
        from darpan import FlowSpec

        app = ApplicationSpec(
            "restart-stream",
            components=(
                ComponentSpec(
                    "source",
                    kind="service",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0,
                ),
                ComponentSpec(
                    "sink",
                    kind="stream",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0,
                ),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(app)
        source = f"{instance}:source"
        sink = f"{instance}:sink"
        assert await session.apply(Action.place(source, "edge-1"))
        assert await session.apply(Action.place(sink, "fog-1"))
        assert session.state.components[sink].status == "running"

        assert await session.apply(Action.restart(source))
        assert session.state.components[source].status == "running"
        assert session.state.components[source].node_id == "edge-1"
        assert session.state.components[source].attempt == 1
        assert session.state.components[sink].status == "running"
        assert session.state.components[sink].node_id == "fog-1"
        assert not any(
            event.kind == EventKind.APPLICATION_COMPLETED for event in session.event_log
        )

        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())


def test_twin_restart_fair_cpu_service_does_not_leak_resources():
    async def run() -> None:
        from darpan import NodeSpec, ResourceSpec, SystemSpec

        system = SystemSpec(
            nodes=(
                NodeSpec(
                    "edge",
                    resources=(
                        ResourceSpec("cpu", 1, attributes={"scheduling": "fair"}),
                    ),
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(_service_app())
        component_id = f"{instance}:api"
        assert await session.apply(Action.place(component_id, "edge"))
        assert session.state.nodes["edge"].resources["cpu"].allocated == 1
        assert await session.apply(Action.restart(component_id))
        assert session.state.nodes["edge"].resources["cpu"].allocated == 1
        assert session.state.components[component_id].attempt == 1
        assert await session.apply(Action.stop(component_id))
        assert session.state.nodes["edge"].resources["cpu"].allocated == 0
        await session.close()

    asyncio.run(run())
