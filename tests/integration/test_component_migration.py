from __future__ import annotations

import asyncio

import pytest

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, ResourceRequest
from darpan.core.action import ActionKind
from darpan.core.event import EventKind


def _service_app() -> ApplicationSpec:
    return ApplicationSpec(
        "migrating-service",
        components=(
            ComponentSpec(
                "api",
                kind="service",
                resources=(ResourceRequest("cpu", 1),),
                work_units=0,
            ),
        ),
    )


@pytest.mark.parametrize("factory", [Darpan.twin, Darpan.real])
def test_long_running_component_migrates_without_terminal_completion(
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
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == component_id
                and event.payload.get("node_id") == "edge-1"
            ),
            timeout=2,
        )
        assert session.state.components[component_id].node_id == "edge-1"
        assert session.state.nodes["edge-1"].resources["cpu"].allocated == 1

        after = session.event_count
        migrate = Action.migrate(component_id, "fog-1", metadata={"reason": "drain"})
        assert await session.apply(migrate)
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == component_id
                and event.payload.get("node_id") == "fog-1"
            ),
            after=after,
            timeout=2,
        )

        component = session.state.components[component_id]
        assert component.status == "running"
        assert component.node_id == "fog-1"
        assert component.attempt == 1
        assert session.state.nodes["edge-1"].resources["cpu"].allocated == 0
        assert session.state.nodes["fog-1"].resources["cpu"].allocated == 1
        assert not any(
            event.kind in {EventKind.COMPONENT_COMPLETED, EventKind.APPLICATION_COMPLETED}
            for event in session.events_since(after)
        )
        migrating = next(
            event
            for event in session.events_since(after)
            if event.kind == EventKind.COMPONENT_MIGRATING
        )
        assert migrating.payload["from_node_id"] == "edge-1"
        assert migrating.payload["to_node_id"] == "fog-1"
        assert migrating.payload["attempt"] == 1
        assert migrating.payload["mode"] == "restart"

        action_request = next(
            event
            for event in session.events_since(after)
            if event.kind == EventKind.ACTION_REQUESTED
            and event.payload.get("action_id") == migrate.id
        )
        assert action_request.payload["metadata"]["reason"] == "drain"

        assert await session.apply(Action.stop(component_id))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.APPLICATION_COMPLETED
                and event.payload.get("instance_id") == instance
            ),
            timeout=2,
        )
        await session.close()

    asyncio.run(run())


def test_migrate_rejects_invalid_targets_and_non_running_components(small_system):
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(_service_app())
        component_id = f"{instance}:api"

        assert not await session.apply(Action.migrate(component_id, "fog-1"))
        assert await session.apply(Action.place(component_id, "edge-1"))
        assert not await session.apply(Action.migrate(component_id, "edge-1"))
        assert not await session.apply(Action.migrate(component_id, "missing"))

        rejections = [
            event.payload["reason"]
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.MIGRATE
        ]
        assert any("not running" in reason for reason in rejections)
        assert any("already on node" in reason for reason in rejections)
        assert any("unknown node" in reason for reason in rejections)
        assert not await session.apply(
            Action.migrate(component_id, "fog-1", mode="live")
        )
        rejection = [
            event.payload["reason"]
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.MIGRATE
        ][-1]
        assert "unsupported migration mode" in rejection
        await session.close()

    asyncio.run(run())


def test_twin_migration_preserves_running_stream_successor(small_system):
    async def run() -> None:
        from darpan import FlowSpec

        app = ApplicationSpec(
            "migrating-stream",
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
        assert session.state.components[sink].status == "ready"
        assert await session.apply(Action.place(sink, "fog-1"))
        assert session.state.components[sink].status == "running"

        assert await session.apply(Action.migrate(source, "cloud-1"))
        assert session.state.components[source].status == "running"
        assert session.state.components[source].node_id == "cloud-1"
        assert session.state.components[sink].status == "running"
        assert session.state.components[sink].node_id == "fog-1"
        assert not any(
            event.kind == EventKind.APPLICATION_COMPLETED for event in session.event_log
        )

        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())


def test_twin_migrates_fair_cpu_service_and_rebalances_resources():
    async def run() -> None:
        from darpan import NodeSpec, ResourceSpec, SystemSpec

        system = SystemSpec(
            nodes=(
                NodeSpec(
                    "left",
                    resources=(
                        ResourceSpec("cpu", 1, attributes={"scheduling": "fair"}),
                    ),
                ),
                NodeSpec(
                    "right",
                    resources=(
                        ResourceSpec("cpu", 1, attributes={"scheduling": "fair"}),
                    ),
                ),
            ),
        )
        app = ApplicationSpec(
            "fair-service",
            components=(
                ComponentSpec(
                    "api",
                    kind="service",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0,
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        component_id = f"{instance}:api"

        assert await session.apply(Action.place(component_id, "left"))
        assert session.state.nodes["left"].resources["cpu"].allocated == 1
        assert await session.apply(Action.migrate(component_id, "right"))
        assert session.state.nodes["left"].resources["cpu"].allocated == 0
        assert session.state.nodes["right"].resources["cpu"].allocated == 1
        assert session.state.components[component_id].status == "running"
        assert session.state.components[component_id].attempt == 1
        assert await session.apply(Action.stop(component_id))
        await session.close()

    asyncio.run(run())
