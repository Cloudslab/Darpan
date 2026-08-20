from __future__ import annotations

import asyncio

import pytest

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    Darpan,
    DigitalTwin,
    ResourceRequest,
)
from darpan.core.action import ActionKind
from darpan.core.event import Event, EventKind
from darpan.experiment.baselines import FirstFitPolicy
from darpan.runtime.dispatch import PolicyDispatcher
from darpan.twin.snapshot import TwinSnapshot


def _service_app() -> ApplicationSpec:
    return ApplicationSpec(
        "scaled-service",
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
def test_long_running_component_scales_out_and_in(factory, small_system):
    async def run() -> None:
        session = factory()
        await session.start()
        await session.register_system(small_system)
        app_instance = await session.submit_application(_service_app())
        primary = f"{app_instance}:api"

        assert await session.apply(Action.place(primary, "edge-1"))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED and event.subject == primary
            ),
            timeout=2,
        )

        after = session.event_count
        scale_out = Action.scale(primary, 3, metadata={"reason": "burst"})
        assert await session.apply(scale_out)
        replicas = session.state.component_replicas(primary, include_terminal=False)
        assert [item.replica_index for item in replicas] == [0, 1, 2]
        assert [item.status for item in replicas] == ["running", "ready", "ready"]
        assert session.state.desired_replicas(primary) == 3
        assert not await session.apply(Action.scale(primary, 4))

        replica1 = f"{primary}#replica-1"
        replica2 = f"{primary}#replica-2"
        assert await session.apply(Action.place(replica1, "fog-1"))
        assert await session.apply(Action.place(replica2, "cloud-1"))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED and event.subject == replica2
            ),
            timeout=2,
        )
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_SCALED
                and event.subject == primary
                and event.payload.get("desired_replicas") == 3
            ),
            timeout=2,
        )
        assert all(
            item.status == "running"
            for item in session.state.component_replicas(primary, include_terminal=False)
        )

        assert await session.apply(Action.scale(replica2, 1))
        assert not session.state.scale_pending(primary)
        assert session.state.desired_replicas(primary) == 1
        active = session.state.component_replicas(primary, include_terminal=False)
        assert [item.id for item in active] == [primary]
        assert session.state.components[replica1].status == "completed"
        assert session.state.components[replica2].status == "completed"
        assert not any(
            event.kind == EventKind.APPLICATION_COMPLETED
            for event in session.events_since(after)
        )

        scaling = [
            event
            for event in session.events_since(after)
            if event.kind == EventKind.COMPONENT_SCALING
        ]
        assert [event.payload["desired_replicas"] for event in scaling] == [3, 1]
        request = next(
            event
            for event in session.events_since(after)
            if event.kind == EventKind.ACTION_REQUESTED
            and event.payload.get("action_id") == scale_out.id
        )
        assert request.payload["metadata"]["reason"] == "burst"

        assert await session.apply(Action.stop(primary))
        completion = await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.APPLICATION_COMPLETED
                and event.payload.get("instance_id") == app_instance
            ),
            timeout=2,
        )
        assert completion.event.payload["success"] is True
        await session.close()

    asyncio.run(run())


def test_scale_rejects_finite_zero_noop_and_unconverged_requests(small_system):
    async def run() -> None:
        finite = ApplicationSpec("finite", components=(ComponentSpec("task"),))
        finite_session = Darpan.twin()
        await finite_session.start()
        await finite_session.register_system(small_system)
        finite_instance = await finite_session.submit_application(finite)
        task = f"{finite_instance}:task"
        assert not await finite_session.apply(Action.scale(task, 2))
        finite_reason = next(
            str(event.payload.get("reason"))
            for event in finite_session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.SCALE
        )
        assert "not long-running" in finite_reason
        await finite_session.close()

        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        app_instance = await session.submit_application(_service_app())
        primary = f"{app_instance}:api"
        assert await session.apply(Action.place(primary, "edge-1"))
        assert not await session.apply(Action.scale(primary, 0))
        assert not await session.apply(Action.scale(primary, 1))
        assert await session.apply(Action.scale(primary, 2))
        assert not await session.apply(Action.scale(primary, 3))

        reasons = [
            str(event.payload.get("reason"))
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.SCALE
        ]
        assert any("at least 1" in reason for reason in reasons)
        assert any("already has 1" in reason for reason in reasons)
        assert any("not converged" in reason for reason in reasons)
        await session.close()

    asyncio.run(run())


def test_scaled_replica_state_survives_twin_snapshot_round_trip(small_system):
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        app_instance = await session.submit_application(_service_app())
        primary = f"{app_instance}:api"
        assert await session.apply(Action.place(primary, "edge-1"))
        assert await session.apply(Action.scale(primary, 2))

        snapshot = DigitalTwin().snapshot(session.state)
        restored = TwinSnapshot.from_dict(snapshot.to_dict())
        replica = restored.state.components[f"{primary}#replica-1"]
        assert replica.replica_index == 1
        assert restored.state.desired_replicas(primary) == 2
        await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("factory", [Darpan.twin, Darpan.real])
def test_scale_out_replicas_follow_normal_policy_placement(factory, small_system):
    async def run() -> None:
        session = factory()
        await session.start()
        await session.register_system(small_system)
        dispatcher = PolicyDispatcher(session, [FirstFitPolicy()])
        session.subscribe(dispatcher)
        app_instance = await session.submit_application(_service_app())
        primary = f"{app_instance}:api"
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED and event.subject == primary
            ),
            timeout=2,
        )

        assert await session.apply(Action.scale(primary, 3))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_SCALED
                and event.subject == primary
                and event.payload.get("desired_replicas") == 3
            ),
            timeout=2,
        )
        replicas = session.state.component_replicas(primary, include_terminal=False)
        assert len(replicas) == 3
        assert all(item.status == "running" for item in replicas)
        assert [item.node_id for item in replicas] == ["cloud-1", "cloud-1", "cloud-1"]
        assert session.state.nodes["cloud-1"].resources["cpu"].allocated == 3

        assert await session.apply(Action.scale(primary, 1))
        assert await session.apply(Action.stop(primary))
        session.unsubscribe(dispatcher)
        await session.close()

    asyncio.run(run())


def test_scaled_replica_set_reconciles_after_member_failure(small_system):
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        dispatcher = PolicyDispatcher(session, [FirstFitPolicy()])
        session.subscribe(dispatcher)
        app_instance = await session.submit_application(_service_app())
        primary = f"{app_instance}:api"
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED
                and event.subject == primary
            ),
            timeout=2,
        )
        assert await session.apply(Action.scale(primary, 2))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_SCALED
                and event.subject == primary
                and event.payload.get("desired_replicas") == 2
            ),
            timeout=2,
        )
        active = session.state.component_replicas(primary, include_terminal=False)
        victim = next(item for item in active if item.replica_index == 1)
        victim_node = victim.node_id
        assert victim_node is not None

        failure_at = session.clock.now()
        session.backend.schedule_event(
            Event(
                kind=EventKind.NODE_OFFLINE,
                event_time=failure_at,
                source="test",
                subject=victim_node,
                payload={"node_id": victim_node},
            ),
            failure_at,
        )
        await session.advance()
        replacement = await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_STARTED
                and bool(event.payload.get("instance_id"))
                and event.payload.get("instance_id") not in {
                    primary, victim.id
                }
            ),
            timeout=2,
        )
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_SCALED
                and event.payload.get("desired_replicas") == 2
                and event.payload.get("running_replicas") == 2
            ),
            timeout=2,
        )
        assert replacement.event.payload["instance_id"].endswith("#replica-2")
        active = session.state.component_replicas(primary, include_terminal=False)
        assert len(active) == 2
        assert all(item.status == "running" for item in active)
        assert session.state.desired_replicas(primary) == 2
        assert not any(
            event.kind == EventKind.APPLICATION_COMPLETED
            for event in session.event_log
        )

        for item in active:
            assert await session.apply(Action.stop(item.id))
        completion = [
            event
            for event in session.event_log
            if event.kind == EventKind.APPLICATION_COMPLETED
            and event.payload.get("instance_id") == app_instance
        ][-1]
        assert completion.payload["success"] is True
        session.unsubscribe(dispatcher)
        await session.close()

    asyncio.run(run())
