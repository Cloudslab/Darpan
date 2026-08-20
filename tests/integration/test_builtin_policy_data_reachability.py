from __future__ import annotations

from dataclasses import replace

import pytest

from darpan.adapters.rl.action import PlacementActionAdapter
from darpan.core.application import ApplicationSpec, ComponentSpec, FlowSpec
from darpan.core.event import Event, EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.baselines import RoundRobinPolicy
from darpan.runtime.dispatch import PolicyDispatcher
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


@pytest.mark.asyncio
async def test_round_robin_and_rl_mask_skip_unreachable_successor_target() -> None:
    session = Session(TwinBackend())
    system = SystemSpec(
        nodes=(
            NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
            NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
        ),
        links=(LinkSpec("edge-fog", "edge", "fog"),),
    )
    app = ApplicationSpec(
        id="app",
        components=(
            ComponentSpec("a", resources=(ResourceRequest("cpu", 1),)),
            ComponentSpec("b", resources=(ResourceRequest("cpu", 1),)),
        ),
        flows=(FlowSpec("a", "b", data_size_bytes=10),),
    )
    policy = RoundRobinPolicy()
    dispatcher = PolicyDispatcher(session, [policy])
    session.subscribe(dispatcher)
    await session.start()
    try:
        await session.register_system(system)
        await session.emit(
            Event(
                kind=EventKind.LINK_REMOVED,
                event_time=session.clock.now(),
                source="test",
                payload={"link_id": "edge-fog"},
            )
        )
        await session.submit_application(app, instance_id="run")
        await session.advance()
        successor = session.state.components["run:b"]
        assert successor.node_id == "edge"

        # Reconstruct a ready successor decision to verify the same contract is
        # visible to RL action masks, not only the built-in policy.
        ready = successor.__class__(
            id=successor.id,
            application_id=successor.application_id,
            application_instance_id=successor.application_instance_id,
            component_id=successor.component_id,
            status="ready",
        )
        state = replace(
            session.state,
            components={**session.state.components, ready.id: ready},
        )
        assert PlacementActionAdapter().action_mask(state, ready) == [True, False]
    finally:
        await session.close()
