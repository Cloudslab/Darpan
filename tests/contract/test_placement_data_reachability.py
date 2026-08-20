from __future__ import annotations

import pytest

from darpan.core.action import Action
from darpan.core.application import ApplicationSpec, ComponentSpec, FlowSpec
from darpan.core.event import Event, EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


@pytest.mark.asyncio
async def test_unreachable_data_dependent_placement_is_rejected_before_backend() -> None:
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
    await session.start()
    try:
        await session.register_system(system)
        await session.register_application(app)
        await session.submit_application(app, instance_id="run")
        await session.apply(Action.place("run:a", "edge", source="test"))
        await session.advance()
        assert session.state.components["run:b"].status == "ready"
        await session.emit(
            Event(
                kind=EventKind.LINK_REMOVED,
                event_time=session.clock.now(),
                source="test",
                payload={"link_id": "edge-fog"},
            )
        )

        accepted = await session.apply(Action.place("run:b", "fog", source="test"))
        assert not accepted
        rejected = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
        ][-1]
        assert "no data path" in rejected.payload["reason"]
    finally:
        await session.close()
