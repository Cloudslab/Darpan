from __future__ import annotations

import pytest

from darpan.core.event import EventKind
from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.scenario_plan import ScenarioPlan, ScenarioPlanEvent
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


@pytest.mark.asyncio
async def test_scenario_plan_uses_canonical_events_and_composes_link_changes() -> None:
    backend = TwinBackend()
    session = Session(backend)
    system = SystemSpec(
        nodes=(
            NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
            NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
        ),
        links=(LinkSpec("edge-fog", "edge", "fog", latency_ms=5, bandwidth_mbps=100),),
    )
    plan = ScenarioPlan(
        name="degradation",
        events=(
            ScenarioPlanEvent(0, EventKind.NODE_OFFLINE, {"node_id": "edge"}),
            ScenarioPlanEvent(
                1,
                EventKind.LINK_CHANGED,
                {"link_id": "edge-fog", "latency_ms": 25},
            ),
            ScenarioPlanEvent(
                2,
                EventKind.LINK_CHANGED,
                {"link_id": "edge-fog", "bandwidth_mbps": 20},
            ),
            ScenarioPlanEvent(
                3,
                EventKind.MEASUREMENT_OBSERVED,
                {
                    "target": "fog",
                    "name": "compute.cpu_capacity",
                    "value": 0.5,
                    "unit": "count",
                },
            ),
        ),
    )

    await session.start()
    try:
        await session.register_system(system)
        await plan.apply(session)
        assert session.state.nodes["edge"].status == "offline"

        await session.advance(until=3)
        link = session.state.links["edge-fog"].spec
        assert link.latency_ms == 25
        assert link.bandwidth_mbps == 20
        assert session.state.nodes["fog"].measurements["compute.cpu_capacity"].value == 0.5
        scenario_events = [
            event
            for event in session.event_log
            if event.source == "experiment.scenario"
        ]
        assert [event.kind for event in scenario_events] == [
            EventKind.NODE_OFFLINE,
            EventKind.LINK_CHANGED,
            EventKind.LINK_CHANGED,
            EventKind.MEASUREMENT_OBSERVED,
        ]
        assert all(event.metadata["scenario"] == "degradation" for event in scenario_events)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_scenario_plan_rejects_unknown_link() -> None:
    session = Session(TwinBackend())
    await session.start()
    try:
        await session.register_system(
            SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        )
        plan = ScenarioPlan(
            events=(ScenarioPlanEvent(0, "link.down", {"link_id": "missing"}),)
        )
        with pytest.raises(ValueError, match="unknown link"):
            await plan.apply(session)
    finally:
        await session.close()
