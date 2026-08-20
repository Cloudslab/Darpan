from __future__ import annotations

import asyncio

import pytest

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    Darpan,
    NodeSpec,
    ResourceRequest,
    ResourceSpec,
    SystemSpec,
)
from darpan.core.event import EventKind
from darpan.core.measurement import Measurement
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.telemetry import measurement_event
from darpan.runtime.session import Session


def _system(*, fair: bool = False) -> SystemSpec:
    return SystemSpec(
        nodes=(
            NodeSpec(
                "edge",
                resources=(
                    ResourceSpec(
                        "cpu",
                        8,
                        attributes={"scheduling": "fair"} if fair else {},
                    ),
                ),
            ),
        ),
    )


def _app(cpu: float) -> ApplicationSpec:
    return ApplicationSpec(
        "capacity-app",
        components=(
            ComponentSpec(
                "task",
                resources=(ResourceRequest("cpu", cpu),),
                work_units=1,
            ),
        ),
    )


@pytest.mark.parametrize("factory", [Darpan.real, Darpan.twin])
@pytest.mark.parametrize("fair", [False, True])
def test_physical_cpu_measurement_caps_placement_admission(factory, fair):
    async def run() -> None:
        session = factory()
        await session.start()
        await session.register_system(_system(fair=fair))
        await session.emit(
            measurement_event(
                Measurement(
                    "compute.cpu_capacity",
                    2.0,
                    "cpu",
                    target="edge",
                    timestamp=session.clock.now(),
                    source="test",
                ),
                source="test",
            )
        )
        instance = await session.submit_application(_app(3.0))
        component_id = f"{instance}:task"
        assert session.state.nodes["edge"].effective_resource_capacity("cpu") == 2.0
        assert not await session.apply(Action.place(component_id, "edge"))
        rejected = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == "component.place"
        ][-1]
        assert "available 2.0" in rejected.payload["reason"]
        await session.close()

    asyncio.run(run())


def test_physical_measurement_can_shrink_but_not_expand_declared_capacity():
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(_system())
        node = session.state.nodes["edge"]
        assert node.effective_resource_capacity("cpu") == 8
        await session.emit(
            measurement_event(
                Measurement(
                    "compute.cpu_capacity",
                    20.0,
                    "cpu",
                    target="edge",
                    timestamp=session.clock.now(),
                    source="test",
                ),
                source="test",
            )
        )
        assert session.state.nodes["edge"].effective_resource_capacity("cpu") == 8
        await session.close()

    asyncio.run(run())


def test_real_backend_fails_impossible_effective_capacity_without_validators():
    async def run() -> None:
        session = Session(RealBackend(), validators=[])
        await session.start()
        await session.register_system(_system())
        await session.emit(
            measurement_event(
                Measurement(
                    "compute.cpu_capacity",
                    2.0,
                    "cpu",
                    target="edge",
                    timestamp=session.clock.now(),
                    source="test",
                ),
                source="test",
            )
        )
        instance = await session.submit_application(_app(3.0))
        component_id = f"{instance}:task"
        assert await session.apply(Action.place(component_id, "edge"))
        failed = await session.wait_for(
            lambda event, _state: (
                event.kind == EventKind.COMPONENT_FAILED
                and event.subject == component_id
            ),
            timeout=2.0,
        )
        assert "request exceeds effective cpu capacity" in failed.event.payload["error"]
        assert session.state.components[component_id].status == "failed"
        await session.close()

    asyncio.run(run())
