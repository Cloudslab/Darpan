from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.cli.run import run_experiment_spec
from darpan.core.event import Event, EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.spec import ExperimentSpec


def _write(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("fair", [False, True])
def test_running_compute_fails_immediately_when_node_goes_offline(
    tmp_path: Path,
    fair: bool,
) -> None:
    resource = (
        [{"name": "cpu", "capacity": 1, "attributes": {"scheduling": "fair"}}]
        if fair
        else {"cpu": 1}
    )
    _write(tmp_path / "system.yaml", {"nodes": [{"id": "edge", "resources": resource}]})
    _write(
        tmp_path / "app.yaml",
        {
            "id": "app",
            "components": {
                "task": {"work_units": 10, "resources": {"cpu": 1}},
            },
        },
    )
    _write(
        tmp_path / "scenario.yaml",
        {
            "events": [
                {"at_s": 1, "kind": "node.offline", "node_id": "edge"},
            ]
        },
    )
    _write(
        tmp_path / "experiment.yaml",
        {
            "system": "system.yaml",
            "application": "app.yaml",
            "runtime": "twin",
            "scenario": "scenario.yaml",
            "metrics": ["application_latency_s", "application_success_rate"],
        },
    )

    output = asyncio.run(
        run_experiment_spec(
            ExperimentSpec.load(tmp_path / "experiment.yaml"),
            emit_output=False,
        )
    )
    assert output["application_success_rate"] == 0
    assert output["application_latency_s"] == pytest.approx(1.0)
    assert output["_experiment"]["successful"] is False


@pytest.mark.parametrize("fair", [False, True])
def test_node_failure_releases_resources_cancels_completion_and_does_not_resurrect(
    fair: bool,
) -> None:
    async def run() -> None:
        cpu = ResourceSpec(
            "cpu",
            1,
            attributes={"scheduling": "fair"} if fair else {},
        )
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(cpu,)),))
        app = ApplicationSpec(
            "node-failure",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=10,
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=1.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                1.0,
            )
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_RECOVERED,
                    event_time=2.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                2.0,
            )
            task_id = f"{instance}:task"
            await session.apply(Action.place(task_id, "edge"))
            await session.advance(until=20.0)

            assert session.state.components[task_id].status == "failed"
            assert session.state.nodes["edge"].status == "online"
            assert session.state.nodes["edge"].resources["cpu"].allocated == 0
            task_events = [
                event
                for event in session.event_log
                if event.payload.get("instance_id") == task_id
            ]
            assert any(event.kind == EventKind.COMPONENT_FAILED for event in task_events)
            assert not any(
                event.kind == EventKind.COMPONENT_COMPLETED for event in task_events
            )
        finally:
            await session.close()

    asyncio.run(run())


def test_active_twin_transfer_fails_when_path_node_goes_offline() -> None:
    async def run() -> None:
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 2),)),
                NodeSpec("cloud", resources=(ResourceSpec("cpu", 2),)),
            ),
            links=(LinkSpec("uplink", "edge", "cloud", bandwidth_mbps=8),),
        )
        app = ApplicationSpec(
            "node-transfer-failure",
            components=(
                ComponentSpec(
                    "source",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.0,
                ),
                ComponentSpec(
                    "target",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.0,
                ),
                # Keep one decision ready so placing target does not automatically
                # advance all the way through its transfer before we inject fault.
                ComponentSpec("keep-ready", work_units=0.0),
            ),
            flows=(
                FlowSpec(
                    "source",
                    "target",
                    data_size_bytes=1_000_000,
                    artifact="payload.bin",
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            await session.apply(Action.place(f"{instance}:source", "edge"))
            await session.advance()
            target_id = f"{instance}:target"
            await session.apply(Action.place(target_id, "cloud"))
            started = next(
                event.event_time
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_STARTED
            )
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=started + 0.25,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                started + 0.25,
            )
            await session.advance()

            transfer_events = [
                event
                for event in session.event_log
                if event.kind
                in {EventKind.DATA_TRANSFER_FAILED, EventKind.DATA_TRANSFER_COMPLETED}
            ]
            assert [event.kind for event in transfer_events] == [
                EventKind.DATA_TRANSFER_FAILED
            ]
            assert session.state.components[target_id].status == "failed"
            assert not any(
                event.kind == EventKind.COMPONENT_COMPLETED
                and event.payload.get("instance_id") == target_id
                for event in session.event_log
            )
        finally:
            await session.close()

    asyncio.run(run())
