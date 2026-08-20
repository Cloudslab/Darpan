import pytest

from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.event import Event, EventKind
from darpan.core.resource import ResourceRequest, ResourceState
from darpan.core.state import ComponentInstanceState, ContinuumState, NodeState
from darpan.twin.models.execution import ExecutionTimeModel


def test_execution_model_skips_contention_contaminated_sample() -> None:
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={"cpu": ResourceState("cpu", 1)},
            )
        }
    )
    model = ExecutionTimeModel(calibration_rate=1.0)
    model.observe(
        Event(
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=2.0,
            source="runtime.real",
            payload={
                "instance_id": "run:task",
                "application_id": "app",
                "component_id": "task",
                "node_id": "edge",
                "duration_s": 10.0,
                "execution_calibration_eligible": False,
            },
        ),
        state,
    )
    prediction = model.predict(
        {
            "application_id": "app",
            "component_id": "task",
            "node_id": "edge",
            "work_units": 1.0,
            "cpu_request": 1.0,
        },
        state,
    )
    assert prediction.metadata["calibrated"] is False
    assert prediction.metadata["samples"] == 0


def test_execution_model_restores_a_sealed_work_rate_prior() -> None:
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={"cpu": ResourceState("cpu", 2)},
            )
        }
    )
    model = ExecutionTimeModel(work_rate_per_cpu=1.0)
    model.restore(
        {
            "calibration_rate": 0.25,
            "work_rate_per_cpu": 20.0,
            "estimates": {},
        }
    )
    prediction = model.predict(
        {
            "application_id": "app",
            "component_id": "task",
            "node_id": "edge",
            "work_units": 2.0,
            "cpu_request": 2.0,
        },
        state,
    )
    assert prediction.estimate == 0.05
    assert prediction.metadata["work_rate_per_cpu"] == 20.0
    assert model.snapshot()["work_rate_per_cpu"] == 20.0


def test_execution_model_learns_global_rate_for_unseen_workload() -> None:
    training = ApplicationSpec(
        "training",
        components=(
            ComponentSpec(
                "known",
                work_units=4.0,
                resources=(ResourceRequest("cpu", 1.0),),
            ),
        ),
    )
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={"cpu": ResourceState("cpu", 2)},
            )
        },
        applications={training.id: training},
        components={
            "run:known": ComponentInstanceState(
                id="run:known",
                application_id=training.id,
                application_instance_id="run",
                component_id="known",
                status="completed",
                node_id="edge",
            )
        },
    )
    model = ExecutionTimeModel(calibration_rate=1.0)
    model.observe(
        Event(
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=2.0,
            source="runtime.real",
            payload={
                "instance_id": "run:known",
                "application_id": training.id,
                "component_id": "known",
                "node_id": "edge",
                "duration_s": 2.0,
            },
        ),
        state,
    )

    prediction = model.predict(
        {
            "application_id": "held-out",
            "component_id": "unseen",
            "node_id": "edge",
            "work_units": 6.0,
            "cpu_request": 1.0,
        },
        state,
    )

    assert prediction.estimate == 3.0
    assert prediction.metadata["calibrated_global"] is True
    assert prediction.metadata["global_samples"] == 1


def test_execution_global_rate_is_order_independent_geometric_mean() -> None:
    application = ApplicationSpec(
        "training",
        components=tuple(
            ComponentSpec(
                component_id,
                work_units=work_units,
                resources=(ResourceRequest("cpu", 1.0),),
            )
            for component_id, work_units in (("slow", 2.0), ("fast", 200.0))
        ),
    )
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={"cpu": ResourceState("cpu", 1)},
            )
        },
        applications={application.id: application},
        components={
            f"run:{component_id}": ComponentInstanceState(
                id=f"run:{component_id}",
                application_id=application.id,
                application_instance_id="run",
                component_id=component_id,
                status="completed",
                node_id="edge",
            )
            for component_id in ("slow", "fast")
        },
    )

    def observe(order: tuple[str, str]) -> ExecutionTimeModel:
        model = ExecutionTimeModel()
        for component_id in order:
            model.observe(
                Event(
                    kind=EventKind.COMPONENT_COMPLETED,
                    event_time=2.0,
                    source="runtime.real",
                    payload={
                        "instance_id": f"run:{component_id}",
                        "application_id": application.id,
                        "component_id": component_id,
                        "node_id": "edge",
                        "duration_s": 2.0,
                    },
                ),
                state,
            )
        return model

    forward = observe(("slow", "fast"))
    reverse = observe(("fast", "slow"))
    forward_rate = forward.snapshot()["global_work_rate"]["geometric_mean"]
    reverse_rate = reverse.snapshot()["global_work_rate"]["geometric_mean"]

    assert forward_rate == pytest.approx(10.0)
    assert reverse_rate == pytest.approx(forward_rate)


def test_execution_model_restores_legacy_arithmetic_global_rate() -> None:
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={"cpu": ResourceState("cpu", 1)},
            )
        }
    )
    model = ExecutionTimeModel()
    model.restore(
        {
            "global_work_rate": {"mean": 20.0, "variance": 1.0, "count": 4},
            "estimates": {},
        }
    )
    prediction = model.predict(
        {
            "application_id": "held-out",
            "component_id": "unseen",
            "node_id": "edge",
            "work_units": 20.0,
            "cpu_request": 1.0,
        },
        state,
    )

    assert prediction.estimate == pytest.approx(1.0)
    assert prediction.metadata["global_samples"] == 4
