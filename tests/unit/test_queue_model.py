from __future__ import annotations

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState, NodeState
from darpan.twin.models.queue import QueueDelayModel


def _state() -> ContinuumState:
    from darpan.core.resource import ResourceState

    return ContinuumState(
        nodes={
            "edge": NodeState(
                "edge",
                "edge",
                resources={"cpu": ResourceState("cpu", capacity=2)},
            )
        }
    )


def test_queue_model_finds_first_capacity_window():
    model = QueueDelayModel()
    prediction = model.predict(
        {
            "node_id": "edge",
            "earliest_start": 0.0,
            "duration_s": 2.0,
            "requests": {"cpu": 1.0},
            "reservations": (
                {"start": 0.0, "end": 5.0, "resources": {"cpu": 1.0}},
                {"start": 0.0, "end": 3.0, "resources": {"cpu": 1.0}},
            ),
        },
        _state(),
    )
    assert prediction.estimate == 3.0
    assert prediction.metadata["structural_wait_s"] == 3.0


def test_queue_model_allows_parallel_work_with_spare_capacity():
    model = QueueDelayModel()
    prediction = model.predict(
        {
            "node_id": "edge",
            "earliest_start": 0.0,
            "duration_s": 2.0,
            "requests": {"cpu": 1.0},
            "reservations": (
                {"start": 0.0, "end": 5.0, "resources": {"cpu": 1.0}},
            ),
        },
        _state(),
    )
    assert prediction.estimate == 0.0


def test_queue_model_learns_real_queue_delay():
    model = QueueDelayModel(calibration_rate=1.0)
    model.observe(
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=1.5,
            source="runtime.real",
            payload={"node_id": "edge", "queue_delay_s": 1.5},
        ),
        _state(),
    )
    prediction = model.predict(
        {
            "node_id": "edge",
            "earliest_start": 2.0,
            "duration_s": 1.0,
            "requests": {"cpu": 1.0},
            "reservations": (),
        },
        _state(),
    )
    assert prediction.estimate == 1.5
    assert prediction.metadata["samples"] == 1
