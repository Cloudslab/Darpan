import pytest

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState
from darpan.experiment.metric import MigrationDowntime


def test_migration_downtime_tracks_migrating_to_next_started_event():
    metric = MigrationDowntime()
    state = ContinuumState()
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=1.0,
            source="test",
            payload={"instance_id": "app:svc"},
        ),
        state,
    )
    assert metric.result() is None
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_MIGRATING,
            event_time=2.0,
            source="test",
            payload={"instance_id": "app:svc", "mode": "restart"},
        ),
        state,
    )
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=2.75,
            source="test",
            payload={"instance_id": "app:svc"},
        ),
        state,
    )
    assert metric.result() == 0.75


def test_restart_downtime_tracks_restarting_to_next_started_event():
    from darpan.experiment.metric import RestartDowntime

    metric = RestartDowntime()
    state = ContinuumState()
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_RESTARTING,
            event_time=4.0,
            source="test",
            payload={"instance_id": "app:svc"},
        ),
        state,
    )
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=4.4,
            source="test",
            payload={"instance_id": "app:svc"},
        ),
        state,
    )
    assert metric.result() == pytest.approx(0.4)


def test_scale_convergence_tracks_scaling_to_converged_event():
    from darpan.experiment.metric import ScaleConvergence

    metric = ScaleConvergence()
    state = ContinuumState()
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_SCALING,
            event_time=5.0,
            source="test",
            payload={"instance_id": "app:svc", "desired_replicas": 3},
        ),
        state,
    )
    metric.observe(
        Event(
            kind=EventKind.COMPONENT_SCALED,
            event_time=6.25,
            source="test",
            payload={"instance_id": "app:svc", "desired_replicas": 3},
        ),
        state,
    )
    assert metric.result() == pytest.approx(1.25)


def test_route_change_latency_tracks_routing_to_routed_event():
    from darpan.experiment.metric import RouteChangeLatency

    metric = RouteChangeLatency()
    state = ContinuumState()
    metric.observe(
        Event(
            kind=EventKind.FLOW_ROUTING,
            event_time=2.0,
            source="test",
            subject="app:source->sink",
            causation_id="route-action",
        ),
        state,
    )
    metric.observe(
        Event(
            kind=EventKind.FLOW_ROUTED,
            event_time=2.35,
            source="test",
            subject="app:source->sink",
            causation_id="route-action",
        ),
        state,
    )
    assert metric.result() == pytest.approx(0.35)
