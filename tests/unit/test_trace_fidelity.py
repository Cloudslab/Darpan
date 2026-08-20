import pytest

from darpan.core.event import Event, EventKind
from darpan.experiment.trace_fidelity import compare_event_traces


def _component(kind: str, component: str, value_field: str, value: float) -> Event:
    return Event(
        kind=kind,
        event_time=value,
        source="test",
        payload={
            "application_id": "app",
            "component_id": component,
            value_field: value,
        },
    )


def test_trace_fidelity_aligns_by_component_and_occurrence_not_instance_id() -> None:
    real = (
        _component(EventKind.COMPONENT_COMPLETED, "a", "duration_s", 2.0),
        _component(EventKind.COMPONENT_COMPLETED, "a", "duration_s", 4.0),
    )
    twin = (
        _component(EventKind.COMPONENT_COMPLETED, "a", "duration_s", 1.0),
        _component(EventKind.COMPONENT_COMPLETED, "a", "duration_s", 3.0),
    )
    report = compare_event_traces(real, twin)
    assert report.execution_duration.matched == 2
    assert report.execution_duration.summary is not None
    assert report.execution_duration.summary.mean_absolute_error == 1.0
    assert report.execution_duration.real_only == 0
    assert report.execution_duration.twin_only == 0


def test_trace_fidelity_compares_end_to_end_application_latency() -> None:
    def application(kind: str, instance: str, at: float) -> Event:
        return Event(
            kind=kind,
            event_time=at,
            source="test",
            subject=instance,
            payload={"application_id": "app", "instance_id": instance},
        )

    real = (
        application(EventKind.APPLICATION_SUBMITTED, "real-1", 1.0),
        application(EventKind.APPLICATION_COMPLETED, "real-1", 6.0),
        application(EventKind.APPLICATION_SUBMITTED, "real-2", 2.0),
        application(EventKind.APPLICATION_COMPLETED, "real-2", 9.0),
    )
    twin = (
        application(EventKind.APPLICATION_SUBMITTED, "twin-1", 10.0),
        application(EventKind.APPLICATION_COMPLETED, "twin-1", 14.0),
        application(EventKind.APPLICATION_SUBMITTED, "twin-2", 20.0),
        application(EventKind.APPLICATION_COMPLETED, "twin-2", 26.0),
    )

    report = compare_event_traces(real, twin)

    assert report.application_latency.matched == 2
    assert report.application_latency.summary is not None
    assert report.application_latency.summary.mean_absolute_error == 1.0



def test_trace_fidelity_compares_artifact_size_from_component_completion() -> None:
    real = (
        Event(
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=2.0,
            source="real",
            payload={
                "application_id": "app",
                "component_id": "producer",
                "duration_s": 2.0,
                "output_bytes": 1_000_000,
            },
        ),
    )
    twin = (
        Event(
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=1.8,
            source="twin",
            payload={
                "application_id": "app",
                "component_id": "producer",
                "duration_s": 1.8,
                "output_bytes": 900_000,
            },
        ),
    )
    report = compare_event_traces(real, twin)
    assert report.artifact_size.matched == 1
    assert report.artifact_size.summary is not None
    assert report.artifact_size.summary.mean_absolute_error == 100_000


def _migration(kind: str, instance: str, at: float) -> Event:
    return Event(
        kind=kind,
        event_time=at,
        source="test",
        subject=instance,
        payload={
            "instance_id": instance,
            "application_id": "app",
            "component_id": "service",
        },
    )


def test_trace_fidelity_compares_restart_migration_downtime() -> None:
    real = (
        _migration(EventKind.COMPONENT_MIGRATING, "real:service", 10.0),
        _migration(EventKind.COMPONENT_STARTED, "real:service", 11.0),
        _migration(EventKind.COMPONENT_MIGRATING, "real:service", 20.0),
        _migration(EventKind.COMPONENT_STARTED, "real:service", 22.0),
    )
    twin = (
        _migration(EventKind.COMPONENT_MIGRATING, "twin:service", 5.0),
        _migration(EventKind.COMPONENT_STARTED, "twin:service", 5.5),
        _migration(EventKind.COMPONENT_MIGRATING, "twin:service", 8.0),
        _migration(EventKind.COMPONENT_STARTED, "twin:service", 9.5),
    )
    report = compare_event_traces(real, twin)
    assert report.migration_downtime.matched == 2
    assert report.migration_downtime.summary is not None
    assert report.migration_downtime.summary.mean_absolute_error == 0.5


def test_trace_fidelity_compares_in_place_restart_downtime() -> None:
    real = (
        _migration(EventKind.COMPONENT_RESTARTING, "real:service", 10.0),
        _migration(EventKind.COMPONENT_STARTED, "real:service", 10.8),
        _migration(EventKind.COMPONENT_RESTARTING, "real:service", 20.0),
        _migration(EventKind.COMPONENT_STARTED, "real:service", 21.2),
    )
    twin = (
        _migration(EventKind.COMPONENT_RESTARTING, "twin:service", 5.0),
        _migration(EventKind.COMPONENT_STARTED, "twin:service", 5.5),
        _migration(EventKind.COMPONENT_RESTARTING, "twin:service", 8.0),
        _migration(EventKind.COMPONENT_STARTED, "twin:service", 9.0),
    )
    report = compare_event_traces(real, twin)
    assert report.restart_downtime.matched == 2
    assert report.restart_downtime.summary is not None
    assert report.restart_downtime.summary.mean_absolute_error == pytest.approx(0.25)


def test_trace_fidelity_compares_retry_downtime() -> None:
    real = (
        Event(
            kind=EventKind.COMPONENT_RETRYING,
            event_time=2.0,
            source="real",
            subject="r:task",
            payload={
                "instance_id": "r:task",
                "application_id": "app",
                "component_id": "task",
            },
        ),
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=5.0,
            source="real",
            subject="r:task",
            payload={
                "instance_id": "r:task",
                "application_id": "app",
                "component_id": "task",
            },
        ),
    )
    twin = (
        Event(
            kind=EventKind.COMPONENT_RETRYING,
            event_time=1.0,
            source="twin",
            subject="t:task",
            payload={
                "instance_id": "t:task",
                "application_id": "app",
                "component_id": "task",
            },
        ),
        Event(
            kind=EventKind.COMPONENT_STARTED,
            event_time=3.5,
            source="twin",
            subject="t:task",
            payload={
                "instance_id": "t:task",
                "application_id": "app",
                "component_id": "task",
            },
        ),
    )
    report = compare_event_traces(real, twin)
    assert report.retry_downtime.matched == 1
    assert report.retry_downtime.summary is not None
    assert report.retry_downtime.summary.mean_absolute_error == pytest.approx(0.5)


def test_trace_fidelity_compares_scale_convergence() -> None:
    real = (
        _migration(EventKind.COMPONENT_SCALING, "real:service", 10.0),
        _migration(EventKind.COMPONENT_SCALED, "real:service", 12.0),
    )
    twin = (
        _migration(EventKind.COMPONENT_SCALING, "twin:service", 5.0),
        _migration(EventKind.COMPONENT_SCALED, "twin:service", 6.5),
    )
    report = compare_event_traces(real, twin)
    assert report.scale_convergence.matched == 1
    assert report.scale_convergence.summary is not None
    assert report.scale_convergence.summary.mean_absolute_error == pytest.approx(0.5)


def _route(kind: str, at: float, action_id: str) -> Event:
    return Event(
        kind=kind,
        event_time=at,
        source="test",
        subject="instance:source->sink",
        causation_id=action_id,
        payload={
            "application_id": "app",
            "source_component_id": "source",
            "target_component_id": "sink",
        },
    )


def test_trace_fidelity_compares_route_change_latency() -> None:
    real = (
        _route(EventKind.FLOW_ROUTING, 10.0, "real-route"),
        _route(EventKind.FLOW_ROUTED, 10.8, "real-route"),
    )
    twin = (
        _route(EventKind.FLOW_ROUTING, 5.0, "twin-route"),
        _route(EventKind.FLOW_ROUTED, 5.2, "twin-route"),
    )
    report = compare_event_traces(real, twin)
    assert report.route_change.matched == 1
    assert report.route_change.summary is not None
    assert report.route_change.summary.mean_absolute_error == pytest.approx(0.6)
