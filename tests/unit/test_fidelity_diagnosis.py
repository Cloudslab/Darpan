from __future__ import annotations

from darpan.core.event import Event, EventKind
from darpan.experiment.fidelity_diagnosis import diagnose_trace_fidelity
from darpan.experiment.trace_fidelity import compare_event_traces


def _completed(event_id: str, component: str, duration: float) -> Event:
    return Event(
        id=event_id,
        kind=EventKind.COMPONENT_COMPLETED,
        event_time=duration,
        source="test",
        subject=component,
        payload={
            "application_id": "app",
            "component_id": component,
            "duration_s": duration,
        },
    )


def test_fidelity_diagnosis_ranks_dominant_residual_source() -> None:
    report = compare_event_traces(
        (_completed("r1", "slow", 10), _completed("r2", "fast", 2)),
        (_completed("t1", "slow", 5), _completed("t2", "fast", 1.5)),
    )
    diagnosis = diagnose_trace_fidelity(report)
    assert diagnosis["dominant_metric"] == "execution_duration"
    assert diagnosis["samples"] == 2
    assert diagnosis["entity_ranking"][0]["entity"] == "slow"
    assert diagnosis["entity_ranking"][0]["absolute_error_sum"] == 5


def test_diagnosis_uses_dimensionless_ranking_across_seconds_and_bytes() -> None:
    real = (
        Event(
            id="r1",
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=2.0,
            source="real",
            subject="app:task",
            payload={
                "application_id": "app",
                "component_id": "task",
                "duration_s": 2.0,
                "output_bytes": 1_000_000,
            },
        ),
    )
    twin = (
        Event(
            id="t1",
            kind=EventKind.COMPONENT_COMPLETED,
            event_time=1.0,
            source="twin",
            subject="other:task",
            payload={
                "application_id": "app",
                "component_id": "task",
                "duration_s": 1.0,
                "output_bytes": 900_000,
            },
        ),
    )
    diagnosis = diagnose_trace_fidelity(compare_event_traces(real, twin))
    assert diagnosis["dominant_metric"] == "execution_duration"
    metrics = {item["metric_key"]: item for item in diagnosis["metric_ranking"]}
    assert metrics["execution_duration"]["mean_normalized_error"] == 0.5
    assert metrics["artifact_size"]["mean_normalized_error"] == 0.1
    assert metrics["artifact_size"]["mean_absolute_error"] == 100_000
