"""Offline Real-vs-Twin fidelity comparison over durable canonical event traces."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from darpan.core.event import Event, EventKind

from .benchmark import FidelityBenchmarkSummary, FidelitySample, summarize_fidelity


@dataclass(frozen=True, slots=True)
class TraceMetricReport:
    metric: str
    matched: int
    real_only: int
    twin_only: int
    summary: FidelityBenchmarkSummary | None
    samples: tuple[FidelitySample, ...]


@dataclass(frozen=True, slots=True)
class TraceFidelityReport:
    application_latency: TraceMetricReport
    execution_duration: TraceMetricReport
    artifact_size: TraceMetricReport
    queue_delay: TraceMetricReport
    transfer_duration: TraceMetricReport
    migration_downtime: TraceMetricReport
    restart_downtime: TraceMetricReport
    retry_downtime: TraceMetricReport
    scale_convergence: TraceMetricReport
    route_change: TraceMetricReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _component_key(event: Event) -> tuple[str, str]:
    return (
        str(event.payload.get("application_id", "")),
        str(event.payload.get("component_id", event.subject or "")),
    )


def _transfer_key(event: Event) -> tuple[str, str, str]:
    return (
        str(event.payload.get("source_component_id", "")),
        str(event.payload.get("target_component_id", "")),
        str(event.payload.get("artifact", "")),
    )


def _indexed_values(
    events: Iterable[Event],
    *,
    kind: str,
    field: str,
    key_fn,
) -> dict[tuple[Any, ...], float]:
    counts: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    values: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.kind != kind or field not in event.payload:
            continue
        base = tuple(key_fn(event))
        occurrence = counts[base]
        counts[base] += 1
        values[(*base, occurrence)] = float(event.payload[field])
    return values


def _transition_downtime_values(
    events: Iterable[Event],
    *,
    transition_kind: str,
) -> dict[tuple[Any, ...], float]:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    pending: dict[str, tuple[tuple[str, str, int], float]] = {}
    values: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.kind == transition_kind:
            base = _component_key(event)
            occurrence = counts[base]
            counts[base] += 1
            instance_id = str(event.payload.get("instance_id", event.subject or ""))
            pending[instance_id] = ((*base, occurrence), event.event_time)
            continue
        if event.kind != EventKind.COMPONENT_STARTED:
            continue
        instance_id = str(event.payload.get("instance_id", event.subject or ""))
        started = pending.pop(instance_id, None)
        if started is None:
            continue
        key, transition_at = started
        values[key] = max(0.0, event.event_time - transition_at)
    return values


def _paired_transition_values(
    events: Iterable[Event],
    *,
    start_kind: str,
    end_kind: str,
) -> dict[tuple[Any, ...], float]:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    pending: dict[str, tuple[tuple[str, str, int], float]] = {}
    values: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.kind == start_kind:
            base = _component_key(event)
            occurrence = counts[base]
            counts[base] += 1
            instance_id = str(event.payload.get("instance_id", event.subject or ""))
            pending[instance_id] = ((*base, occurrence), event.event_time)
            continue
        if event.kind != end_kind:
            continue
        instance_id = str(event.payload.get("instance_id", event.subject or ""))
        started = pending.pop(instance_id, None)
        if started is None:
            continue
        key, started_at = started
        values[key] = max(0.0, event.event_time - started_at)
    return values


def _flow_transition_values(
    events: Iterable[Event],
) -> dict[tuple[Any, ...], float]:
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    pending: dict[str, tuple[tuple[str, str, str, int], float]] = {}
    values: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.kind == EventKind.FLOW_ROUTING:
            base = (
                str(event.payload.get("application_id", "")),
                str(event.payload.get("source_component_id", "")),
                str(event.payload.get("target_component_id", "")),
            )
            occurrence = counts[base]
            counts[base] += 1
            key = str(event.causation_id or event.subject or event.id)
            pending[key] = ((*base, occurrence), event.event_time)
            continue
        if event.kind != EventKind.FLOW_ROUTED:
            continue
        key = str(event.causation_id or event.subject or event.id)
        started = pending.pop(key, None)
        if started is None:
            continue
        alignment_key, started_at = started
        values[alignment_key] = max(0.0, event.event_time - started_at)
    return values


def _application_latency_values(events: Iterable[Event]) -> dict[tuple[Any, ...], float]:
    counts: defaultdict[str, int] = defaultdict(int)
    pending: dict[str, tuple[tuple[str, int], float]] = {}
    values: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.kind == EventKind.APPLICATION_SUBMITTED:
            application_id = str(event.payload.get("application_id", ""))
            occurrence = counts[application_id]
            counts[application_id] += 1
            instance_id = str(event.payload.get("instance_id", event.subject or ""))
            pending[instance_id] = ((application_id, occurrence), event.event_time)
            continue
        if event.kind != EventKind.APPLICATION_COMPLETED:
            continue
        instance_id = str(event.payload.get("instance_id", event.subject or ""))
        started = pending.pop(instance_id, None)
        if started is None:
            continue
        key, submitted_at = started
        values[key] = max(0.0, event.event_time - submitted_at)
    return values


def _compare_metric(
    *,
    metric: str,
    real: Mapping[tuple[Any, ...], float],
    twin: Mapping[tuple[Any, ...], float],
) -> TraceMetricReport:
    shared = sorted(real.keys() & twin.keys(), key=repr)
    samples = tuple(
        FidelitySample(
            predicted=twin[key],
            observed=real[key],
            metadata={"alignment_key": list(key)},
        )
        for key in shared
    )
    return TraceMetricReport(
        metric=metric,
        matched=len(shared),
        real_only=len(real.keys() - twin.keys()),
        twin_only=len(twin.keys() - real.keys()),
        summary=(None if not samples else summarize_fidelity(samples)),
        samples=samples,
    )


def compare_event_traces(
    real_events: Iterable[Event],
    twin_events: Iterable[Event],
) -> TraceFidelityReport:
    """Align repeated canonical events and summarize Twin-vs-Real errors.

    Alignment intentionally ignores generated application-instance IDs because
    independent Real and Twin runs usually generate different IDs. Components
    are instead matched by application/component identity and occurrence order.
    Transfer samples use source/target component, artifact, and occurrence.
    """

    real_items = tuple(real_events)
    twin_items = tuple(twin_events)
    return TraceFidelityReport(
        application_latency=_compare_metric(
            metric="application.latency_s",
            real=_application_latency_values(real_items),
            twin=_application_latency_values(twin_items),
        ),
        execution_duration=_compare_metric(
            metric="component.duration_s",
            real=_indexed_values(
                real_items,
                kind=EventKind.COMPONENT_COMPLETED,
                field="duration_s",
                key_fn=_component_key,
            ),
            twin=_indexed_values(
                twin_items,
                kind=EventKind.COMPONENT_COMPLETED,
                field="duration_s",
                key_fn=_component_key,
            ),
        ),
        artifact_size=_compare_metric(
            metric="component.output_bytes",
            real=_indexed_values(
                real_items,
                kind=EventKind.COMPONENT_COMPLETED,
                field="output_bytes",
                key_fn=_component_key,
            ),
            twin=_indexed_values(
                twin_items,
                kind=EventKind.COMPONENT_COMPLETED,
                field="output_bytes",
                key_fn=_component_key,
            ),
        ),
        queue_delay=_compare_metric(
            metric="component.queue_delay_s",
            real=_indexed_values(
                real_items,
                kind=EventKind.COMPONENT_STARTED,
                field="queue_delay_s",
                key_fn=_component_key,
            ),
            twin=_indexed_values(
                twin_items,
                kind=EventKind.COMPONENT_STARTED,
                field="queue_delay_s",
                key_fn=_component_key,
            ),
        ),
        transfer_duration=_compare_metric(
            metric="transfer.duration_s",
            real=_indexed_values(
                real_items,
                kind=EventKind.DATA_TRANSFER_COMPLETED,
                field="duration_s",
                key_fn=_transfer_key,
            ),
            twin=_indexed_values(
                twin_items,
                kind=EventKind.DATA_TRANSFER_COMPLETED,
                field="duration_s",
                key_fn=_transfer_key,
            ),
        ),
        migration_downtime=_compare_metric(
            metric="migration.downtime_s",
            real=_transition_downtime_values(
                real_items, transition_kind=EventKind.COMPONENT_MIGRATING
            ),
            twin=_transition_downtime_values(
                twin_items, transition_kind=EventKind.COMPONENT_MIGRATING
            ),
        ),
        restart_downtime=_compare_metric(
            metric="restart.downtime_s",
            real=_transition_downtime_values(
                real_items, transition_kind=EventKind.COMPONENT_RESTARTING
            ),
            twin=_transition_downtime_values(
                twin_items, transition_kind=EventKind.COMPONENT_RESTARTING
            ),
        ),
        retry_downtime=_compare_metric(
            metric="retry.downtime_s",
            real=_transition_downtime_values(
                real_items, transition_kind=EventKind.COMPONENT_RETRYING
            ),
            twin=_transition_downtime_values(
                twin_items, transition_kind=EventKind.COMPONENT_RETRYING
            ),
        ),
        scale_convergence=_compare_metric(
            metric="scale.convergence_s",
            real=_paired_transition_values(
                real_items,
                start_kind=EventKind.COMPONENT_SCALING,
                end_kind=EventKind.COMPONENT_SCALED,
            ),
            twin=_paired_transition_values(
                twin_items,
                start_kind=EventKind.COMPONENT_SCALING,
                end_kind=EventKind.COMPONENT_SCALED,
            ),
        ),
        route_change=_compare_metric(
            metric="route.change_s",
            real=_flow_transition_values(real_items),
            twin=_flow_transition_values(twin_items),
        ),
    )


def resolve_event_trace(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "events.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(f"event trace does not exist: {candidate}")
    return candidate


def load_event_trace(path: str | Path) -> tuple[Event, ...]:
    source = resolve_event_trace(path)
    events = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                events.append(Event.from_dict(raw))
            except Exception as exc:
                raise ValueError(
                    f"invalid event trace {source}:{line_number}: {exc}"
                ) from exc
    return tuple(events)
