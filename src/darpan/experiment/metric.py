"""Built-in metrics and metric composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from darpan.core.event import Event, EventKind
from darpan.core.protocols.metric import Metric
from darpan.core.state import ContinuumState


class ApplicationLatency:
    name = "application_latency_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start: dict[str, float] = {}
        self._completed: dict[str, float] = {}

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.APPLICATION_SUBMITTED:
            self._start[str(event.payload["instance_id"])] = event.event_time
        elif event.kind == EventKind.APPLICATION_COMPLETED:
            instance_id = str(event.payload["instance_id"])
            if instance_id in self._start:
                self._completed[instance_id] = event.event_time - self._start[instance_id]

    def result(self) -> float | None:
        if not self._completed:
            return None
        return sum(self._completed.values()) / len(self._completed)


class ApplicationSuccessRate:
    name = "application_success_rate"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._outcomes: list[bool] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.APPLICATION_COMPLETED:
            self._outcomes.append(bool(event.payload.get("success", True)))

    def result(self) -> float | None:
        if not self._outcomes:
            return None
        return sum(self._outcomes) / len(self._outcomes)


class ComponentRetryCount:
    """Number of automatic finite-component retry attempts."""

    name = "component_retry_count"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.COMPONENT_RETRYING:
            self.count += 1

    def result(self) -> int:
        return self.count


class ComponentRetryRecoveryRate:
    """Fraction of retried components that eventually complete successfully."""

    name = "component_retry_recovery_rate"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._retried: set[str] = set()
        self._recovered: set[str] = set()

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        instance_id = event.payload.get("instance_id")
        if instance_id is None:
            return
        instance_id = str(instance_id)
        if event.kind == EventKind.COMPONENT_RETRYING:
            self._retried.add(instance_id)
        elif event.kind == EventKind.COMPONENT_COMPLETED and instance_id in self._retried:
            self._recovered.add(instance_id)

    def result(self) -> float | None:
        if not self._retried:
            return None
        return len(self._recovered) / len(self._retried)


class ComponentRetryExhaustionCount:
    """Retried components that remain failed when their application terminates."""

    name = "component_retry_exhaustion_count"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._retried: set[str] = set()
        self._exhausted: set[str] = set()

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind == EventKind.COMPONENT_RETRYING:
            self._retried.add(str(event.payload["instance_id"]))
            return
        if event.kind != EventKind.APPLICATION_COMPLETED:
            return
        app_instance = str(event.payload["instance_id"])
        for instance_id in self._retried:
            component = state.components.get(instance_id)
            if (
                component is not None
                and component.application_instance_id == app_instance
                and component.status == "failed"
            ):
                self._exhausted.add(instance_id)

    def result(self) -> int:
        return len(self._exhausted)


class EventCount:
    def __init__(self, kind: str, *, name: str | None = None) -> None:
        self.kind = kind
        self.name = name or f"count:{kind}"
        self.reset()

    def reset(self) -> None:
        self.count = 0

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == self.kind:
            self.count += 1

    def result(self) -> int:
        return self.count


class MeasurementPeak:
    def __init__(self, measurement_name: str, *, name: str | None = None) -> None:
        self.measurement_name = measurement_name
        self.name = name or f"peak:{measurement_name}"
        self.reset()

    def reset(self) -> None:
        self.peak: float | None = None

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind != EventKind.MEASUREMENT_OBSERVED:
            return
        raw = event.payload["measurement"]
        if raw.get("name") != self.measurement_name:
            return
        value = float(raw["value"])
        self.peak = value if self.peak is None else max(self.peak, value)

    def result(self) -> float | None:
        return self.peak


class MetricSet:
    def __init__(self, metrics: Iterable[Metric]) -> None:
        self.metrics = list(metrics)

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def observe(self, event: Event, state: ContinuumState) -> None:
        for metric in self.metrics:
            metric.observe(event, state)

    def results(self) -> dict[str, Any]:
        return {metric.name: metric.result() for metric in self.metrics}


class RetryDowntime:
    """Mean finite-component retry interruption before the next attempt starts."""

    name = "retry_downtime_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started: dict[str, float] = {}
        self._durations: list[float] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.COMPONENT_RETRYING:
            self._started[str(event.payload["instance_id"])] = event.event_time
            return
        if event.kind != EventKind.COMPONENT_STARTED:
            return
        instance_id = str(event.payload["instance_id"])
        started = self._started.pop(instance_id, None)
        if started is not None:
            self._durations.append(max(0.0, event.event_time - started))

    def result(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)


class RestartDowntime:
    """Mean in-place long-running component restart interruption."""

    name = "restart_downtime_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started: dict[str, float] = {}
        self._durations: list[float] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.COMPONENT_RESTARTING:
            self._started[str(event.payload["instance_id"])] = event.event_time
            return
        if event.kind != EventKind.COMPONENT_STARTED:
            return
        instance_id = str(event.payload["instance_id"])
        started = self._started.pop(instance_id, None)
        if started is not None:
            self._durations.append(max(0.0, event.event_time - started))

    def result(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)


class MigrationDowntime:
    """Mean restart-style migration interruption observed in canonical events."""

    name = "migration_downtime_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started: dict[str, float] = {}
        self._durations: list[float] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.COMPONENT_MIGRATING:
            self._started[str(event.payload["instance_id"])] = event.event_time
            return
        if event.kind != EventKind.COMPONENT_STARTED:
            return
        instance_id = str(event.payload["instance_id"])
        started = self._started.pop(instance_id, None)
        if started is not None:
            self._durations.append(max(0.0, event.event_time - started))

    def result(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)


class ScaleConvergence:
    """Mean time from a scale target change to a converged running replica set."""

    name = "scale_convergence_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started: dict[str, float] = {}
        self._durations: list[float] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.COMPONENT_SCALING:
            self._started[str(event.payload["instance_id"])] = event.event_time
            return
        if event.kind != EventKind.COMPONENT_SCALED:
            return
        instance_id = str(event.payload["instance_id"])
        started = self._started.pop(instance_id, None)
        if started is not None:
            self._durations.append(max(0.0, event.event_time - started))

    def result(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

class RouteChangeLatency:
    """Mean control-plane time to apply or clear an explicit flow route."""

    name = "route_change_s"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started: dict[str, float] = {}
        self._durations: list[float] = []

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind == EventKind.FLOW_ROUTING:
            key = str(event.causation_id or event.subject or event.id)
            self._started[key] = event.event_time
            return
        if event.kind != EventKind.FLOW_ROUTED:
            return
        key = str(event.causation_id or event.subject or event.id)
        started = self._started.pop(key, None)
        if started is not None:
            self._durations.append(max(0.0, event.event_time - started))

    def result(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)
