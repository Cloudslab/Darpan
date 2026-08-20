"""Online Real-to-Twin execution fidelity tracking."""

from __future__ import annotations

from dataclasses import asdict

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState
from darpan.twin.models.registry import ModelRegistry

from .benchmark import (
    FidelitySample,
    summarize_calibration_progress,
    summarize_fidelity,
)


class ExecutionFidelityTracker:
    """Make out-of-sample execution predictions before calibrating on outcomes.

    Subscribe this tracker to a physical Session. On ``component.started`` it
    predicts duration using the current Twin execution model. On completion it
    records expected-vs-observed fidelity and only then updates that model,
    preventing the benchmark from evaluating a sample after training on it.
    """

    def __init__(self, models: ModelRegistry, *, calibrate: bool = True) -> None:
        self.models = models
        self.calibrate = calibrate
        self.samples: list[FidelitySample] = []
        self._pending: dict[str, object] = {}

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind == EventKind.COMPONENT_STARTED:
            instance_id = str(event.payload["instance_id"])
            instance = state.components[instance_id]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            node_id = str(event.payload["node_id"])
            cpu_request = next(
                (item.amount for item in component.resources if item.name == "cpu"),
                1.0,
            )
            self._pending[instance_id] = self.models.get("execution").predict(
                {
                    "application_id": app.id,
                    "component_id": component.id,
                    "node_id": node_id,
                    "work_units": component.work_units,
                    "cpu_request": cpu_request,
                },
                state,
            )
            return

        if event.kind != EventKind.COMPONENT_COMPLETED:
            return
        instance_id = str(event.payload["instance_id"])
        prediction = self._pending.pop(instance_id, None)
        if prediction is not None and "duration_s" in event.payload:
            self.samples.append(
                FidelitySample(
                    predicted=float(prediction.estimate),
                    observed=float(event.payload["duration_s"]),
                    lower=prediction.lower,
                    upper=prediction.upper,
                    uncertainty=prediction.uncertainty,
                    metadata={
                        "instance_id": instance_id,
                        "node_id": event.payload.get("node_id"),
                        "application_id": event.payload.get("application_id"),
                        "component_id": event.payload.get("component_id"),
                        **dict(prediction.metadata),
                    },
                )
            )
        if self.calibrate:
            self.models.get("execution").observe(event, state)

    def summary(self):
        return summarize_fidelity(self.samples)

    def calibration_progress(self, *, window_size: int | None = None):
        return summarize_calibration_progress(self.samples, window_size=window_size)


class ArtifactFidelityTracker:
    """Predict output size at component start, score it before calibration."""

    def __init__(self, models: ModelRegistry, *, calibrate: bool = True) -> None:
        self.models = models
        self.calibrate = calibrate
        self.samples: list[FidelitySample] = []
        self._pending: dict[str, object] = {}

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind == EventKind.COMPONENT_STARTED:
            instance_id = str(event.payload["instance_id"])
            instance = state.components[instance_id]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            output_hint = max(
                (flow.data_size_bytes for flow in app.flows if flow.source == component.id),
                default=0,
            )
            self._pending[instance_id] = self.models.get("artifact_size").predict(
                {
                    "application_id": app.id,
                    "component_id": component.id,
                    "hint_bytes": output_hint,
                },
                state,
            )
            return

        if event.kind != EventKind.COMPONENT_COMPLETED:
            return
        instance_id = str(event.payload["instance_id"])
        prediction = self._pending.pop(instance_id, None)
        if prediction is not None and "output_bytes" in event.payload:
            self.samples.append(
                FidelitySample(
                    predicted=float(prediction.estimate),
                    observed=float(event.payload["output_bytes"]),
                    lower=prediction.lower,
                    upper=prediction.upper,
                    uncertainty=prediction.uncertainty,
                    metadata={
                        "instance_id": instance_id,
                        "node_id": event.payload.get("node_id"),
                        "application_id": event.payload.get("application_id"),
                        "component_id": event.payload.get("component_id"),
                        **dict(prediction.metadata),
                    },
                )
            )
        if self.calibrate:
            self.models.get("artifact_size").observe(event, state)

    def summary(self):
        return summarize_fidelity(self.samples)

    def calibration_progress(self, *, window_size: int | None = None):
        return summarize_calibration_progress(self.samples, window_size=window_size)


class NetworkFidelityTracker:
    """Predict physical artifact transfer duration before observing its outcome."""

    def __init__(self, models: ModelRegistry, *, calibrate: bool = True) -> None:
        self.models = models
        self.calibrate = calibrate
        self.samples: list[FidelitySample] = []
        self._pending: dict[str, object] = {}

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind == EventKind.DATA_TRANSFER_STARTED:
            source = event.payload.get("source_node_id")
            target = event.payload.get("target_node_id")
            size_bytes = event.payload.get("size_bytes")
            if source is None or target is None or size_bytes is None:
                return
            transfer_id = str(event.subject or event.id)
            self._pending[transfer_id] = self.models.get("network").predict(
                {
                    "source": str(source),
                    "target": str(target),
                    "size_bytes": int(size_bytes),
                    "earliest_start": state.time,
                    "reservations": (),
                },
                state,
            )
            return

        if event.kind != EventKind.DATA_TRANSFER_COMPLETED:
            return
        transfer_id = str(event.subject or event.id)
        prediction = self._pending.pop(transfer_id, None)
        if prediction is not None and "duration_s" in event.payload:
            self.samples.append(
                FidelitySample(
                    predicted=float(prediction.estimate),
                    observed=float(event.payload["duration_s"]),
                    lower=prediction.lower,
                    upper=prediction.upper,
                    uncertainty=prediction.uncertainty,
                    metadata={
                        "transfer_id": transfer_id,
                        "source_node_id": event.payload.get("source_node_id"),
                        "target_node_id": event.payload.get("target_node_id"),
                        "size_bytes": event.payload.get("size_bytes"),
                        "contention_observed": event.payload.get(
                            "network_contention_observed", False
                        ),
                        "path_representative": event.payload.get(
                            "network_path_representative", True
                        ),
                        "transport": event.payload.get("transport"),
                        "calibration_eligible": event.payload.get(
                            "network_calibration_eligible", True
                        ),
                        **dict(prediction.metadata),
                    },
                )
            )
        if self.calibrate:
            self.models.get("network").observe(event, state)

    def summary(self):
        return summarize_fidelity(self.samples)

    def calibration_progress(self, *, window_size: int | None = None):
        return summarize_calibration_progress(self.samples, window_size=window_size)


class FidelitySuiteTracker:
    """One physical-session observer for execution, artifact, and network fidelity."""

    def __init__(self, models: ModelRegistry, *, calibrate: bool = True) -> None:
        self.execution = ExecutionFidelityTracker(models, calibrate=calibrate)
        self.artifact = ArtifactFidelityTracker(models, calibrate=calibrate)
        self.network = NetworkFidelityTracker(models, calibrate=calibrate)

    def observe(self, event: Event, state: ContinuumState) -> None:
        self.execution.observe(event, state)
        self.artifact.observe(event, state)
        self.network.observe(event, state)

    def summaries(self) -> dict[str, object | None]:
        return {
            "execution": (
                None if not self.execution.samples else self.execution.summary()
            ),
            "artifact": None if not self.artifact.samples else self.artifact.summary(),
            "network": None if not self.network.samples else self.network.summary(),
        }

    @staticmethod
    def _tracker_payload(tracker) -> dict[str, object]:
        samples = tuple(tracker.samples)
        payload: dict[str, object] = {
            "samples": [asdict(item) for item in samples],
            "summary": None if not samples else asdict(tracker.summary()),
        }
        if len(samples) >= 2:
            payload["calibration_progress"] = asdict(tracker.calibration_progress())
        else:
            payload["calibration_progress"] = None
        return payload

    def payload(self) -> dict[str, object]:
        return {
            "execution": self._tracker_payload(self.execution),
            "artifact": self._tracker_payload(self.artifact),
            "network": self._tracker_payload(self.network),
        }
