"""Online-calibrated execution-time model with uncertainty."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from darpan.core.event import Event, EventKind
from darpan.core.protocols.model import ModelPrediction
from darpan.core.state import ContinuumState


@dataclass(slots=True)
class _Estimate:
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, value: float, rate: float) -> None:
        if self.count == 0:
            self.mean = value
            self.variance = 0.0
            self.count = 1
            return
        delta = value - self.mean
        self.mean += rate * delta
        self.variance = (1 - rate) * (self.variance + rate * delta * delta)
        self.count += 1


class ExecutionTimeModel:
    name = "execution"

    def __init__(
        self,
        *,
        calibration_rate: float = 0.25,
        work_rate_per_cpu: float = 1.0,
    ) -> None:
        self.calibration_rate = calibration_rate
        self.work_rate_per_cpu = self._validate_work_rate(work_rate_per_cpu)
        self._estimates: dict[str, _Estimate] = {}
        self._global_log_work_rate = _Estimate()

    @staticmethod
    def _validate_work_rate(value: float) -> float:
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("execution work_rate_per_cpu must be finite and positive")
        return rate

    @staticmethod
    def _key(app_id: str, component_id: str, node_id: str) -> str:
        return f"{app_id}|{component_id}|{node_id}"

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind != EventKind.COMPONENT_COMPLETED:
            return
        if event.payload.get("execution_calibration_eligible") is False:
            return
        if "duration_s" not in event.payload or "node_id" not in event.payload:
            return
        instance_id = str(event.payload["instance_id"])
        component_id = str(event.payload.get("component_id", instance_id.split(":", 1)[-1]))
        app_id = event.payload.get("application_id")
        if app_id is None:
            return
        key = self._key(str(app_id), component_id, str(event.payload["node_id"]))
        duration_s = float(event.payload["duration_s"])
        self._estimates.setdefault(key, _Estimate()).update(duration_s, self.calibration_rate)
        instance = state.components.get(instance_id)
        node = state.nodes.get(str(event.payload["node_id"]))
        if instance is None or node is None or duration_s <= 0.0:
            return
        app = state.applications.get(instance.application_id)
        if app is None:
            return
        component = app.component(instance.component_id)
        cpu_request = next(
            (item.amount for item in component.resources if item.name == "cpu"),
            1.0,
        )
        effective_cpu = max(
            1e-6,
            min(node.effective_resource_capacity("cpu"), max(cpu_request, 1e-6)),
        )
        work_rate = component.work_units / (duration_s * effective_cpu)
        if math.isfinite(work_rate) and work_rate > 0.0:
            cumulative_rate = 1.0 / (self._global_log_work_rate.count + 1)
            self._global_log_work_rate.update(math.log(work_rate), cumulative_rate)

    def advance(self, start_time: float, end_time: float, state: ContinuumState) -> None:
        del start_time, end_time, state

    def predict(self, query: Mapping[str, Any], state: ContinuumState) -> ModelPrediction:
        app_id = str(query["application_id"])
        component_id = str(query["component_id"])
        node_id = str(query["node_id"])
        estimate = self._estimates.get(self._key(app_id, component_id, node_id))
        if estimate and estimate.count:
            std = math.sqrt(max(0.0, estimate.variance))
            confidence_width = 1.96 * std
            uncertainty = min(1.0, 1.0 / math.sqrt(estimate.count) + std / max(estimate.mean, 1e-9))
            return ModelPrediction(
                estimate=max(1e-6, estimate.mean),
                uncertainty=uncertainty,
                lower=max(0.0, estimate.mean - confidence_width),
                upper=estimate.mean + confidence_width,
                metadata={"samples": estimate.count, "calibrated": True},
            )

        node = state.nodes[node_id]
        cpu = node.resources.get("cpu")
        capacity = (
            node.effective_resource_capacity("cpu")
            if cpu is not None
            else 1.0
        )
        work_units = float(query.get("work_units", 1.0))
        requested = float(query.get("cpu_request", 1.0))
        effective_cpu = max(1e-6, min(capacity, max(requested, 1e-6)))
        global_rate = self._global_log_work_rate
        calibrated_global = global_rate.count > 0
        work_rate = (
            math.exp(global_rate.mean)
            if calibrated_global
            else self.work_rate_per_cpu
        )
        duration = max(1e-6, work_units / (effective_cpu * work_rate))
        if calibrated_global:
            relative_std = math.sqrt(max(0.0, math.exp(global_rate.variance) - 1.0))
            uncertainty = min(
                1.0,
                1.0 / math.sqrt(global_rate.count) + relative_std,
            )
        else:
            uncertainty = 0.75
        return ModelPrediction(
            estimate=duration,
            uncertainty=uncertainty,
            lower=max(0.0, duration * (1.0 - uncertainty)),
            upper=duration * (1.0 + uncertainty),
            metadata={
                "samples": 0,
                "calibrated": False,
                "calibrated_global": calibrated_global,
                "global_samples": global_rate.count,
                "work_rate_per_cpu": work_rate,
            },
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "calibration_rate": self.calibration_rate,
            "work_rate_per_cpu": self.work_rate_per_cpu,
            "global_work_rate": {
                "geometric_mean": (
                    math.exp(self._global_log_work_rate.mean)
                    if self._global_log_work_rate.count
                    else self.work_rate_per_cpu
                ),
                "log_mean": self._global_log_work_rate.mean,
                "log_variance": self._global_log_work_rate.variance,
                "count": self._global_log_work_rate.count,
            },
            "estimates": {
                key: {
                    "mean": value.mean,
                    "variance": value.variance,
                    "count": value.count,
                }
                for key, value in self._estimates.items()
            },
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.calibration_rate = float(snapshot.get("calibration_rate", self.calibration_rate))
        self.work_rate_per_cpu = self._validate_work_rate(
            float(snapshot.get("work_rate_per_cpu", self.work_rate_per_cpu))
        )
        global_rate = dict(snapshot.get("global_work_rate", {}))
        log_mean = global_rate.get("log_mean")
        if log_mean is None and int(global_rate.get("count", 0)) > 0:
            legacy_mean = float(global_rate.get("mean", self.work_rate_per_cpu))
            log_mean = math.log(self._validate_work_rate(legacy_mean))
        self._global_log_work_rate = _Estimate(
            mean=float(log_mean or 0.0),
            variance=float(global_rate.get("log_variance", 0.0)),
            count=int(global_rate.get("count", 0)),
        )
        self._estimates = {
            key: _Estimate(
                mean=float(value["mean"]),
                variance=float(value["variance"]),
                count=int(value["count"]),
            )
            for key, value in snapshot.get("estimates", {}).items()
        }
