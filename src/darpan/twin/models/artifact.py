"""Online-calibrated component artifact-size model.

Real execution reports the bytes actually produced by a component.  The Twin
uses those observations to replace static flow size hints over time while
retaining explicit uncertainty for sparse or unseen observations.
"""

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


class ArtifactSizeModel:
    """Predict bytes produced by an application component."""

    name = "artifact_size"

    def __init__(self, *, calibration_rate: float = 0.25) -> None:
        if not 0 < calibration_rate <= 1:
            raise ValueError("calibration_rate must be in (0, 1]")
        self.calibration_rate = calibration_rate
        self._estimates: dict[str, _Estimate] = {}

    @staticmethod
    def _key(
        app_id: str,
        component_id: str,
        artifact: str | None = None,
    ) -> str:
        suffix = "" if artifact is None else f"|artifact:{artifact}"
        return f"{app_id}|{component_id}{suffix}"

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind == EventKind.COMPONENT_COMPLETED:
            if "output_bytes" not in event.payload:
                return
            app_id = event.payload.get("application_id")
            component_id = event.payload.get("component_id")
            if app_id is None or component_id is None:
                return
            output_bytes = max(0.0, float(event.payload["output_bytes"]))
            key = self._key(str(app_id), str(component_id))
        elif event.kind == EventKind.DATA_TRANSFER_COMPLETED:
            artifact = event.payload.get("artifact")
            source_instance_id = event.payload.get("source_instance_id")
            if artifact is None or source_instance_id is None:
                return
            source = state.components.get(str(source_instance_id))
            if source is None:
                return
            output_bytes = max(0.0, float(event.payload.get("size_bytes", 0)))
            key = self._key(
                source.application_id,
                source.component_id,
                str(artifact),
            )
        else:
            return
        self._estimates.setdefault(key, _Estimate()).update(
            output_bytes,
            self.calibration_rate,
        )

    def advance(self, start_time: float, end_time: float, state: ContinuumState) -> None:
        del start_time, end_time, state

    def predict(self, query: Mapping[str, Any], state: ContinuumState) -> ModelPrediction:
        del state
        app_id = str(query["application_id"])
        component_id = str(query["component_id"])
        artifact = query.get("artifact")
        estimate = self._estimates.get(
            self._key(
                app_id,
                component_id,
                None if artifact is None else str(artifact),
            )
        )
        if estimate is not None and estimate.count:
            std = math.sqrt(max(0.0, estimate.variance))
            relative_std = std / max(estimate.mean, 1.0)
            uncertainty = min(1.0, 1.0 / math.sqrt(estimate.count) + relative_std)
            width = 1.96 * std
            return ModelPrediction(
                estimate=max(0, int(round(estimate.mean))),
                uncertainty=uncertainty,
                lower=max(0.0, estimate.mean - width),
                upper=max(0.0, estimate.mean + width),
                metadata={
                    "samples": estimate.count,
                    "calibrated": True,
                    "artifact": artifact,
                },
            )

        hint = max(0, int(query.get("hint_bytes", 0)))
        return ModelPrediction(
            estimate=hint,
            uncertainty=0.60 if hint else 1.0,
            lower=0.5 * hint if hint else 0.0,
            upper=2.0 * hint if hint else None,
            metadata={"samples": 0, "calibrated": False, "source": "flow_hint"},
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "calibration_rate": self.calibration_rate,
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
        self._estimates = {
            key: _Estimate(
                mean=float(value["mean"]),
                variance=float(value["variance"]),
                count=int(value["count"]),
            )
            for key, value in snapshot.get("estimates", {}).items()
        }
