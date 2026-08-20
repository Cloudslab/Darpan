"""Resource-aware queue-delay model with online physical calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from darpan.core.event import Event, EventKind
from darpan.core.protocols.model import ModelPrediction
from darpan.core.state import ContinuumState


@dataclass(slots=True)
class _DelayEstimate:
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, value: float, rate: float) -> None:
        value = max(0.0, value)
        if self.count == 0:
            self.mean = value
            self.variance = 0.0
            self.count = 1
            return
        delta = value - self.mean
        self.mean += rate * delta
        self.variance = (1 - rate) * (self.variance + rate * delta * delta)
        self.count += 1


class QueueDelayModel:
    """Predict waiting time before a component can acquire node resources.

    The structural part is deterministic: it searches the Twin's future
    reservations for the first interval with enough capacity.  The calibrated
    part learns any additional physical queue delay reported by Real runtime
    ``component.started`` events.  Keeping these pieces separate prevents a
    calibrated mean from hiding an actual capacity conflict.
    """

    name = "queue"

    def __init__(self, *, calibration_rate: float = 0.25) -> None:
        if not 0 < calibration_rate <= 1:
            raise ValueError("calibration_rate must be in (0, 1]")
        self.calibration_rate = calibration_rate
        self._estimates: dict[str, _DelayEstimate] = {}

    def observe(self, event: Event, state: ContinuumState) -> None:
        del state
        if event.kind != EventKind.COMPONENT_STARTED:
            return
        if event.source == "runtime.twin":
            return
        node_id = event.payload.get("node_id")
        queue_delay = event.payload.get("queue_delay_s")
        if node_id is None or queue_delay is None:
            return
        self._estimates.setdefault(str(node_id), _DelayEstimate()).update(
            float(queue_delay), self.calibration_rate
        )

    def advance(self, start_time: float, end_time: float, state: ContinuumState) -> None:
        del start_time, end_time, state

    @staticmethod
    def _capacity(state: ContinuumState, node_id: str, resource: str) -> float:
        node = state.nodes[node_id]
        return node.effective_resource_capacity(resource)

    @staticmethod
    def _reservation_resources(reservation: Mapping[str, Any]) -> Mapping[str, float]:
        raw = reservation.get("resources", {})
        return {str(key): float(value) for key, value in dict(raw).items()}

    @classmethod
    def _fits(
        cls,
        *,
        start: float,
        duration: float,
        requests: Mapping[str, float],
        reservations: Sequence[Mapping[str, Any]],
        state: ContinuumState,
        node_id: str,
    ) -> bool:
        end = float("inf") if math.isinf(duration) else start + max(0.0, duration)
        for resource, requested in requests.items():
            resource_state = state.nodes[node_id].resources.get(resource)
            if resource_state is not None and resource_state.is_shareable:
                if requested > resource_state.capacity + 1e-12:
                    return False
                continue
            capacity = cls._capacity(state, node_id, resource)
            if requested > capacity + 1e-12:
                return False

            relevant = []
            for reservation in reservations:
                amount = cls._reservation_resources(reservation).get(resource, 0.0)
                if amount <= 0:
                    continue
                reservation_start = float(reservation["start"])
                reservation_end = float(reservation["end"])
                if reservation_start < end and reservation_end > start:
                    relevant.append((reservation_start, reservation_end, amount))

            boundaries = {start}
            if math.isfinite(end):
                boundaries.add(end)
            for reservation_start, reservation_end, _ in relevant:
                if start < reservation_start < end:
                    boundaries.add(reservation_start)
                if start < reservation_end < end:
                    boundaries.add(reservation_end)
            ordered = sorted(boundaries)
            if len(ordered) == 1:
                ordered.append(end)

            for left, right in zip(ordered, ordered[1:], strict=False):
                if right <= left:
                    continue
                allocated = sum(
                    amount
                    for reservation_start, reservation_end, amount in relevant
                    if reservation_start < right and reservation_end > left
                )
                if allocated + requested > capacity + 1e-12:
                    return False
        return True

    @classmethod
    def structural_wait(
        cls,
        *,
        earliest_start: float,
        duration: float,
        requests: Mapping[str, float],
        reservations: Sequence[Mapping[str, Any]],
        state: ContinuumState,
        node_id: str,
    ) -> float:
        if not requests:
            return 0.0
        for resource, requested in requests.items():
            resource_state = state.nodes[node_id].resources.get(resource)
            capacity = (
                float(resource_state.capacity)
                if resource_state is not None and resource_state.is_shareable
                else cls._capacity(state, node_id, resource)
            )
            if requested > capacity + 1e-12:
                return float("inf")

        candidates = {float(earliest_start)}
        for reservation in reservations:
            end = float(reservation["end"])
            if math.isfinite(end) and end >= earliest_start:
                candidates.add(end)
        for candidate in sorted(candidates):
            if cls._fits(
                start=candidate,
                duration=duration,
                requests=requests,
                reservations=reservations,
                state=state,
                node_id=node_id,
            ):
                return max(0.0, candidate - earliest_start)
        return float("inf")

    def predict(self, query: Mapping[str, Any], state: ContinuumState) -> ModelPrediction:
        node_id = str(query["node_id"])
        earliest_start = float(query.get("earliest_start", state.time))
        duration = float(query.get("duration_s", 0.0))
        requests = {
            str(key): float(value)
            for key, value in dict(query.get("requests", {})).items()
            if float(value) > 0
        }
        reservations = tuple(query.get("reservations", ()))
        structural = self.structural_wait(
            earliest_start=earliest_start,
            duration=duration,
            requests=requests,
            reservations=reservations,
            state=state,
            node_id=node_id,
        )
        if not math.isfinite(structural):
            return ModelPrediction(
                estimate=float("inf"),
                uncertainty=1.0,
                metadata={"reachable": False, "structural_wait_s": structural},
            )

        estimate = self._estimates.get(node_id)
        learned = 0.0 if estimate is None else estimate.mean
        count = 0 if estimate is None else estimate.count
        std = 0.0 if estimate is None else math.sqrt(max(0.0, estimate.variance))
        total = structural + learned
        uncertainty = 0.5 if count == 0 else min(
            1.0,
            1.0 / math.sqrt(count) + std / max(total, 1e-9),
        )
        return ModelPrediction(
            estimate=total,
            uncertainty=uncertainty,
            lower=max(0.0, structural + learned - 1.96 * std),
            upper=structural + learned + 1.96 * std,
            metadata={
                "reachable": True,
                "structural_wait_s": structural,
                "calibrated_wait_s": learned,
                "samples": count,
            },
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "calibration_rate": self.calibration_rate,
            "estimates": {
                node_id: {
                    "mean": estimate.mean,
                    "variance": estimate.variance,
                    "count": estimate.count,
                }
                for node_id, estimate in self._estimates.items()
            },
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.calibration_rate = float(snapshot.get("calibration_rate", self.calibration_rate))
        self._estimates = {
            str(node_id): _DelayEstimate(
                mean=float(value["mean"]),
                variance=float(value["variance"]),
                count=int(value["count"]),
            )
            for node_id, value in snapshot.get("estimates", {}).items()
        }
