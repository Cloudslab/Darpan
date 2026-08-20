"""Network transfer model with topology routing and online physical calibration."""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from darpan.core.event import Event, EventKind
from darpan.core.protocols.model import ModelPrediction
from darpan.core.state import ContinuumState


@dataclass(slots=True)
class _ScaleEstimate:
    mean: float = 1.0
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


class NetworkDelayModel:
    """Predict end-to-end transfer delay from topology and measured transfers."""

    name = "network"

    def __init__(
        self,
        *,
        calibration_rate: float = 0.25,
        min_scale: float = 0.01,
        max_scale: float = 1000.0,
    ) -> None:
        if not 0 < calibration_rate <= 1:
            raise ValueError("calibration_rate must be in (0, 1]")
        self.calibration_rate = calibration_rate
        self.min_scale, self.max_scale = self._validate_scale_bounds(
            min_scale, max_scale
        )
        self._scales: dict[str, _ScaleEstimate] = {}

    @staticmethod
    def _validate_scale_bounds(
        minimum: float, maximum: float
    ) -> tuple[float, float]:
        minimum = float(minimum)
        maximum = float(maximum)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum <= 0.0
            or maximum <= minimum
        ):
            raise ValueError(
                "network calibration scale bounds must be finite, positive, "
                "and ordered"
            )
        return minimum, maximum

    @staticmethod
    def _key(
        source: str,
        target: str,
        links: tuple[str, ...] = (),
    ) -> str:
        suffix = "" if not links else "|route=" + ">".join(links)
        return f"{source}|{target}{suffix}"

    def observe(self, event: Event, state: ContinuumState) -> None:
        if event.kind != EventKind.DATA_TRANSFER_COMPLETED:
            return
        if event.payload.get("network_calibration_eligible") is False:
            return
        source = event.payload.get("source_node_id")
        target = event.payload.get("target_node_id")
        duration = event.payload.get("duration_s")
        if source is None or target is None or duration is None or source == target:
            return
        size_bytes = max(0, int(event.payload.get("size_bytes", 0)))
        route_path = tuple(str(item) for item in event.payload.get("route_path", ()))
        route_links = tuple(str(item) for item in event.payload.get("route_links", ()))
        baseline = self._topology_prediction(
            str(source),
            str(target),
            size_bytes,
            state,
            explicit_path=route_path,
            explicit_links=route_links,
        )
        if not math.isfinite(float(baseline.estimate)) or baseline.estimate <= 1e-9:
            return
        ratio = max(
            self.min_scale,
            min(self.max_scale, float(duration) / float(baseline.estimate)),
        )
        key = self._key(str(source), str(target), route_links)
        self._scales.setdefault(key, _ScaleEstimate()).update(
            ratio, self.calibration_rate
        )

    def advance(self, start_time: float, end_time: float, state: ContinuumState) -> None:
        del start_time, end_time, state

    @staticmethod
    def _edge_delay(latency_ms: float, bandwidth_mbps: float, size_bytes: int) -> float:
        serialization_s = 0.0
        if size_bytes and bandwidth_mbps != float("inf"):
            serialization_s = size_bytes * 8 / (bandwidth_mbps * 1_000_000)
        return latency_ms / 1000.0 + serialization_s

    def _topology_prediction(
        self,
        source: str,
        target: str,
        size_bytes: int,
        state: ContinuumState,
        *,
        explicit_path: tuple[str, ...] = (),
        explicit_links: tuple[str, ...] = (),
    ) -> ModelPrediction:
        if source == target:
            return ModelPrediction(
                estimate=0.0,
                uncertainty=0.0,
                metadata={"path": [source], "links": [], "reachable": True},
            )
        if not state.links:
            return ModelPrediction(
                estimate=0.0,
                uncertainty=0.0,
                metadata={
                    "path": [source, target],
                    "links": [],
                    "reachable": True,
                    "implicit_topology": True,
                    "topology_propagation_s": 0.0,
                    "topology_serialization_s": 0.0,
                    "bottleneck_mbps": float("inf"),
                },
            )

        graph: dict[str, list[tuple[str, float, str]]] = {}
        telemetry_used: dict[str, dict[str, float]] = {}
        link_details: dict[str, dict[str, float]] = {}
        for link in state.links.values():
            if link.status != "up":
                continue
            spec = link.spec
            latency_ms = float(spec.latency_ms)
            bandwidth_mbps = float(spec.bandwidth_mbps)
            latency_measurement = (
                link.measurements.get("network.latency_ms")
                or link.measurements.get("latency_ms")
            )
            bandwidth_measurement = link.measurements.get(
                "network.bandwidth_mbps"
            ) or link.measurements.get("bandwidth_mbps")
            if latency_measurement is not None:
                latency_ms = max(0.0, float(latency_measurement.value))
            if bandwidth_measurement is not None:
                bandwidth_mbps = max(1e-9, float(bandwidth_measurement.value))
            if latency_measurement is not None or bandwidth_measurement is not None:
                telemetry_used[spec.id] = {
                    "latency_ms": latency_ms,
                    "bandwidth_mbps": bandwidth_mbps,
                }
            link_details[spec.id] = {
                "latency_ms": latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
            }
            delay = self._edge_delay(latency_ms, bandwidth_mbps, size_bytes)
            graph.setdefault(spec.source, []).append((spec.target, delay, spec.id))
            if spec.bidirectional:
                graph.setdefault(spec.target, []).append((spec.source, delay, spec.id))

        if explicit_path:
            if explicit_path[0] != source or explicit_path[-1] != target:
                return ModelPrediction(
                    estimate=float("inf"),
                    uncertainty=1.0,
                    metadata={
                        "reachable": False,
                        "path": list(explicit_path),
                        "links": list(explicit_links),
                        "route_bound": True,
                        "route_error": "explicit path endpoints do not match transfer",
                    },
                )
            if len(explicit_links) != max(0, len(explicit_path) - 1):
                return ModelPrediction(
                    estimate=float("inf"),
                    uncertainty=1.0,
                    metadata={
                        "reachable": False,
                        "path": list(explicit_path),
                        "links": list(explicit_links),
                        "route_bound": True,
                        "route_error": "explicit path/link lengths do not match",
                    },
                )
            if any(
                node_id not in state.nodes or state.nodes[node_id].status != "online"
                for node_id in explicit_path
            ):
                return ModelPrediction(
                    estimate=float("inf"),
                    uncertainty=1.0,
                    metadata={
                        "reachable": False,
                        "path": list(explicit_path),
                        "links": list(explicit_links),
                        "route_bound": True,
                        "route_error": "explicit route contains an unavailable node",
                    },
                )
            details = []
            for index, link_id in enumerate(explicit_links):
                link = state.links.get(link_id)
                source_node = explicit_path[index]
                target_node = explicit_path[index + 1]
                if link is None or link.status != "up":
                    return ModelPrediction(
                        estimate=float("inf"),
                        uncertainty=1.0,
                        metadata={
                            "reachable": False,
                            "path": list(explicit_path),
                            "links": list(explicit_links),
                            "route_bound": True,
                            "route_error": f"explicit route link unavailable: {link_id}",
                        },
                    )
                spec = link.spec
                connected = (
                    spec.source == source_node and spec.target == target_node
                ) or (
                    spec.bidirectional
                    and spec.source == target_node
                    and spec.target == source_node
                )
                if not connected:
                    return ModelPrediction(
                        estimate=float("inf"),
                        uncertainty=1.0,
                        metadata={
                            "reachable": False,
                            "path": list(explicit_path),
                            "links": list(explicit_links),
                            "route_bound": True,
                            "route_error": f"explicit route link mismatch: {link_id}",
                        },
                    )
                details.append(link_details[link_id])
            propagation_s = sum(item["latency_ms"] for item in details) / 1000.0
            serialization_s = sum(
                0.0
                if size_bytes <= 0 or item["bandwidth_mbps"] == float("inf")
                else size_bytes * 8 / (item["bandwidth_mbps"] * 1_000_000)
                for item in details
            )
            cost = propagation_s + serialization_s
            hops = max(1, len(explicit_path) - 1)
            bottleneck_mbps = min(
                (item["bandwidth_mbps"] for item in details),
                default=float("inf"),
            )
            return ModelPrediction(
                estimate=cost,
                uncertainty=min(0.8, 0.10 + 0.05 * hops),
                lower=max(0.0, 0.9 * cost),
                upper=1.1 * cost,
                metadata={
                    "path": list(explicit_path),
                    "links": list(explicit_links),
                    "reachable": True,
                    "route_bound": True,
                    "topology_propagation_s": propagation_s,
                    "topology_serialization_s": serialization_s,
                    "bottleneck_mbps": bottleneck_mbps,
                    "link_characteristics": {
                        link_id: link_details[link_id] for link_id in explicit_links
                    },
                    "telemetry": {
                        link_id: telemetry_used[link_id]
                        for link_id in explicit_links
                        if link_id in telemetry_used
                    },
                },
            )

        queue: list[tuple[float, str, tuple[str, ...], tuple[str, ...]]] = [
            (0.0, source, (source,), ())
        ]
        best = {source: 0.0}
        while queue:
            cost, node, path, links = heapq.heappop(queue)
            if node == target:
                hops = max(1, len(path) - 1)
                details = [link_details[link_id] for link_id in links]
                propagation_s = sum(item["latency_ms"] for item in details) / 1000.0
                serialization_s = max(0.0, cost - propagation_s)
                bottleneck_mbps = min(
                    (item["bandwidth_mbps"] for item in details),
                    default=float("inf"),
                )
                return ModelPrediction(
                    estimate=cost,
                    uncertainty=min(0.8, 0.10 + 0.05 * hops),
                    lower=max(0.0, 0.9 * cost),
                    upper=1.1 * cost,
                    metadata={
                        "path": list(path),
                        "links": list(links),
                        "reachable": True,
                        "topology_propagation_s": propagation_s,
                        "topology_serialization_s": serialization_s,
                        "bottleneck_mbps": bottleneck_mbps,
                        "link_characteristics": {
                            link_id: link_details[link_id] for link_id in links
                        },
                        "telemetry": {
                            link_id: telemetry_used[link_id]
                            for link_id in links
                            if link_id in telemetry_used
                        },
                    },
                )
            if cost > best.get(node, float("inf")):
                continue
            for neighbor, edge_cost, link_id in graph.get(node, []):
                new_cost = cost + edge_cost
                if new_cost < best.get(neighbor, float("inf")):
                    best[neighbor] = new_cost
                    heapq.heappush(
                        queue,
                        (new_cost, neighbor, (*path, neighbor), (*links, link_id)),
                    )

        return ModelPrediction(
            estimate=float("inf"),
            uncertainty=1.0,
            metadata={"reachable": False, "path": [], "links": []},
        )

    @staticmethod
    def _contention_wait(
        *,
        links: tuple[str, ...],
        earliest_start: float,
        duration_s: float,
        reservations,
    ) -> float:
        if not links or duration_s <= 0:
            return 0.0
        link_set = set(links)
        relevant = [
            item
            for item in reservations
            if link_set.intersection(str(link) for link in item.get("links", ()))
        ]
        candidates = {float(earliest_start)}
        candidates.update(
            float(item["end"])
            for item in relevant
            if math.isfinite(float(item["end"]))
            and float(item["end"]) >= earliest_start
        )
        for candidate in sorted(candidates):
            end = candidate + duration_s
            if not any(
                float(item["start"]) < end and float(item["end"]) > candidate
                for item in relevant
            ):
                return max(0.0, candidate - earliest_start)
        return float("inf")

    def predict(self, query: Mapping[str, Any], state: ContinuumState) -> ModelPrediction:
        source = str(query["source"])
        target = str(query["target"])
        size_bytes = max(0, int(query.get("size_bytes", 0)))
        baseline = self._topology_prediction(
            source,
            target,
            size_bytes,
            state,
            explicit_path=tuple(str(item) for item in query.get("path", ())),
            explicit_links=tuple(str(item) for item in query.get("links", ())),
        )
        if not math.isfinite(float(baseline.estimate)) or source == target:
            return baseline

        baseline_metadata = dict(baseline.metadata)
        route_links = tuple(
            str(item) for item in baseline_metadata.get("links", ())
        )
        scale_key = self._key(
            source,
            target,
            route_links if baseline_metadata.get("route_bound") else (),
        )
        scale = self._scales.get(scale_key)
        if scale is None or not scale.count:
            transfer_s = float(baseline.estimate)
            uncertainty = baseline.uncertainty
            lower = baseline.lower
            upper = baseline.upper
            metadata = dict(baseline_metadata)
            metadata.update({"calibrated": False, "samples": 0, "scale": 1.0})
        else:
            transfer_s = float(baseline.estimate) * scale.mean
            relative_std = math.sqrt(max(0.0, scale.variance)) / max(scale.mean, 1e-9)
            uncertainty = min(1.0, 0.25 / math.sqrt(scale.count) + relative_std)
            lower = max(0.0, transfer_s * (1 - uncertainty))
            upper = transfer_s * (1 + uncertainty)
            metadata = dict(baseline_metadata)
            metadata.update(
                {
                    "calibrated": True,
                    "samples": scale.count,
                    "scale": scale.mean,
                }
            )

        scale_value = float(metadata.get("scale", 1.0))
        metadata.update(
            {
                "calibration_scale": scale_value,
                "propagation_s": float(metadata.get("topology_propagation_s", 0.0))
                * scale_value,
                "serialization_s": float(metadata.get("topology_serialization_s", 0.0))
                * scale_value,
            }
        )

        earliest_start = float(query.get("earliest_start", state.time))
        contention_wait = self._contention_wait(
            links=tuple(str(item) for item in metadata.get("links", ())),
            earliest_start=earliest_start,
            duration_s=transfer_s,
            reservations=tuple(query.get("reservations", ())),
        )
        if not math.isfinite(contention_wait):
            return ModelPrediction(
                estimate=float("inf"),
                uncertainty=1.0,
                metadata={**metadata, "reachable": False, "contention_wait_s": contention_wait},
            )
        metadata.update(
            {
                "base_transfer_s": transfer_s,
                "contention_wait_s": contention_wait,
                "transfer_start_at": earliest_start + contention_wait,
            }
        )
        return ModelPrediction(
            estimate=contention_wait + transfer_s,
            uncertainty=uncertainty,
            lower=None if lower is None else contention_wait + lower,
            upper=None if upper is None else contention_wait + upper,
            metadata=metadata,
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "calibration_rate": self.calibration_rate,
            "scale_bounds": {
                "minimum": self.min_scale,
                "maximum": self.max_scale,
            },
            "scales": {
                key: {
                    "mean": value.mean,
                    "variance": value.variance,
                    "count": value.count,
                }
                for key, value in self._scales.items()
            },
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.calibration_rate = float(
            snapshot.get("calibration_rate", self.calibration_rate)
        )
        bounds = snapshot.get("scale_bounds")
        if bounds is None:
            # dev25 and earlier snapshots were created with hard-coded bounds.
            # Preserve their exact semantics unless a migrated snapshot opts in
            # to the wider, explicit range.
            self.min_scale, self.max_scale = 0.05, 20.0
        else:
            bounds = dict(bounds)
            self.min_scale, self.max_scale = self._validate_scale_bounds(
                float(bounds["minimum"]), float(bounds["maximum"])
            )
        self._scales = {
            key: _ScaleEstimate(
                mean=float(value["mean"]),
                variance=float(value["variance"]),
                count=int(value["count"]),
            )
            for key, value in snapshot.get("scales", {}).items()
        }
