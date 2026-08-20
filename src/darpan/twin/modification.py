"""Composable, serializable scenario modifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from darpan.core.measurement import Measurement
from darpan.core.state import ContinuumState


class ScenarioModification(Protocol):
    def apply(self, state: ContinuumState) -> ContinuumState: ...


@dataclass(frozen=True, slots=True)
class RemoveNode:
    node_id: str

    def apply(self, state: ContinuumState) -> ContinuumState:
        nodes = dict(state.nodes)
        if self.node_id not in nodes:
            raise KeyError(self.node_id)
        nodes[self.node_id] = replace(nodes[self.node_id], status="offline")
        return replace(state, nodes=nodes)


@dataclass(frozen=True, slots=True)
class ChangeLink:
    link_id: str
    latency_ms: float | None = None
    bandwidth_mbps: float | None = None

    def apply(self, state: ContinuumState) -> ContinuumState:
        links = dict(state.links)
        old = links[self.link_id]
        changes = {}
        if self.latency_ms is not None:
            changes["latency_ms"] = self.latency_ms
        if self.bandwidth_mbps is not None:
            changes["bandwidth_mbps"] = self.bandwidth_mbps
        links[self.link_id] = replace(old, spec=replace(old.spec, **changes))
        return replace(state, links=links)


@dataclass(frozen=True, slots=True)
class SetMeasurement:
    measurement: Measurement

    def apply(self, state: ContinuumState) -> ContinuumState:
        if self.measurement.target in state.nodes:
            nodes = dict(state.nodes)
            node = nodes[self.measurement.target]
            measurements = dict(node.measurements)
            measurements[self.measurement.name] = self.measurement
            nodes[self.measurement.target] = replace(node, measurements=measurements)
            return replace(state, nodes=nodes)
        measurements = dict(state.measurements)
        measurements[self.measurement.name] = self.measurement
        return replace(state, measurements=measurements)
