"""Tiny built-in policies used by quickstarts and smoke tests."""

from __future__ import annotations

import heapq
import math

from darpan.core.action import Action
from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState
from darpan.runtime.action_plan import placement_feasibility

_PLACEMENT_REEVALUATION_KINDS = frozenset(
    {
        EventKind.NODE_REGISTERED,
        EventKind.NODE_RECOVERED,
        EventKind.LINK_REGISTERED,
        EventKind.LINK_CHANGED,
        EventKind.RESOURCE_RELEASED,
        EventKind.MEASUREMENT_OBSERVED,
    }
)


def _placement_candidates(state: ContinuumState, trigger: Event):
    if trigger.kind == EventKind.COMPONENT_READY and trigger.subject is not None:
        instance = state.components.get(trigger.subject)
        return () if instance is None or instance.status != "ready" else (instance,)
    if trigger.kind not in _PLACEMENT_REEVALUATION_KINDS:
        return ()
    return tuple(sorted(state.ready_components(), key=lambda item: item.id))


class FirstFitPolicy:
    def decide(self, state: ContinuumState, trigger: Event):
        actions = []
        for instance in _placement_candidates(state, trigger):
            for node_id in sorted(state.nodes):
                feasible, _ = placement_feasibility(state, instance.id, node_id)
                if feasible:
                    actions.append(Action.place(instance.id, node_id, source="first-fit"))
                    break
        if not actions:
            return None
        return actions[0] if len(actions) == 1 else actions


class RoundRobinPolicy:
    def __init__(self) -> None:
        self._index = 0

    def decide(self, state: ContinuumState, trigger: Event):
        actions = []
        nodes = sorted(state.nodes)
        if not nodes:
            return None
        for instance in _placement_candidates(state, trigger):
            for offset in range(len(nodes)):
                index = (self._index + offset) % len(nodes)
                node_id = nodes[index]
                feasible, _ = placement_feasibility(state, instance.id, node_id)
                if feasible:
                    self._index = (index + 1) % len(nodes)
                    actions.append(
                        Action.place(instance.id, node_id, source="round-robin")
                    )
                    break
        if not actions:
            return None
        return actions[0] if len(actions) == 1 else actions


def _link_transfer_cost_s(link, data_size_bytes: int) -> float:
    latency = link.measurements.get("network.latency_ms")
    bandwidth = link.measurements.get("network.bandwidth_mbps")
    latency_ms = (
        float(link.spec.latency_ms)
        if latency is None
        else max(0.0, float(latency.value))
    )
    bandwidth_mbps = (
        float(link.spec.bandwidth_mbps)
        if bandwidth is None
        else max(1e-12, float(bandwidth.value))
    )
    serialization = 0.0
    if math.isfinite(bandwidth_mbps):
        serialization = max(0, int(data_size_bytes)) * 8.0 / (bandwidth_mbps * 1_000_000.0)
    return latency_ms / 1000.0 + serialization


def _shortest_transfer_cost_s(
    state: ContinuumState,
    source: str,
    target: str,
    data_size_bytes: int,
) -> float:
    if source == target:
        return 0.0
    if not state.links:
        return 0.0
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for link in state.links.values():
        if link.status != "up":
            continue
        cost = _link_transfer_cost_s(link, data_size_bytes)
        adjacency.setdefault(link.spec.source, []).append((link.spec.target, cost))
        if link.spec.bidirectional:
            adjacency.setdefault(link.spec.target, []).append((link.spec.source, cost))
    queue = [(0.0, source)]
    best = {source: 0.0}
    while queue:
        cost, node = heapq.heappop(queue)
        if node == target:
            return cost
        if cost > best.get(node, float("inf")) + 1e-12:
            continue
        for neighbor, edge_cost in adjacency.get(node, ()):
            if neighbor not in state.nodes or state.nodes[neighbor].status != "online":
                continue
            candidate = cost + edge_cost
            if candidate + 1e-12 < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return float("inf")


def _input_transfer_score(state: ContinuumState, instance_id: str, node_id: str) -> float:
    instance = state.components[instance_id]
    app = state.applications[instance.application_id]
    score = 0.0
    for predecessor_id in app.predecessors(instance.component_id):
        predecessor_key = f"{instance.application_instance_id}:{predecessor_id}"
        predecessor = state.components.get(predecessor_key)
        if predecessor is None or predecessor.node_id is None:
            continue
        flow = app.flow(predecessor_id, instance.component_id)
        data_size = 0 if flow is None else flow.data_size_bytes
        score += _shortest_transfer_cost_s(
            state,
            predecessor.node_id,
            node_id,
            data_size,
        )
    return score


class LatencyAwarePolicy:
    """Place ready work where predecessor input transfer cost is lowest.

    The score uses only current canonical topology/measurements and declared flow
    sizes. It does not consult Twin future state or execution-time predictions.
    """

    def decide(self, state: ContinuumState, trigger: Event):
        actions = []
        for instance in _placement_candidates(state, trigger):
            candidates = []
            for node_id in sorted(state.nodes):
                feasible, _ = placement_feasibility(state, instance.id, node_id)
                if not feasible:
                    continue
                score = _input_transfer_score(state, instance.id, node_id)
                candidates.append((score, node_id))
            if not candidates:
                continue
            _, node_id = min(candidates, key=lambda item: (item[0], item[1]))
            actions.append(Action.place(instance.id, node_id, source="latency-aware"))
        if not actions:
            return None
        return actions[0] if len(actions) == 1 else actions
