"""Observation adapters map canonical ContinuumState to researcher tensors."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from darpan.core.state import ComponentInstanceState, ContinuumState


class ObservationAdapter(Protocol):
    def encode(
        self, state: ContinuumState, decision: ComponentInstanceState | None
    ) -> np.ndarray: ...


class PlacementObservation:
    """Compact default observation; custom research can replace it entirely."""

    def __init__(self, resource_names: tuple[str, ...] = ("cpu", "memory")) -> None:
        self.resource_names = resource_names

    def encode(
        self, state: ContinuumState, decision: ComponentInstanceState | None
    ) -> np.ndarray:
        features: list[float] = []
        for node_id in sorted(state.nodes):
            node = state.nodes[node_id]
            features.append(1.0 if node.status == "online" else 0.0)
            for name in self.resource_names:
                resource = node.resources.get(name)
                if resource is None or resource.capacity <= 0:
                    features.extend([0.0, 0.0])
                else:
                    features.extend(
                        [resource.available / resource.capacity, resource.utilization]
                    )
        if decision is None:
            features.extend([0.0] * (1 + len(self.resource_names)))
        else:
            app = state.applications[decision.application_id]
            component = app.component(decision.component_id)
            features.append(component.work_units)
            request_map = {item.name: item.amount for item in component.resources}
            features.extend(request_map.get(name, 0.0) for name in self.resource_names)
        return np.asarray(features, dtype=np.float32)
