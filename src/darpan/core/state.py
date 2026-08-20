"""Materialized Continuum state derived from the append-only EventLog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .application import ApplicationSpec
from .data import DataLocation, DataObjectSpec, StorageSpec
from .measurement import Measurement
from .resource import ResourceState
from .topology import LinkSpec


@dataclass(frozen=True, slots=True)
class NodeState:
    id: str
    tier: str
    resources: Mapping[str, ResourceState] = field(default_factory=dict)
    capabilities: frozenset[str] = frozenset()
    labels: Mapping[str, str] = field(default_factory=dict)
    measurements: Mapping[str, Measurement] = field(default_factory=dict)
    status: str = "online"

    def resource(self, name: str) -> ResourceState | None:
        return self.resources.get(name)

    def measurement(self, name: str) -> Measurement | None:
        return self.measurements.get(name)

    def effective_resource_capacity(self, name: str) -> float:
        """Return declared capacity capped by the latest physical measurement."""

        resource = self.resources.get(name)
        if resource is None:
            return 0.0
        capacity = max(0.0, float(resource.capacity))
        measured = self.measurements.get(f"resource.{name}.capacity")
        if name == "cpu":
            measured = self.measurements.get("compute.cpu_capacity") or measured
        if measured is None:
            return capacity
        return max(0.0, min(capacity, float(measured.value)))

    def effective_resource_available(self, name: str) -> float:
        resource = self.resources.get(name)
        if resource is None:
            return 0.0
        return max(
            0.0,
            self.effective_resource_capacity(name) - float(resource.allocated),
        )

    def can_admit_resource(self, name: str, amount: float) -> bool:
        resource = self.resources.get(name)
        if resource is None or amount < 0:
            return False
        capacity = self.effective_resource_capacity(name)
        if resource.is_shareable:
            return amount <= capacity + 1e-12
        return amount <= self.effective_resource_available(name) + 1e-12


@dataclass(frozen=True, slots=True)
class LinkState:
    spec: LinkSpec
    measurements: Mapping[str, Measurement] = field(default_factory=dict)
    status: str = "up"


@dataclass(frozen=True, slots=True)
class FlowRouteBinding:
    application_id: str
    application_instance_id: str
    source_component_id: str
    target_component_id: str
    path: tuple[str, ...]
    links: tuple[str, ...]
    updated_at: float
    source: str = "controller"


@dataclass(frozen=True, slots=True)
class ComponentInstanceState:
    id: str
    application_id: str
    application_instance_id: str
    component_id: str
    status: str = "created"
    node_id: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    attempt: int = 0
    replica_index: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContinuumState:
    time: float = 0.0
    nodes: Mapping[str, NodeState] = field(default_factory=dict)
    links: Mapping[str, LinkState] = field(default_factory=dict)
    applications: Mapping[str, ApplicationSpec] = field(default_factory=dict)
    application_instances: Mapping[str, str] = field(default_factory=dict)
    components: Mapping[str, ComponentInstanceState] = field(default_factory=dict)
    measurements: Mapping[str, Measurement] = field(default_factory=dict)
    data_objects: Mapping[str, DataObjectSpec] = field(default_factory=dict)
    storages: Mapping[str, StorageSpec] = field(default_factory=dict)
    data_locations: Mapping[str, DataLocation] = field(default_factory=dict)
    flow_routes: Mapping[str, FlowRouteBinding] = field(default_factory=dict)
    component_scale_targets: Mapping[str, int] = field(default_factory=dict)
    component_scale_pending: frozenset[str] = frozenset()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def ready_components(self) -> tuple[ComponentInstanceState, ...]:
        return tuple(
            component
            for component in self.components.values()
            if component.status == "ready"
        )

    def component(self, instance_id: str) -> ComponentInstanceState:
        return self.components[instance_id]

    def component_replicas(
        self,
        instance_id: str,
        *,
        include_terminal: bool = True,
    ) -> tuple[ComponentInstanceState, ...]:
        """Return deterministic members of one logical component replica set."""

        instance = self.components[instance_id]
        replicas = (
            item
            for item in self.components.values()
            if item.application_instance_id == instance.application_instance_id
            and item.component_id == instance.component_id
            and (include_terminal or item.status not in {"completed", "failed"})
        )
        return tuple(sorted(replicas, key=lambda item: (item.replica_index, item.id)))

    def component_scale_key(self, instance_id: str) -> str:
        instance = self.components[instance_id]
        return f"{instance.application_instance_id}:{instance.component_id}"

    def desired_replicas(self, instance_id: str) -> int:
        return int(self.component_scale_targets.get(self.component_scale_key(instance_id), 1))

    def scale_pending(self, instance_id: str) -> bool:
        return self.component_scale_key(instance_id) in self.component_scale_pending

    @staticmethod
    def flow_route_key(
        application_instance_id: str,
        source_component_id: str,
        target_component_id: str,
    ) -> str:
        return (
            f"{application_instance_id}|{source_component_id}"
            f"->{target_component_id}"
        )

    def flow_route(
        self,
        application_instance_id: str,
        source_component_id: str,
        target_component_id: str,
    ) -> FlowRouteBinding | None:
        return self.flow_routes.get(
            self.flow_route_key(
                application_instance_id,
                source_component_id,
                target_component_id,
            )
        )
