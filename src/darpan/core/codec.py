"""Serialization and configuration codecs for canonical Darpan objects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .application import ApplicationSpec, ComponentSpec, FlowSpec, RetryPolicy
from .data import DataLocation, DataObjectSpec, StorageSpec
from .measurement import Measurement
from .resource import ResourceRequest, ResourceSpec, ResourceState
from .state import (
    ComponentInstanceState,
    ContinuumState,
    FlowRouteBinding,
    LinkState,
    NodeState,
)
from .topology import LinkSpec, NodeSpec, SystemSpec
from .workload import ArrivalSpec, WorkloadSpec


def resource_spec_from_dict(data: Mapping[str, Any]) -> ResourceSpec:
    return ResourceSpec(
        name=str(data["name"]),
        capacity=float(data["capacity"]),
        unit=str(data.get("unit", "count")),
        attributes=dict(data.get("attributes", {})),
    )


def resource_request_from_dict(data: Mapping[str, Any]) -> ResourceRequest:
    return ResourceRequest(
        name=str(data["name"]),
        amount=float(data["amount"]),
        unit=str(data.get("unit", "count")),
    )


def node_spec_from_dict(data: Mapping[str, Any]) -> NodeSpec:
    resources_raw = data.get("resources", [])
    if isinstance(resources_raw, Mapping):
        resources = tuple(
            ResourceSpec(str(name), float(value))
            for name, value in resources_raw.items()
        )
    else:
        resources = tuple(resource_spec_from_dict(item) for item in resources_raw)
    return NodeSpec(
        id=str(data["id"]),
        tier=str(data.get("tier", "edge")),
        resources=resources,
        capabilities=frozenset(data.get("capabilities", [])),
        labels=dict(data.get("labels", {})),
    )


def link_spec_from_dict(data: Mapping[str, Any]) -> LinkSpec:
    return LinkSpec(
        id=str(data.get("id", f'{data["source"]}->{data["target"]}')),
        source=str(data["source"]),
        target=str(data["target"]),
        latency_ms=float(data.get("latency_ms", 0.0)),
        bandwidth_mbps=float(data.get("bandwidth_mbps", float("inf"))),
        bidirectional=bool(data.get("bidirectional", True)),
        labels=dict(data.get("labels", {})),
    )


def system_spec_from_dict(data: Mapping[str, Any]) -> SystemSpec:
    return SystemSpec(
        name=str(data.get("name", "continuum")),
        nodes=tuple(node_spec_from_dict(item) for item in data.get("nodes", [])),
        links=tuple(link_spec_from_dict(item) for item in data.get("links", [])),
    )


def retry_policy_from_dict(data: Any) -> RetryPolicy:
    if data is None:
        return RetryPolicy()
    if isinstance(data, int):
        return RetryPolicy(max_retries=int(data))
    if not isinstance(data, Mapping):
        raise ValueError("component retry must be an integer or mapping")
    raw_on = data.get("on", ())
    if isinstance(raw_on, str):
        raw_on = (raw_on,)
    return RetryPolicy(
        max_retries=int(data.get("max_retries", data.get("retries", 0))),
        on=frozenset(str(item) for item in raw_on),
        backoff_s=float(data.get("backoff_s", 0.0)),
    )


def component_spec_from_dict(data: Mapping[str, Any]) -> ComponentSpec:
    resources_raw = data.get("resources", [])
    if isinstance(resources_raw, Mapping):
        resources = tuple(
            ResourceRequest(str(name), float(value))
            for name, value in resources_raw.items()
        )
    else:
        resources = tuple(resource_request_from_dict(item) for item in resources_raw)
    command = data.get("command", ())
    if isinstance(command, str):
        command = (command,)
    return ComponentSpec(
        id=str(data["id"]),
        kind=str(data.get("kind", "task")),
        image=data.get("image"),
        command=tuple(str(part) for part in command),
        resources=resources,
        work_units=float(data.get("work_units", 1.0)),
        labels=dict(data.get("labels", {})),
        retry=retry_policy_from_dict(data.get("retry")),
    )


def application_spec_from_dict(data: Mapping[str, Any]) -> ApplicationSpec:
    components_raw = data.get("components", [])
    if isinstance(components_raw, Mapping):
        components = tuple(
            component_spec_from_dict({"id": key, **dict(value)})
            for key, value in components_raw.items()
        )
    else:
        components = tuple(component_spec_from_dict(item) for item in components_raw)
    flows_raw = data.get("flows", data.get("workflow", []))
    flows: list[FlowSpec] = []
    for item in flows_raw:
        if isinstance(item, (list, tuple)):
            source, target = item[:2]
            flows.append(FlowSpec(str(source), str(target)))
        else:
            flows.append(
                FlowSpec(
                    source=str(item["source"]),
                    target=str(item["target"]),
                    data_size_bytes=int(item.get("data_size_bytes", 0)),
                    kind=str(item.get("kind", "data")),
                    artifact=(
                        None if item.get("artifact") is None else str(item["artifact"])
                    ),
                    target_path=(
                        None
                        if item.get("target_path") is None
                        else str(item["target_path"])
                    ),
                    labels=dict(item.get("labels", {})),
                )
            )
    return ApplicationSpec(
        id=str(data.get("id", data.get("name", "application"))),
        components=components,
        flows=tuple(flows),
        labels=dict(data.get("labels", {})),
    )


def arrival_spec_from_dict(data: Mapping[str, Any]) -> ArrivalSpec:
    return ArrivalSpec(
        application_id=str(data.get("application_id", data.get("application", ""))),
        at_s=float(data.get("at_s", data.get("at", 0.0))),
        count=int(data.get("count", 1)),
    )


def workload_spec_from_dict(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> WorkloadSpec:
    """Decode a workload containing inline apps and/or application files.

    ``applications`` accepts either normal application mappings or path
    strings.  Paths are resolved relative to the workload file, which makes a
    workload directory relocatable and suitable for paper artifacts.
    """

    root = Path(base_dir or ".")
    applications: list[ApplicationSpec] = []
    for item in data.get("applications", []):
        if isinstance(item, (str, Path)):
            applications.append(load_application(root / item))
        elif isinstance(item, Mapping):
            if "file" in item:
                applications.append(load_application(root / str(item["file"])))
            else:
                applications.append(application_spec_from_dict(item))
        else:
            raise ValueError("workload applications must be mappings or file paths")

    arrivals = tuple(arrival_spec_from_dict(item) for item in data.get("arrivals", []))
    return WorkloadSpec(
        applications=tuple(applications),
        arrivals=arrivals,
        name=str(data.get("name", "workload")),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping in {path}")
    return loaded


def load_system(path: str | Path) -> SystemSpec:
    return system_spec_from_dict(load_yaml(path))


def load_application(path: str | Path) -> ApplicationSpec:
    return application_spec_from_dict(load_yaml(path))


def load_workload(path: str | Path) -> WorkloadSpec:
    source = Path(path)
    return workload_spec_from_dict(load_yaml(source), base_dir=source.parent)


def measurement_from_dict(data: Mapping[str, Any]) -> Measurement:
    return Measurement(
        name=str(data["name"]),
        value=data.get("value"),
        unit=str(data.get("unit", "1")),
        target=data.get("target"),
        timestamp=float(data.get("timestamp", 0.0)),
        source=str(data.get("source", "unknown")),
        uncertainty=(
            None if data.get("uncertainty") is None else float(data["uncertainty"])
        ),
        quality=None if data.get("quality") is None else float(data["quality"]),
        metadata=dict(data.get("metadata", {})),
    )


def continuum_state_from_dict(data: Mapping[str, Any]) -> ContinuumState:
    nodes: dict[str, NodeState] = {}
    for node_id, raw in data.get("nodes", {}).items():
        resources = {
            name: ResourceState(
                name=str(item.get("name", name)),
                capacity=float(item["capacity"]),
                allocated=float(item.get("allocated", 0.0)),
                unit=str(item.get("unit", "count")),
                attributes=dict(item.get("attributes", {})),
            )
            for name, item in raw.get("resources", {}).items()
        }
        measurements = {
            name: measurement_from_dict(item)
            for name, item in raw.get("measurements", {}).items()
        }
        nodes[str(node_id)] = NodeState(
            id=str(raw.get("id", node_id)),
            tier=str(raw.get("tier", "edge")),
            resources=resources,
            capabilities=frozenset(raw.get("capabilities", [])),
            labels=dict(raw.get("labels", {})),
            measurements=measurements,
            status=str(raw.get("status", "online")),
        )

    links = {
        str(link_id): LinkState(
            spec=link_spec_from_dict(raw["spec"]),
            measurements={
                name: measurement_from_dict(item)
                for name, item in raw.get("measurements", {}).items()
            },
            status=str(raw.get("status", "up")),
        )
        for link_id, raw in data.get("links", {}).items()
    }
    applications = {
        str(app_id): application_spec_from_dict(raw)
        for app_id, raw in data.get("applications", {}).items()
    }
    components = {
        str(instance_id): ComponentInstanceState(
            id=str(raw.get("id", instance_id)),
            application_id=str(raw["application_id"]),
            application_instance_id=str(raw["application_instance_id"]),
            component_id=str(raw["component_id"]),
            status=str(raw.get("status", "created")),
            node_id=None if raw.get("node_id") is None else str(raw["node_id"]),
            created_at=float(raw.get("created_at", 0.0)),
            started_at=(
                None if raw.get("started_at") is None else float(raw["started_at"])
            ),
            completed_at=(
                None
                if raw.get("completed_at") is None
                else float(raw["completed_at"])
            ),
            attempt=int(raw.get("attempt", 0)),
            replica_index=int(raw.get("replica_index", 0)),
            metadata=dict(raw.get("metadata", {})),
        )
        for instance_id, raw in data.get("components", {}).items()
    }
    measurements = {
        str(name): measurement_from_dict(raw)
        for name, raw in data.get("measurements", {}).items()
    }
    data_objects = {
        str(data_id): DataObjectSpec(
            id=str(raw.get("id", data_id)),
            size_bytes=int(raw["size_bytes"]),
            labels=dict(raw.get("labels", {})),
        )
        for data_id, raw in data.get("data_objects", {}).items()
    }
    storages = {
        str(storage_id): StorageSpec(
            id=str(raw.get("id", storage_id)),
            node_id=str(raw["node_id"]),
            capacity_bytes=int(raw["capacity_bytes"]),
            kind=str(raw.get("kind", "local")),
            labels=dict(raw.get("labels", {})),
        )
        for storage_id, raw in data.get("storages", {}).items()
    }
    data_locations = {
        str(data_id): DataLocation(
            data_id=str(raw["data_id"]),
            storage_id=str(raw["storage_id"]),
            node_id=str(raw["node_id"]),
        )
        for data_id, raw in data.get("data_locations", {}).items()
    }
    flow_routes = {
        str(key): FlowRouteBinding(
            application_id=str(raw["application_id"]),
            application_instance_id=str(raw["application_instance_id"]),
            source_component_id=str(raw["source_component_id"]),
            target_component_id=str(raw["target_component_id"]),
            path=tuple(str(item) for item in raw.get("path", [])),
            links=tuple(str(item) for item in raw.get("links", [])),
            updated_at=float(raw.get("updated_at", 0.0)),
            source=str(raw.get("source", "controller")),
        )
        for key, raw in data.get("flow_routes", {}).items()
    }
    return ContinuumState(
        time=float(data.get("time", 0.0)),
        nodes=nodes,
        links=links,
        applications=applications,
        application_instances={
            str(key): str(value)
            for key, value in data.get("application_instances", {}).items()
        },
        components=components,
        measurements=measurements,
        data_objects=data_objects,
        storages=storages,
        data_locations=data_locations,
        flow_routes=flow_routes,
        component_scale_targets={
            str(key): int(value)
            for key, value in data.get("component_scale_targets", {}).items()
        },
        component_scale_pending=frozenset(
            str(item) for item in data.get("component_scale_pending", [])
        ),
        extensions=dict(data.get("extensions", {})),
    )
