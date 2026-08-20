"""Pure-ish reducer from canonical events to ContinuumState."""

from __future__ import annotations

from dataclasses import replace

from darpan.core.codec import application_spec_from_dict, link_spec_from_dict, node_spec_from_dict
from darpan.core.event import Event, EventKind
from darpan.core.measurement import Measurement
from darpan.core.resource import ResourceState
from darpan.core.state import (
    ComponentInstanceState,
    ContinuumState,
    FlowRouteBinding,
    LinkState,
    NodeState,
)


def _dict(mapping):
    return dict(mapping)


def reduce_event(state: ContinuumState, event: Event) -> ContinuumState:
    kind = event.kind
    payload = event.payload
    new_state = replace(state, time=max(state.time, event.event_time))

    if kind == EventKind.NODE_REGISTERED:
        spec = node_spec_from_dict(payload["node"])
        nodes = _dict(new_state.nodes)
        nodes[spec.id] = NodeState(
            id=spec.id,
            tier=spec.tier,
            resources={
                resource.name: ResourceState(
                    name=resource.name,
                    capacity=resource.capacity,
                    unit=resource.unit,
                    attributes=resource.attributes,
                )
                for resource in spec.resources
            },
            capabilities=spec.capabilities,
            labels=spec.labels,
        )
        return replace(new_state, nodes=nodes)

    if kind in {EventKind.NODE_REMOVED, EventKind.NODE_OFFLINE, EventKind.NODE_RECOVERED}:
        nodes = _dict(new_state.nodes)
        node_id = str(payload["node_id"])
        if node_id in nodes:
            status = "online" if kind == EventKind.NODE_RECOVERED else "offline"
            nodes[node_id] = replace(nodes[node_id], status=status)
        return replace(new_state, nodes=nodes)

    if kind in {EventKind.LINK_REGISTERED, EventKind.LINK_CHANGED}:
        spec = link_spec_from_dict(payload["link"])
        links = _dict(new_state.links)
        old = links.get(spec.id)
        links[spec.id] = LinkState(
            spec=spec,
            measurements={} if old is None else old.measurements,
            status="up",
        )
        return replace(new_state, links=links)

    if kind == EventKind.LINK_REMOVED:
        links = _dict(new_state.links)
        link_id = str(payload["link_id"])
        if link_id in links:
            links[link_id] = replace(links[link_id], status="down")
        return replace(new_state, links=links)

    if kind == EventKind.APPLICATION_REGISTERED:
        app = application_spec_from_dict(payload["application"])
        applications = _dict(new_state.applications)
        applications[app.id] = app
        return replace(new_state, applications=applications)

    if kind == EventKind.APPLICATION_SUBMITTED:
        instances = _dict(new_state.application_instances)
        instances[str(payload["instance_id"])] = str(payload["application_id"])
        return replace(new_state, application_instances=instances)

    if kind == EventKind.COMPONENT_CREATED:
        components = _dict(new_state.components)
        item = ComponentInstanceState(
            id=str(payload["instance_id"]),
            application_id=str(payload["application_id"]),
            application_instance_id=str(payload["application_instance_id"]),
            component_id=str(payload["component_id"]),
            status="created",
            created_at=event.event_time,
            replica_index=int(payload.get("replica_index", 0)),
        )
        components[item.id] = item
        return replace(new_state, components=components)

    if kind in {EventKind.COMPONENT_SCALING, EventKind.COMPONENT_SCALED}:
        instance_id = str(payload["instance_id"])
        instance = new_state.components[instance_id]
        key = f"{instance.application_instance_id}:{instance.component_id}"
        targets = _dict(new_state.component_scale_targets)
        targets[key] = int(payload["desired_replicas"])
        pending = set(new_state.component_scale_pending)
        if kind == EventKind.COMPONENT_SCALING:
            pending.add(key)
        else:
            pending.discard(key)
        return replace(
            new_state,
            component_scale_targets=targets,
            component_scale_pending=frozenset(pending),
        )

    if kind == EventKind.FLOW_ROUTED:
        routes = _dict(new_state.flow_routes)
        application_instance_id = str(payload["application_instance_id"])
        source_component_id = str(payload["source_component_id"])
        target_component_id = str(payload["target_component_id"])
        key = ContinuumState.flow_route_key(
            application_instance_id,
            source_component_id,
            target_component_id,
        )
        path = tuple(str(item) for item in payload.get("path", ()))
        if not path:
            routes.pop(key, None)
        else:
            routes[key] = FlowRouteBinding(
                application_id=str(payload["application_id"]),
                application_instance_id=application_instance_id,
                source_component_id=source_component_id,
                target_component_id=target_component_id,
                path=path,
                links=tuple(str(item) for item in payload.get("links", ())),
                updated_at=event.event_time,
                source=event.source,
            )
        return replace(new_state, flow_routes=routes)

    if kind in {
        EventKind.COMPONENT_READY,
        EventKind.COMPONENT_SCHEDULED,
        EventKind.COMPONENT_STARTED,
        EventKind.COMPONENT_MIGRATING,
        EventKind.COMPONENT_RESTARTING,
        EventKind.COMPONENT_RETRYING,
        EventKind.COMPONENT_COMPLETED,
        EventKind.COMPONENT_FAILED,
    }:
        components = _dict(new_state.components)
        instance_id = str(payload["instance_id"])
        old = components[instance_id]
        status = {
            EventKind.COMPONENT_READY: "ready",
            EventKind.COMPONENT_SCHEDULED: "scheduled",
            EventKind.COMPONENT_STARTED: "running",
            EventKind.COMPONENT_MIGRATING: "migrating",
            EventKind.COMPONENT_RESTARTING: "restarting",
            EventKind.COMPONENT_RETRYING: "retrying",
            EventKind.COMPONENT_COMPLETED: "completed",
            EventKind.COMPONENT_FAILED: "failed",
        }[kind]
        updates = {"status": status}
        if "node_id" in payload:
            updates["node_id"] = str(payload["node_id"])
        if kind == EventKind.COMPONENT_STARTED:
            updates["started_at"] = event.event_time
        if kind in {
            EventKind.COMPONENT_MIGRATING,
            EventKind.COMPONENT_RESTARTING,
            EventKind.COMPONENT_RETRYING,
        }:
            updates["attempt"] = int(payload.get("attempt", old.attempt + 1))
            updates["started_at"] = None
            updates["completed_at"] = None
        if kind == EventKind.COMPONENT_RETRYING:
            updates["node_id"] = None
        if kind in {EventKind.COMPONENT_COMPLETED, EventKind.COMPONENT_FAILED}:
            updates["completed_at"] = event.event_time
        components[instance_id] = replace(old, **updates)
        return replace(new_state, components=components)

    if kind in {EventKind.RESOURCE_ALLOCATED, EventKind.RESOURCE_RELEASED}:
        node_id = str(payload["node_id"])
        nodes = _dict(new_state.nodes)
        node = nodes[node_id]
        resources = _dict(node.resources)
        resource_name = str(payload["resource"])
        amount = float(payload["amount"])
        old_resource = resources[resource_name]
        delta = amount if kind == EventKind.RESOURCE_ALLOCATED else -amount
        raw_allocated = max(0.0, old_resource.allocated + delta)
        allocated = (
            raw_allocated
            if old_resource.is_shareable
            else min(old_resource.capacity, raw_allocated)
        )
        resources[resource_name] = replace(old_resource, allocated=allocated)
        nodes[node_id] = replace(node, resources=resources)
        return replace(new_state, nodes=nodes)

    if kind == EventKind.MEASUREMENT_OBSERVED:
        raw = payload["measurement"]
        measurement = Measurement(
            name=str(raw["name"]),
            value=raw["value"],
            unit=str(raw.get("unit", "1")),
            target=raw.get("target"),
            timestamp=float(raw.get("timestamp", event.event_time)),
            source=str(raw.get("source", event.source)),
            uncertainty=raw.get("uncertainty"),
            quality=raw.get("quality"),
            metadata=dict(raw.get("metadata", {})),
        )
        if measurement.target in new_state.nodes:
            nodes = _dict(new_state.nodes)
            node = nodes[measurement.target]
            measurements = _dict(node.measurements)
            measurements[measurement.name] = measurement
            nodes[measurement.target] = replace(node, measurements=measurements)
            return replace(new_state, nodes=nodes)
        if measurement.target in new_state.links:
            links = _dict(new_state.links)
            link = links[measurement.target]
            measurements = _dict(link.measurements)
            measurements[measurement.name] = measurement
            links[measurement.target] = replace(link, measurements=measurements)
            return replace(new_state, links=links)
        measurements = _dict(new_state.measurements)
        key = (
            measurement.name
            if measurement.target is None
            else f"{measurement.target}:{measurement.name}"
        )
        measurements[key] = measurement
        return replace(new_state, measurements=measurements)

    return new_state
