"""Explicit logical-flow routing contracts shared by Real and Twin runtimes."""

from __future__ import annotations

from darpan.core.state import ContinuumState, FlowRouteBinding


def _link_connects(link, source: str, target: str) -> bool:
    spec = link.spec
    return (spec.source == source and spec.target == target) or (
        spec.bidirectional and spec.source == target and spec.target == source
    )


def resolve_route_links(
    state: ContinuumState,
    path: tuple[str, ...],
    requested_links: tuple[str, ...] = (),
) -> tuple[tuple[str, ...] | None, str | None]:
    """Resolve and validate the concrete link sequence for a node path."""

    if not path:
        if requested_links:
            return None, "automatic routing cannot specify explicit links"
        return (), None
    for node_id in path:
        if node_id not in state.nodes:
            return None, f"unknown route node: {node_id}"
        if state.nodes[node_id].status != "online":
            return None, f"route node is not online: {node_id}"
    if len(path) == 1:
        if requested_links:
            return None, "node-local route cannot contain links"
        return (), None
    if not state.links:
        return None, "explicit routes require a declared network topology"
    if requested_links and len(requested_links) != len(path) - 1:
        return None, "explicit route link count must equal len(path) - 1"

    resolved: list[str] = []
    for index, (source, target) in enumerate(zip(path, path[1:], strict=False)):
        if requested_links:
            link_id = requested_links[index]
            link = state.links.get(link_id)
            if link is None:
                return None, f"unknown route link: {link_id}"
            if link.status != "up":
                return None, f"route link is not up: {link_id}"
            if not _link_connects(link, source, target):
                return None, f"route link {link_id} does not connect {source} -> {target}"
            resolved.append(link_id)
            continue

        candidates = sorted(
            link_id
            for link_id, link in state.links.items()
            if link.status == "up" and _link_connects(link, source, target)
        )
        if not candidates:
            return None, f"no up link connects route hop {source} -> {target}"
        if len(candidates) > 1:
            return None, (
                f"route hop {source} -> {target} is ambiguous; specify link IDs"
            )
        resolved.append(candidates[0])
    return tuple(resolved), None


def flow_route_feasibility(
    state: ContinuumState,
    application_instance_id: str,
    source_component_id: str,
    target_component_id: str,
    path: tuple[str, ...],
    requested_links: tuple[str, ...] = (),
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Validate one logical-flow route binding against current canonical state."""

    application_id = state.application_instances.get(application_instance_id)
    if application_id is None:
        return False, f"unknown application instance: {application_instance_id}", ()
    app = state.applications[application_id]
    if app.flow(source_component_id, target_component_id) is None:
        return False, (
            f"application {application_id} has no flow "
            f"{source_component_id} -> {target_component_id}"
        ), ()
    source_id = f"{application_instance_id}:{source_component_id}"
    target_id = f"{application_instance_id}:{target_component_id}"
    if source_id not in state.components or target_id not in state.components:
        return False, "flow component instances have not been created", ()

    source_replicas = state.component_replicas(source_id, include_terminal=False)
    target_replicas = state.component_replicas(target_id, include_terminal=False)
    if len(source_replicas) > 1 or len(target_replicas) > 1:
        return False, (
            "explicit logical-flow routing currently requires unscaled endpoints; "
            "per-replica load balancing is a separate control contract"
        ), ()

    if not path:
        if state.flow_route(
            application_instance_id,
            source_component_id,
            target_component_id,
        ) is None:
            return False, "flow already uses automatic topology routing", ()
        if requested_links:
            return False, "automatic routing cannot specify explicit links", ()
        return True, None, ()

    source_node = state.components[source_id].node_id
    target_node = state.components[target_id].node_id
    if source_node is None:
        return False, "route source component has no node yet", ()
    if path[0] != source_node:
        return False, f"route must start at source node {source_node}", ()
    if target_node is not None and path[-1] != target_node:
        return False, f"route must end at target node {target_node}", ()
    links, reason = resolve_route_links(state, path, requested_links)
    if links is None:
        return False, reason, ()
    return True, None, links


def route_endpoint_feasibility(
    state: ContinuumState,
    instance_id: str,
    node_id: str,
) -> tuple[bool, str | None]:
    """Require placements/migrations to honor existing explicit flow endpoints."""

    instance = state.components[instance_id]
    app = state.applications[instance.application_id]
    app_instance = instance.application_instance_id
    for flow in app.flows:
        binding: FlowRouteBinding | None = None
        expected: str | None = None
        if flow.target == instance.component_id:
            binding = state.flow_route(app_instance, flow.source, flow.target)
            if binding is not None:
                expected = binding.path[-1]
        elif flow.source == instance.component_id:
            binding = state.flow_route(app_instance, flow.source, flow.target)
            if binding is not None:
                expected = binding.path[0]
        if binding is None:
            continue
        if node_id != expected:
            return False, (
                f"placement conflicts with explicit route endpoint: "
                f"need {expected}, got {node_id}"
            )
        links, reason = resolve_route_links(state, binding.path, binding.links)
        if links is None:
            return False, f"explicit route is unavailable: {reason}"
    return True, None
