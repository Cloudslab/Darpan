"""Action validation and arbitration for multiple autonomous controllers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from darpan.core.action import Action
from darpan.core.state import ContinuumState

from .routing import flow_route_feasibility, route_endpoint_feasibility


@dataclass(frozen=True, slots=True)
class ValidationResult:
    action: Action
    accepted: bool
    reason: str | None = None


def nodes_reachable(state: ContinuumState, source: str, target: str) -> bool:
    """Return whether two online nodes are connected by current up links."""

    if source == target:
        return source in state.nodes and state.nodes[source].status == "online"
    if source not in state.nodes or target not in state.nodes:
        return False
    if state.nodes[source].status != "online" or state.nodes[target].status != "online":
        return False
    # A system with no declared links uses the backward-compatible implicit
    # fabric contract. Once any links are declared, the topology is explicit
    # and reachability must follow currently-up links.
    if not state.links:
        return True
    adjacency: dict[str, set[str]] = {}
    for link in state.links.values():
        if link.status != "up":
            continue
        spec = link.spec
        adjacency.setdefault(spec.source, set()).add(spec.target)
        if spec.bidirectional:
            adjacency.setdefault(spec.target, set()).add(spec.source)
    frontier = [source]
    visited = {source}
    while frontier:
        node_id = frontier.pop()
        for neighbor in adjacency.get(node_id, ()):
            if neighbor in visited:
                continue
            if neighbor not in state.nodes or state.nodes[neighbor].status != "online":
                continue
            if neighbor == target:
                return True
            visited.add(neighbor)
            frontier.append(neighbor)
    return False


def placement_feasibility(
    state: ContinuumState,
    instance_id: str,
    node_id: str,
) -> tuple[bool, str | None]:
    """Shared placement feasibility used by policies, RL masks, and validators."""

    if instance_id not in state.components:
        return False, f"unknown component instance: {instance_id}"
    if node_id not in state.nodes:
        return False, f"unknown node: {node_id}"
    node = state.nodes[node_id]
    if node.status != "online":
        return False, f"node is not online: {node_id}"
    instance = state.components[instance_id]
    if instance.status != "ready":
        return False, f"component is not ready: {instance_id}"
    route_ok, route_reason = route_endpoint_feasibility(state, instance_id, node_id)
    if not route_ok:
        return False, route_reason
    app = state.applications[instance.application_id]
    component = app.component(instance.component_id)
    for request in component.resources:
        resource = node.resources.get(request.name)
        if resource is None:
            return False, f"node {node_id} has no resource {request.name}"
        if not node.can_admit_resource(request.name, request.amount):
            return False, (
                f"insufficient {request.name} on {node_id}: "
                f"need {request.amount}, "
                f"available {node.effective_resource_available(request.name)}"
            )
    for predecessor_id in app.predecessors(instance.component_id):
        predecessor = state.components.get(
            f"{instance.application_instance_id}:{predecessor_id}"
        )
        if predecessor is None or predecessor.node_id is None:
            continue
        if not nodes_reachable(state, predecessor.node_id, node_id):
            return False, (
                "no data path for placement: "
                f"{predecessor.node_id} -> {node_id}"
            )
    return True, None


def scale_feasibility(
    state: ContinuumState,
    instance_id: str,
    desired_replicas: int,
) -> tuple[bool, str | None]:
    """Validate one horizontal scale request for a stable long-running set."""

    if instance_id not in state.components:
        return False, f"unknown component instance: {instance_id}"
    if desired_replicas < 1:
        return False, (
            "desired replicas must be at least 1; use STOP to terminate a service"
        )
    instance = state.components[instance_id]
    app = state.applications[instance.application_id]
    component = app.component(instance.component_id)
    if not component.is_long_running:
        return False, f"component is not long-running: {instance_id}"
    for flow in app.flows:
        if flow.source != instance.component_id and flow.target != instance.component_id:
            continue
        if state.flow_route(
            instance.application_instance_id,
            flow.source,
            flow.target,
        ) is not None:
            return False, (
                "cannot scale an endpoint with an explicit logical-flow route; "
                "clear the route before changing replica count"
            )
    active = state.component_replicas(instance_id, include_terminal=False)
    if not active:
        return False, f"component has no active replicas: {instance_id}"
    unstable = [item.id for item in active if item.status != "running"]
    if unstable:
        return False, "replica set is not converged: " + ", ".join(sorted(unstable))
    if len(active) == desired_replicas:
        return False, f"component already has {desired_replicas} active replicas"
    return True, None


class DefaultActionValidator:
    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]:
        if action.kind == "component.place":
            instance_id = str(action.payload.get("instance_id", ""))
            node_id = str(action.payload.get("node_id", ""))
            if instance_id not in state.components:
                return False, f"unknown component instance: {instance_id}"
            if node_id not in state.nodes:
                return False, f"unknown node: {node_id}"
            if state.nodes[node_id].status != "online":
                return False, f"node is not online: {node_id}"
            component = state.components[instance_id]
            if component.status != "ready":
                return False, f"component is not ready: {instance_id}"
        if action.kind == "component.restart":
            instance_id = str(action.payload.get("instance_id", ""))
            if instance_id not in state.components:
                return False, f"unknown component instance: {instance_id}"
            instance = state.components[instance_id]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            if not component.is_long_running:
                return False, f"component is not long-running: {instance_id}"
            if instance.status != "running":
                return False, f"component is not running: {instance_id}"
            node_id = instance.node_id
            if node_id is None or node_id not in state.nodes:
                return False, f"component has no valid node: {instance_id}"
            node = state.nodes[node_id]
            if node.status != "online":
                return False, f"node is not online: {node_id}"
            for request in component.resources:
                resource = node.resources.get(request.name)
                if resource is None:
                    return False, f"node {node_id} has no resource {request.name}"
                if request.amount > node.effective_resource_capacity(request.name) + 1e-12:
                    return False, (
                        f"restart request exceeds effective {request.name} capacity "
                        f"on {node_id}: need {request.amount}, capacity "
                        f"{node.effective_resource_capacity(request.name)}"
                    )
            for predecessor_id in app.predecessors(instance.component_id):
                predecessor = state.components.get(
                    f"{instance.application_instance_id}:{predecessor_id}"
                )
                if predecessor is None or predecessor.node_id is None:
                    continue
                if not nodes_reachable(state, predecessor.node_id, node_id):
                    return False, (
                        "no data path for restart: "
                        f"{predecessor.node_id} -> {node_id}"
                    )
        if action.kind == "service.scale":
            instance_id = str(action.payload.get("instance_id", ""))
            try:
                desired = int(action.payload.get("replicas", 0))
            except (TypeError, ValueError):
                return False, "desired replicas must be an integer"
            return scale_feasibility(state, instance_id, desired)
        if action.kind == "network.route":
            application_instance_id = str(
                action.payload.get("application_instance_id", "")
            )
            source_component_id = str(action.payload.get("source_component_id", ""))
            target_component_id = str(action.payload.get("target_component_id", ""))
            raw_path = action.payload.get("path", ())
            raw_links = action.payload.get("links", ())
            if isinstance(raw_path, str) or not isinstance(raw_path, (list, tuple)):
                return False, "route path must be a sequence of node IDs"
            if isinstance(raw_links, str) or not isinstance(raw_links, (list, tuple)):
                return False, "route links must be a sequence of link IDs"
            ok, reason, _ = flow_route_feasibility(
                state,
                application_instance_id,
                source_component_id,
                target_component_id,
                tuple(str(item) for item in raw_path),
                tuple(str(item) for item in raw_links),
            )
            return ok, reason
        if action.kind == "component.stop":
            instance_id = str(action.payload.get("instance_id", ""))
            if instance_id not in state.components:
                return False, f"unknown component instance: {instance_id}"
            instance = state.components[instance_id]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            if not component.is_long_running:
                return False, f"component is not long-running: {instance_id}"
            if instance.status != "running":
                return False, f"component is not running: {instance_id}"
        if action.kind == "component.migrate":
            instance_id = str(action.payload.get("instance_id", ""))
            node_id = str(action.payload.get("node_id", ""))
            if instance_id not in state.components:
                return False, f"unknown component instance: {instance_id}"
            if node_id not in state.nodes:
                return False, f"unknown node: {node_id}"
            instance = state.components[instance_id]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            if not component.is_long_running:
                return False, f"component is not long-running: {instance_id}"
            if instance.status != "running":
                return False, f"component is not running: {instance_id}"
            if instance.node_id == node_id:
                return False, f"component is already on node: {node_id}"
            mode = str(action.payload.get("mode", "restart"))
            if mode != "restart":
                return False, f"unsupported migration mode: {mode}"
            if state.nodes[node_id].status != "online":
                return False, f"node is not online: {node_id}"
        return True, None


class BackendCapabilityValidator:
    """Reject action kinds a backend explicitly declares unsupported.

    ``ActionKind`` is the canonical vocabulary, not a promise that every
    backend implements every action.  Rejecting before dispatch keeps Real and
    Twin failure semantics deterministic and avoids late ``NotImplementedError``
    exceptions after an action was already recorded as accepted.
    """

    def __init__(self, backend) -> None:
        supported = getattr(backend, "supported_action_kinds", None)
        self.supported = None if supported is None else frozenset(supported)

    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]:
        del state
        if self.supported is None or action.kind in self.supported:
            return True, None
        return False, f"backend does not support action kind: {action.kind}"


class BackendNodeValidator:
    """Reject placements that a strict physical backend cannot execute.

    Local Real runtimes intentionally retain a default executor. Physical
    cluster sessions instead expose ``allowed_node_ids`` so a topology typo
    can never silently fall back to executing work on the controller host.
    """

    def __init__(self, backend) -> None:
        allowed = getattr(backend, "allowed_node_ids", None)
        self.allowed = None if allowed is None else frozenset(str(item) for item in allowed)

    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]:
        if self.allowed is None:
            return True, None
        node_id: str | None = None
        if action.kind in {"component.place", "component.migrate"}:
            node_id = str(action.payload.get("node_id", ""))
        elif action.kind in {"component.restart", "component.stop"}:
            instance_id = str(action.payload.get("instance_id", ""))
            instance = state.components.get(instance_id)
            node_id = None if instance is None else instance.node_id
        if node_id is None or not node_id:
            return True, None
        if node_id not in self.allowed:
            return False, f"physical cluster has no executor for node: {node_id}"
        return True, None


class ResourceFeasibilityValidator:
    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]:
        if action.kind not in {"component.place", "component.migrate"}:
            return True, None
        instance_id = str(action.payload.get("instance_id", ""))
        node_id = str(action.payload.get("node_id", ""))
        if instance_id not in state.components or node_id not in state.nodes:
            return True, None  # structural validator reports this first
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        node = state.nodes[node_id]
        for request in component.resources:
            resource = node.resources.get(request.name)
            if resource is None:
                return False, f"node {node_id} has no resource {request.name}"
            if not node.can_admit_resource(request.name, request.amount):
                return False, (
                    f"insufficient {request.name} on {node_id}: "
                    f"need {request.amount}, "
                    f"available {node.effective_resource_available(request.name)}"
                )
        return True, None


class DataReachabilityValidator:
    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]:
        if action.kind not in {"component.place", "component.migrate"}:
            return True, None
        instance_id = str(action.payload.get("instance_id", ""))
        node_id = str(action.payload.get("node_id", ""))
        if instance_id not in state.components or node_id not in state.nodes:
            return True, None
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        for predecessor_id in app.predecessors(instance.component_id):
            predecessor = state.components.get(
                f"{instance.application_instance_id}:{predecessor_id}"
            )
            if predecessor is None or predecessor.node_id is None:
                continue
            if not nodes_reachable(state, predecessor.node_id, node_id):
                return False, (
                    "no data path for placement: "
                    f"{predecessor.node_id} -> {node_id}"
                )
        route_ok, route_reason = route_endpoint_feasibility(
            state,
            instance_id,
            node_id,
        )
        if not route_ok:
            return False, route_reason
        return True, None


class PriorityArbiter:
    """Pick highest-priority non-conflicting action for each target."""

    def select(self, actions: Sequence[Action], state: ContinuumState) -> list[Action]:
        del state
        selected: dict[str, Action] = {}
        targetless: list[Action] = []
        for action in actions:
            if action.target is None:
                targetless.append(action)
                continue
            current = selected.get(action.target)
            if current is None or (action.priority, action.id) > (
                current.priority,
                current.id,
            ):
                selected[action.target] = action
        return [*targetless, *selected.values()]


def validate_actions(
    actions: Iterable[Action], validators, state: ContinuumState
) -> tuple[list[Action], list[ValidationResult]]:
    accepted: list[Action] = []
    results: list[ValidationResult] = []
    for action in actions:
        reason = None
        ok = True
        for validator in validators:
            ok, reason = validator.validate(action, state)
            if not ok:
                break
        results.append(ValidationResult(action, ok, reason))
        if ok:
            accepted.append(action)
    return accepted, results
