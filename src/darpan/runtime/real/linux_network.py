"""Concrete Linux host-route implementation of NetworkControlDriver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from darpan.core.state import ContinuumState, FlowRouteBinding


@dataclass(slots=True)
class AgentLinuxNetworkControlDriver:
    """Bind Darpan flows using explicit Linux host routes on source Agents.

    This is intentionally a host-route driver, not an SDN fabric. A flow route
    is accepted only when the topology supplies enough physical addressing
    metadata. The OS route may affect all traffic to the destination host; that
    limitation is exposed rather than hidden as per-flow enforcement.
    """

    clients: Mapping[str, Any]
    bindings: dict[tuple[str, str, str], dict[str, str]] = field(default_factory=dict)

    @staticmethod
    def _single_running_node(
        state: ContinuumState,
        application_instance_id: str,
        component_id: str,
    ) -> str:
        members = [
            item
            for item in state.components.values()
            if item.application_instance_id == application_instance_id
            and item.component_id == component_id
            and item.status == "running"
        ]
        if len(members) != 1 or members[0].node_id is None:
            raise RuntimeError(
                "Linux host-route control requires exactly one running replica per endpoint"
            )
        return str(members[0].node_id)

    @staticmethod
    def _link_for_hop(state: ContinuumState, source: str, target: str):
        candidates = []
        for link in state.links.values():
            spec = link.spec
            if spec.source == source and spec.target == target:
                candidates.append(link)
            elif spec.bidirectional and spec.source == target and spec.target == source:
                candidates.append(link)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Linux route requires exactly one physical link for hop {source}->{target}"
            )
        return candidates[0]

    async def bind_route(
        self,
        binding: FlowRouteBinding,
        state: ContinuumState,
    ) -> None:
        source_node = self._single_running_node(
            state,
            binding.application_instance_id,
            binding.source_component_id,
        )
        target_node = self._single_running_node(
            state,
            binding.application_instance_id,
            binding.target_component_id,
        )
        if binding.path[0] != source_node or binding.path[-1] != target_node:
            raise RuntimeError("route path endpoints do not match running component nodes")
        if len(binding.path) < 2:
            raise RuntimeError("Linux route requires at least two path nodes")
        destination = str(state.nodes[target_node].labels.get("route_ipv4", "")).strip()
        if not destination:
            raise RuntimeError(f"target node {target_node!r} requires label route_ipv4")
        destination = destination if "/" in destination else f"{destination}/32"
        next_node = binding.path[1]
        hop = self._link_for_hop(state, source_node, next_node)
        labels = dict(hop.spec.labels)
        if hop.spec.source == source_node:
            interface = str(labels.get("physical_source_interface", "")).strip()
            via = str(labels.get("physical_target_ipv4", "")).strip()
        else:
            interface = str(labels.get("physical_target_interface", "")).strip()
            via = str(labels.get("physical_source_ipv4", "")).strip()
        if not interface or not via:
            raise RuntimeError(
                f"link {hop.spec.id!r} requires physical interface and peer IPv4 labels"
            )
        try:
            client = self.clients[source_node]
        except KeyError as exc:
            raise RuntimeError(f"no Linux route-control Agent for {source_node!r}") from exc
        await client.control_route_bind(
            destination=destination,
            via=via,
            interface=interface,
        )
        self.bindings[
            (
                binding.application_instance_id,
                binding.source_component_id,
                binding.target_component_id,
            )
        ] = {
            "node_id": source_node,
            "destination": destination,
            "via": via,
            "interface": interface,
        }

    async def clear_route(
        self,
        application_instance_id: str,
        source_component_id: str,
        target_component_id: str,
        state: ContinuumState,
    ) -> None:
        key = (application_instance_id, source_component_id, target_component_id)
        previous = self.bindings.pop(key, None)
        if previous is None:
            return
        client = self.clients[previous["node_id"]]
        await client.control_route_clear(destination=previous["destination"])
