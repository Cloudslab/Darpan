"""Agent-driven discovery for generating physical inventory/system candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .inventory import ClusterInventory


@dataclass(frozen=True, slots=True)
class DiscoveredNode:
    id: str
    host: str
    port: int
    ping: dict[str, Any]
    telemetry: dict[str, Any]
    physical_control: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ClusterDiscovery:
    nodes: tuple[DiscoveredNode, ...]
    environment_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def discover_cluster(inventory: ClusterInventory) -> ClusterDiscovery:
    async def inspect(node):
        client = inventory.client(node)
        ping, telemetry = await asyncio.gather(client.ping(), client.telemetry())
        control = None
        if node.physical_control:
            control = await client.control_inspect()
        return DiscoveredNode(
            id=node.id,
            host=node.host,
            port=node.port,
            ping=dict(ping),
            telemetry=dict(telemetry),
            physical_control=None if control is None else dict(control),
        )

    nodes = tuple(await asyncio.gather(*(inspect(node) for node in inventory.nodes)))
    environments = {
        node.id: node.ping.get("environment", {})
        for node in sorted(nodes, key=lambda item: item.id)
    }
    encoded = json.dumps(environments, sort_keys=True, separators=(",", ":")).encode()
    return ClusterDiscovery(nodes, hashlib.sha256(encoded).hexdigest())


def discovered_inventory_dict(
    inventory: ClusterInventory,
    discovery: ClusterDiscovery,
) -> dict[str, Any]:
    discovered = {node.id: node for node in discovery.nodes}
    output = []
    for node in inventory.nodes:
        environment = dict(discovered[node.id].ping.get("environment", {}))
        labels = dict(node.labels)
        if environment.get("machine"):
            labels.setdefault("architecture", str(environment["machine"]))
        if environment.get("hostname"):
            labels.setdefault("hostname", str(environment["hostname"]))
        item: dict[str, Any] = {
            "id": node.id,
            "host": node.host,
            "port": node.port,
        }
        if node.ssh_user is not None:
            item["ssh_user"] = node.ssh_user
        if node.ssh_port != 22:
            item["ssh_port"] = node.ssh_port
        if node.ssh_identity_file is not None:
            item["ssh_identity_file"] = node.ssh_identity_file
        if node.token_env is not None:
            item["token_env"] = node.token_env
        if node.tls_ca is not None:
            item["tls_ca"] = node.tls_ca
        if node.tls_server_name is not None:
            item["tls_server_name"] = node.tls_server_name
        if node.tls_cert is not None:
            item["tls_cert"] = node.tls_cert
            item["tls_key"] = node.tls_key
        if node.network_probe:
            item["network_probe"] = True
        if node.artifact_forward:
            item["artifact_forward"] = True
        if node.physical_control:
            item["physical_control"] = True
            item["control_interfaces"] = list(node.control_interfaces)
            if node.control_cpu_max is not None:
                item["control_cpu_max"] = node.control_cpu_max
            if node.allow_replace_existing_qdisc:
                item["allow_replace_existing_qdisc"] = True
        if labels:
            item["labels"] = labels
        output.append(item)
    return {"nodes": output}


def discovered_system_dict(
    inventory: ClusterInventory,
    discovery: ClusterDiscovery,
) -> dict[str, Any]:
    by_id = {node.id: node for node in discovery.nodes}
    nodes = []
    for inventory_node in inventory.nodes:
        observed = by_id[inventory_node.id]
        environment = dict(observed.ping.get("environment", {}))
        telemetry = observed.telemetry
        cpu = float(telemetry.get("cpu_capacity", environment.get("cpu_capacity", 1.0)))
        memory_bytes = environment.get("memory_capacity_bytes")
        resources: dict[str, Any] = {"cpu": cpu}
        if memory_bytes is not None:
            resources["memory"] = round(int(memory_bytes) / (1024**3), 6)
        labels = dict(inventory_node.labels)
        labels.update(
            {
                "architecture": str(environment.get("machine", "unknown")),
                "hostname": str(environment.get("hostname", inventory_node.host)),
            }
        )
        nodes.append(
            {
                "id": inventory_node.id,
                "tier": labels.get("tier", "unknown"),
                "resources": resources,
                "labels": labels,
            }
        )
    return {
        "name": "discovered-physical-continuum",
        "nodes": nodes,
        "links": [],
    }
