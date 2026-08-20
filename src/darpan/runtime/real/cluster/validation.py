"""Non-destructive physical-cluster validation for paper and deployment preflight."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from darpan._version import __version__
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.runtime.real.agent import AGENT_PROTOCOL_VERSION

from .inventory import ClusterInventory, ClusterNode


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    ok: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NodeValidation:
    node_id: str
    reachable: bool
    checks: tuple[ValidationCheck, ...]


@dataclass(frozen=True, slots=True)
class LinkValidation:
    link_id: str
    source: str
    target: str
    checks: tuple[ValidationCheck, ...]


@dataclass(frozen=True, slots=True)
class ClusterValidationReport:
    ready: bool
    nodes: tuple[NodeValidation, ...]
    links: tuple[LinkValidation, ...]
    errors: tuple[str, ...]
    environment_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _timed(name: str, operation) -> ValidationCheck:
    started = perf_counter()
    try:
        details = await operation()
    except Exception as exc:  # physical boundary: preserve all failures in report
        return ValidationCheck(
            name=name,
            ok=False,
            duration_s=max(0.0, perf_counter() - started),
            error=f"{type(exc).__name__}: {exc}",
        )
    return ValidationCheck(
        name=name,
        ok=True,
        duration_s=max(0.0, perf_counter() - started),
        details=dict(details or {}),
    )


async def _validate_node(
    inventory: ClusterInventory,
    node: ClusterNode,
    *,
    system_node: NodeSpec | None = None,
    max_clock_offset_s: float = 1.0,
) -> NodeValidation:
    client = inventory.client(node)

    async def ping() -> dict[str, Any]:
        result = await client.ping()
        reported = str(result.get("node_id", ""))
        if reported != node.id:
            raise RuntimeError(
                f"inventory node id {node.id!r} does not match agent id {reported!r}"
            )
        agent_version = str(result.get("darpan_version", ""))
        if agent_version != __version__:
            raise RuntimeError(
                f"agent Darpan version {agent_version!r} does not match controller "
                f"version {__version__!r}"
            )
        protocol = int(result.get("agent_protocol_version", -1))
        if protocol != AGENT_PROTOCOL_VERSION:
            raise RuntimeError(
                f"agent protocol {protocol} is incompatible with controller "
                f"protocol {AGENT_PROTOCOL_VERSION}"
            )
        environment = result.get("environment")
        if not isinstance(environment, dict):
            raise RuntimeError("agent ping is missing environment snapshot")
        required_environment = (
            "system",
            "release",
            "machine",
            "python_version",
            "cpu_capacity",
            "cgroup_version",
        )
        missing_environment = tuple(
            key for key in required_environment if environment.get(key) is None
        )
        if missing_environment:
            raise RuntimeError(
                "agent environment snapshot missing fields: "
                + ", ".join(missing_environment)
            )
        features = dict(result.get("features", {}))
        expected = {
            "network_probe": node.network_probe,
            "artifact_forward": node.artifact_forward,
            "tls": node.tls_ca is not None,
            "physical_control": node.physical_control,
        }
        missing = tuple(
            name for name, enabled in expected.items() if enabled and not features.get(name)
        )
        if missing:
            raise RuntimeError(
                "inventory requires Agent features that are not enabled: "
                + ", ".join(missing)
            )
        return result

    async def clock_sync() -> dict[str, Any]:
        started_wall = time()
        result = await client.ping()
        finished_wall = time()
        remote_wall = result.get("wall_time_s")
        if remote_wall is None:
            raise RuntimeError("agent ping is missing wall_time_s")
        rtt_s = max(0.0, finished_wall - started_wall)
        midpoint = (started_wall + finished_wall) / 2.0
        offset_s = float(remote_wall) - midpoint
        uncertainty_s = rtt_s / 2.0
        return {
            "offset_s": offset_s,
            "absolute_offset_s": abs(offset_s),
            "rtt_s": rtt_s,
            "uncertainty_s": uncertainty_s,
            "method": "controller-midpoint",
        }

    async def physical_control() -> dict[str, Any]:
        if not node.physical_control:
            return {"required": False}
        result = await client.control_inspect()
        if not bool(result.get("enabled", False)):
            raise RuntimeError("inventory requires physical control but Agent disabled it")
        observed_interfaces = set(str(item) for item in result.get("interfaces", ()))
        missing_interfaces = sorted(set(node.control_interfaces) - observed_interfaces)
        if missing_interfaces:
            raise RuntimeError(
                "Agent physical control is missing inventory interfaces: "
                + ", ".join(missing_interfaces)
            )
        readiness = result.get("readiness")
        if isinstance(readiness, dict) and readiness:
            interface_status = readiness.get("interfaces", {})
            unavailable_interfaces = sorted(
                interface
                for interface in node.control_interfaces
                if not bool(interface_status.get(interface, False))
            )
            if unavailable_interfaces:
                raise RuntimeError(
                    "Agent physical-control interfaces do not exist: "
                    + ", ".join(unavailable_interfaces)
                )
        if node.control_cpu_max is not None:
            if not bool(result.get("cpu_capacity")):
                raise RuntimeError(
                    "inventory configures control_cpu_max but Agent lacks CPU control"
                )
            if isinstance(readiness, dict) and readiness and not bool(
                readiness.get("cpu_capacity_ready", False)
            ):
                raise RuntimeError(
                    "configured control_cpu_max is missing or not writable on Agent"
                )
        return result

    async def telemetry() -> dict[str, Any]:
        result = await client.telemetry()
        reported = str(result.get("node_id", ""))
        if reported != node.id:
            raise RuntimeError(
                f"telemetry node id {reported!r} does not match inventory {node.id!r}"
            )
        required = ("cpu_capacity", "workspace_free_bytes")
        missing = tuple(key for key in required if result.get(key) is None)
        if missing:
            raise RuntimeError("telemetry missing fields: " + ", ".join(missing))
        return result

    ping_check = await _timed("ping", ping)
    clock_check = await _timed("clock_sync", clock_sync)
    if clock_check.ok:
        absolute = float(clock_check.details.get("absolute_offset_s", float("inf")))
        uncertainty = float(clock_check.details.get("uncertainty_s", 0.0))
        if max(0.0, absolute - uncertainty) > max_clock_offset_s:
            clock_check = replace(
                clock_check,
                ok=False,
                error=(
                    f"clock offset {absolute:.6f}s exceeds limit "
                    f"{max_clock_offset_s:.6f}s after RTT uncertainty"
                ),
            )
    telemetry_check = await _timed("telemetry", telemetry)
    control_check = await _timed("physical_control", physical_control)
    checks: list[ValidationCheck] = [
        ping_check,
        clock_check,
        telemetry_check,
        control_check,
    ]

    declared_cpu = None
    if system_node is not None:
        declared = next(
            (resource for resource in system_node.resources if resource.name == "cpu"),
            None,
        )
        if declared is not None:
            declared_cpu = float(declared.capacity)
    if declared_cpu is not None and telemetry_check.ok:
        observed_cpu = float(telemetry_check.details["cpu_capacity"])
        started = perf_counter()
        effective_cpu = min(declared_cpu, observed_cpu)
        if declared_cpu > observed_cpu + 1e-12:
            checks.append(
                ValidationCheck(
                    name="declared_cpu_capacity",
                    ok=False,
                    duration_s=max(0.0, perf_counter() - started),
                    details={
                        "declared": declared_cpu,
                        "observed": observed_cpu,
                        "effective": effective_cpu,
                    },
                    error=(
                        f"declared CPU capacity {declared_cpu} exceeds observed "
                        f"physical capacity {observed_cpu}"
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="declared_cpu_capacity",
                    ok=True,
                    duration_s=max(0.0, perf_counter() - started),
                    details={
                        "declared": declared_cpu,
                        "observed": observed_cpu,
                        "effective": effective_cpu,
                    },
                )
            )
    return NodeValidation(
        node_id=node.id,
        reachable=ping_check.ok,
        checks=tuple(checks),
    )


def _resolve_links(system: SystemSpec, inventory: ClusterInventory):
    nodes = {node.id for node in inventory.nodes}
    return tuple(
        link for link in system.links if link.source in nodes and link.target in nodes
    )


async def _validate_probe(
    inventory: ClusterInventory,
    source: ClusterNode,
    target: ClusterNode,
    *,
    payload_bytes: int,
    timeout_s: float,
) -> dict[str, Any]:
    if not source.network_probe:
        raise RuntimeError(
            f"source node {source.id!r} has network_probe disabled in inventory"
        )
    result = await inventory.client(source).probe_agent(
        target.host,
        target.port,
        token=target.token(),
        ca_pem=inventory.tls_ca_pem(target),
        server_hostname=target.tls_server_name or target.host,
        samples=2,
        payload_bytes=payload_bytes,
        timeout=timeout_s,
    )
    if float(result.get("rtt_ms", 0.0)) <= 0:
        raise RuntimeError("agent-to-agent probe returned non-positive RTT")
    return result


async def _validate_direct_artifact(
    inventory: ClusterInventory,
    source: ClusterNode,
    target: ClusterNode,
    *,
    payload_bytes: int,
) -> dict[str, Any]:
    if not source.artifact_forward:
        raise RuntimeError(
            f"source node {source.id!r} has artifact_forward disabled in inventory"
        )
    source_client = inventory.client(source)
    target_client = inventory.client(target)
    workspace = f"cluster-validate-{uuid4()}"
    source_path = "source.bin"
    target_path = "target.bin"
    payload = os.urandom(payload_bytes)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        await source_client.put_file(workspace, source_path, payload)
        ticket = await target_client.prepare_incoming_transfer(
            workspace,
            target_path,
            size_bytes=len(payload),
            expires_s=30.0,
        )
        endpoint = target_client.direct_transfer_endpoint()
        transferred, data_duration = await source_client.forward_artifact(
            workspace,
            source_path,
            target_host=str(endpoint["host"]),
            target_port=int(endpoint["port"]),
            target_workspace=workspace,
            target_path=target_path,
            ticket=ticket,
            target_ca_pem=endpoint.get("ca_pem"),
            target_server_hostname=endpoint.get("server_hostname"),
        )
        observed = await target_client.get_file(workspace, target_path)
        observed_digest = hashlib.sha256(observed).hexdigest()
        if observed_digest != digest:
            raise OSError("direct artifact validation checksum mismatch")
        return {
            "size_bytes": transferred,
            "sha256": digest,
            "data_duration_s": data_duration,
        }
    finally:
        await asyncio.gather(
            source_client.delete_file(workspace, source_path),
            target_client.delete_file(workspace, target_path),
            return_exceptions=True,
        )


async def validate_cluster(
    inventory: ClusterInventory,
    *,
    system: SystemSpec | None = None,
    exercise_data_plane: bool = False,
    payload_bytes: int = 4096,
    timeout_s: float = 5.0,
    max_clock_offset_s: float = 1.0,
) -> ClusterValidationReport:
    """Validate physical agents and, optionally, declared physical link data paths.

    Node checks are always non-destructive. Link checks only run for links whose
    endpoints occur in both the system and inventory. The direct-artifact check
    uses an isolated temporary workspace and deletes its validation files.
    """

    if payload_bytes < 0 or payload_bytes > 4 * 1024 * 1024:
        raise ValueError("payload_bytes must be in [0, 4194304]")
    if timeout_s <= 0 or timeout_s > 30:
        raise ValueError("timeout_s must be in (0, 30]")
    if max_clock_offset_s < 0:
        raise ValueError("max_clock_offset_s cannot be negative")
    if not inventory.nodes:
        raise ValueError("cluster inventory cannot be empty")

    system_node_map = (
        {node.id: node for node in system.nodes}
        if system is not None
        else {}
    )
    node_reports = tuple(
        await asyncio.gather(
            *(
                _validate_node(
                    inventory,
                    node,
                    system_node=system_node_map.get(node.id),
                    max_clock_offset_s=max_clock_offset_s,
                )
                for node in inventory.nodes
            )
        )
    )
    link_reports: list[LinkValidation] = []
    errors: list[str] = []

    node_map = {node.id: node for node in inventory.nodes}
    for report in node_reports:
        for check in report.checks:
            if not check.ok:
                errors.append(f"node {report.node_id} {check.name}: {check.error}")

    if system is not None:
        system_nodes = {node.id for node in system.nodes}
        missing_inventory = sorted(system_nodes - set(node_map))
        for node_id in missing_inventory:
            errors.append(f"system node {node_id!r} is missing from cluster inventory")

        for link in _resolve_links(system, inventory):
            source = node_map[link.source]
            target = node_map[link.target]
            checks: list[ValidationCheck] = []
            if exercise_data_plane:
                checks.append(
                    await _timed(
                        "network_probe",
                        lambda source=source, target=target: _validate_probe(
                            inventory,
                            source,
                            target,
                            payload_bytes=payload_bytes,
                            timeout_s=timeout_s,
                        ),
                    )
                )
                checks.append(
                    await _timed(
                        "direct_artifact",
                        lambda source=source, target=target: _validate_direct_artifact(
                            inventory,
                            source,
                            target,
                            payload_bytes=payload_bytes,
                        ),
                    )
                )
            link_report = LinkValidation(
                link_id=link.id,
                source=link.source,
                target=link.target,
                checks=tuple(checks),
            )
            link_reports.append(link_report)
            for check in checks:
                if not check.ok:
                    errors.append(f"link {link.id} {check.name}: {check.error}")

    environments = {}
    for report in node_reports:
        ping_check = next(
            (check for check in report.checks if check.name == "ping" and check.ok),
            None,
        )
        if ping_check is not None and isinstance(
            ping_check.details.get("environment"), dict
        ):
            environments[report.node_id] = ping_check.details["environment"]
    environment_fingerprint = None
    if environments:
        rendered = json.dumps(
            environments,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        environment_fingerprint = hashlib.sha256(rendered).hexdigest()

    return ClusterValidationReport(
        ready=not errors,
        nodes=node_reports,
        links=tuple(link_reports),
        errors=tuple(errors),
        environment_fingerprint=environment_fingerprint,
    )
