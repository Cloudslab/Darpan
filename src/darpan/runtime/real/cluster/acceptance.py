"""First-run acceptance suite for a deployed physical Darpan cluster."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from darpan._version import __version__
from darpan.core.action import Action
from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest
from darpan.core.serialization import to_primitive
from darpan.core.topology import SystemSpec

from .exercise import ClusterRuntimeExerciseReport, exercise_cluster_runtime
from .inventory import ClusterInventory
from .session import session_from_inventory
from .validation import ClusterValidationReport, validate_cluster

DEPLOYMENT_RECEIPT_SCHEMA = "darpan.deployment-receipt/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentReceipt:
    schema: str
    darpan_version: str
    ready_for_study: bool
    cluster_sha256: str
    system_sha256: str
    environment_fingerprint: str | None
    source_node_id: str
    target_node_id: str
    accepted_capabilities: tuple[str, ...]
    deployment_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_deployment_receipt(
    report: ClusterAcceptanceReport,
    *,
    cluster: str | Path,
    system: str | Path,
    additional_capabilities: tuple[str, ...] = (),
) -> DeploymentReceipt:
    """Build a stable receipt tying active acceptance to frozen inputs/environment."""

    cluster_path = Path(cluster).expanduser().resolve()
    system_path = Path(system).expanduser().resolve()
    capabilities = [
        "cluster_preflight",
        "data_plane",
        "lifecycle_control",
        "scale",
        *additional_capabilities,
    ]
    for step in report.steps:
        if step.ok and step.name == "workload_control_restore":
            capabilities.append("physical_control")
        if step.ok and step.name == "link_control_restore":
            capabilities.append("link_control")
    basis = {
        "schema": DEPLOYMENT_RECEIPT_SCHEMA,
        "darpan_version": __version__,
        "ready_for_study": report.ready_for_study,
        "cluster_sha256": _file_sha256(cluster_path),
        "system_sha256": _file_sha256(system_path),
        "environment_fingerprint": report.preflight.environment_fingerprint,
        "source_node_id": report.source_node_id,
        "target_node_id": report.target_node_id,
        "accepted_capabilities": sorted(set(capabilities)),
    }
    return DeploymentReceipt(
        schema=DEPLOYMENT_RECEIPT_SCHEMA,
        darpan_version=__version__,
        ready_for_study=report.ready_for_study,
        cluster_sha256=str(basis["cluster_sha256"]),
        system_sha256=str(basis["system_sha256"]),
        environment_fingerprint=report.preflight.environment_fingerprint,
        source_node_id=report.source_node_id,
        target_node_id=report.target_node_id,
        accepted_capabilities=tuple(basis["accepted_capabilities"]),
        deployment_fingerprint=_canonical_hash(basis),
    )


def verify_deployment_receipt_payload(payload: dict[str, Any]) -> DeploymentReceipt:
    """Validate a deployment receipt payload and return its typed representation."""

    if payload.get("schema") != DEPLOYMENT_RECEIPT_SCHEMA:
        raise ValueError("unsupported deployment receipt schema")
    required = (
        "darpan_version",
        "ready_for_study",
        "cluster_sha256",
        "system_sha256",
        "source_node_id",
        "target_node_id",
        "accepted_capabilities",
        "deployment_fingerprint",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("deployment receipt is missing: " + ", ".join(missing))
    capabilities = tuple(sorted(str(item) for item in payload["accepted_capabilities"]))
    basis = {
        "schema": DEPLOYMENT_RECEIPT_SCHEMA,
        "darpan_version": str(payload["darpan_version"]),
        "ready_for_study": bool(payload["ready_for_study"]),
        "cluster_sha256": str(payload["cluster_sha256"]),
        "system_sha256": str(payload["system_sha256"]),
        "environment_fingerprint": payload.get("environment_fingerprint"),
        "source_node_id": str(payload["source_node_id"]),
        "target_node_id": str(payload["target_node_id"]),
        "accepted_capabilities": list(capabilities),
    }
    observed = str(payload["deployment_fingerprint"])
    expected = _canonical_hash(basis)
    if observed != expected:
        raise RuntimeError("deployment receipt fingerprint is invalid")
    return DeploymentReceipt(
        schema=DEPLOYMENT_RECEIPT_SCHEMA,
        darpan_version=str(payload["darpan_version"]),
        ready_for_study=bool(payload["ready_for_study"]),
        cluster_sha256=str(payload["cluster_sha256"]),
        system_sha256=str(payload["system_sha256"]),
        environment_fingerprint=(
            None
            if payload.get("environment_fingerprint") is None
            else str(payload["environment_fingerprint"])
        ),
        source_node_id=str(payload["source_node_id"]),
        target_node_id=str(payload["target_node_id"]),
        accepted_capabilities=capabilities,
        deployment_fingerprint=observed,
    )


@dataclass(frozen=True, slots=True)
class AcceptanceStep:
    name: str
    ok: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClusterAcceptanceReport:
    ready_for_study: bool
    source_node_id: str
    target_node_id: str
    preflight: ClusterValidationReport
    lifecycle: ClusterRuntimeExerciseReport | None
    steps: tuple[AcceptanceStep, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _step(name: str, operation) -> AcceptanceStep:
    started = perf_counter()
    try:
        details = await operation()
    except Exception as exc:
        return AcceptanceStep(
            name=name,
            ok=False,
            duration_s=max(0.0, perf_counter() - started),
            error=f"{type(exc).__name__}: {exc}",
        )
    return AcceptanceStep(
        name=name,
        ok=True,
        duration_s=max(0.0, perf_counter() - started),
        details=dict(details or {}),
    )


async def _wait_active_executions(client, expected: int, *, timeout_s: float) -> int:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    observed = -1
    while True:
        observed = int((await client.telemetry())["active_executions"])
        if observed == expected:
            return observed
        if loop.time() >= deadline:
            raise RuntimeError(
                "Agent active execution count did not converge: "
                f"expected {expected}, observed {observed}"
            )
        await asyncio.sleep(0.02)


async def _scale_drill(
    inventory: ClusterInventory,
    system: SystemSpec,
    *,
    source_node_id: str,
    target_node_id: str,
    command: tuple[str, ...],
    cpu_request: float | None,
    timeout_s: float,
) -> dict[str, Any]:
    resources = (
        ()
        if cpu_request is None
        else (ResourceRequest("cpu", float(cpu_request)),)
    )
    app = ApplicationSpec(
        "darpan-cluster-scale-acceptance",
        components=(
            ComponentSpec(
                "service",
                kind="service",
                command=command,
                resources=resources,
                work_units=0,
            ),
        ),
    )
    instance = f"cluster-scale-{uuid4().hex[:10]}"
    primary = f"{instance}:service"
    replica = f"{primary}#replica-1"
    source_node = next(node for node in inventory.nodes if node.id == source_node_id)
    source_client = inventory.client(source_node)
    target_node = next(node for node in inventory.nodes if node.id == target_node_id)
    target_client = inventory.client(target_node)
    session = session_from_inventory(
        inventory,
        system=system,
        monitor=False,
        link_probes=False,
    )
    try:
        await session.start()
        await session.register_system(system)
        await session.submit_application(app, instance_id=instance)
        assert await session.apply(Action.place(primary, source_node_id, source="cluster.accept"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_STARTED
            and event.subject == primary,
            timeout=timeout_s,
        )
        assert await session.apply(Action.scale(primary, 2, source="cluster.accept"))
        await session.wait_for(
            lambda event, state: (
                replica in state.components and state.components[replica].status == "ready"
            ),
            timeout=timeout_s,
        )
        assert await session.apply(Action.place(replica, target_node_id, source="cluster.accept"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_SCALED
            and event.subject == primary
            and event.payload.get("desired_replicas") == 2,
            timeout=timeout_s,
        )
        source_active = int((await source_client.telemetry())["active_executions"])
        target_active = int((await target_client.telemetry())["active_executions"])
        if source_active != 1 or target_active != 1:
            raise RuntimeError(
                "scale-out did not produce one physical execution on each acceptance node"
            )
        assert await session.apply(Action.scale(primary, 1, source="cluster.accept"))
        await session.wait_for(
            lambda event, state: (
                replica in state.components
                and state.components[replica].status == "completed"
            ),
            timeout=timeout_s,
        )
        target_after = await _wait_active_executions(
            target_client,
            0,
            timeout_s=timeout_s,
        )
        assert await session.apply(Action.stop(primary, source="cluster.accept"))
        completion = await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED
            and event.payload.get("instance_id") == instance,
            timeout=timeout_s,
        )
        if not bool(completion.event.payload.get("success", False)):
            raise RuntimeError("scale acceptance application completed unsuccessfully")
        return {
            "desired_replicas": session.state.desired_replicas(primary),
            "source_active_after_scale_out": source_active,
            "target_active_after_scale_out": target_active,
            "target_active_after_scale_in": target_after,
            "application_success": True,
        }
    finally:
        await session.close()


async def _workload_control_drill(
    inventory: ClusterInventory,
    *,
    node_id: str,
    python_command: str,
) -> dict[str, Any]:
    node = next(node for node in inventory.nodes if node.id == node_id)
    if not node.physical_control:
        raise RuntimeError(f"node {node_id!r} has physical_control disabled")
    client = inventory.client(node)
    await client.control_workload(enabled=False, lease_s=30.0)
    blocked = False
    try:
        try:
            await client.start_execution(
                to_primitive(
                    ComponentSpec(
                        "acceptance-probe",
                        command=(python_command, "-c", "print('blocked-probe')"),
                    )
                )
            )
        except RuntimeError:
            blocked = True
        if not blocked:
            raise RuntimeError("disabled workload plane still accepted a new execution")
    finally:
        restored = await client.control_restore()
    execution_id = await client.start_execution(
        to_primitive(
            ComponentSpec(
                "acceptance-probe",
                command=(python_command, "-c", "print('darpan-acceptance-ok')"),
            )
        )
    )
    result = await client.wait_execution(execution_id)
    execution = dict(result.get("result") or {})
    if int(execution.get("return_code", -1)) != 0:
        raise RuntimeError("restored workload plane could not execute a probe")
    return {
        "blocked_while_disabled": blocked,
        "restore_ok": bool(restored.get("ok", False)),
        "post_restore_return_code": int(execution["return_code"]),
    }


async def _link_control_drill(
    inventory: ClusterInventory,
    system: SystemSpec,
) -> dict[str, Any]:
    link = next(
        (
            item
            for item in system.links
            if item.labels.get("physical_control_scope") == "interface"
            and item.labels.get("physical_source_interface")
        ),
        None,
    )
    if link is None:
        raise RuntimeError("system has no interface-scoped link eligible for acceptance control")
    node = next(node for node in inventory.nodes if node.id == link.source)
    if not node.physical_control:
        raise RuntimeError(f"link source {link.source!r} has physical_control disabled")
    interface = str(link.labels["physical_source_interface"])
    client = inventory.client(node)
    try:
        applied = await client.control_netem(
            interface=interface,
            latency_ms=max(0.1, float(link.latency_ms) + 0.5),
            lease_s=30.0,
        )
    finally:
        restored = await client.control_restore()
    if not bool(restored.get("ok", False)):
        raise RuntimeError("link-control acceptance restoration was not verified")
    return {
        "link_id": link.id,
        "node_id": node.id,
        "interface": interface,
        "applied": applied,
        "restored": restored,
    }


async def accept_cluster(
    inventory: ClusterInventory,
    system: SystemSpec,
    *,
    source_node_id: str,
    target_node_id: str,
    command: tuple[str, ...],
    python_command: str = "python3",
    cpu_request: float | None = None,
    timeout_s: float = 10.0,
    exercise_physical_control: bool = False,
    exercise_link_control: bool = False,
) -> ClusterAcceptanceReport:
    """Run a short active acceptance suite before a long Physical Study."""

    preflight = await validate_cluster(
        inventory,
        system=system,
        exercise_data_plane=True,
        timeout_s=min(timeout_s, 30.0),
    )
    if not preflight.ready:
        return ClusterAcceptanceReport(
            ready_for_study=False,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            preflight=preflight,
            lifecycle=None,
            steps=(),
            errors=tuple(preflight.errors),
        )
    lifecycle = await exercise_cluster_runtime(
        inventory,
        system,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        command=command,
        cpu_request=cpu_request,
        timeout_s=timeout_s,
    )
    steps: list[AcceptanceStep] = []
    errors: list[str] = list(lifecycle.errors)
    scale = await _step(
        "scale_out_in",
        lambda: _scale_drill(
            inventory,
            system,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            command=command,
            cpu_request=cpu_request,
            timeout_s=timeout_s,
        ),
    )
    steps.append(scale)
    if not scale.ok:
        errors.append(scale.error or "scale acceptance failed")
    if exercise_physical_control:
        control = await _step(
            "workload_control_restore",
            lambda: _workload_control_drill(
                inventory,
                node_id=source_node_id,
                python_command=python_command,
            ),
        )
        steps.append(control)
        if not control.ok:
            errors.append(control.error or "physical-control acceptance failed")
    if exercise_link_control:
        link = await _step(
            "link_control_restore",
            lambda: _link_control_drill(inventory, system),
        )
        steps.append(link)
        if not link.ok:
            errors.append(link.error or "link-control acceptance failed")
    return ClusterAcceptanceReport(
        ready_for_study=(lifecycle.ready and not errors and all(step.ok for step in steps)),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        preflight=preflight,
        lifecycle=lifecycle,
        steps=tuple(steps),
        errors=tuple(errors),
    )
