"""Active physical-runtime drill for deployment and paper preflight."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

from darpan.core.action import Action
from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest
from darpan.core.topology import SystemSpec

from .inventory import ClusterInventory
from .session import session_from_inventory
from .validation import ClusterValidationReport, validate_cluster


@dataclass(frozen=True, slots=True)
class RuntimeExerciseStep:
    name: str
    ok: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClusterRuntimeExerciseReport:
    ready: bool
    source_node_id: str
    target_node_id: str
    preflight: ClusterValidationReport
    steps: tuple[RuntimeExerciseStep, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _wait_active_executions(client, expected: int, *, timeout_s: float) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, Any] | None = None
    while True:
        last = await client.telemetry()
        active = int(last.get("active_executions", -1))
        if active == expected:
            return {"active_executions": active}
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"agent active_executions did not become {expected}; last={active}"
            )
        await asyncio.sleep(0.02)


async def _step(name: str, operation) -> RuntimeExerciseStep:
    started = perf_counter()
    try:
        details = await operation()
    except Exception as exc:
        return RuntimeExerciseStep(
            name=name,
            ok=False,
            duration_s=max(0.0, perf_counter() - started),
            error=f"{type(exc).__name__}: {exc}",
        )
    return RuntimeExerciseStep(
        name=name,
        ok=True,
        duration_s=max(0.0, perf_counter() - started),
        details=dict(details or {}),
    )


async def exercise_cluster_runtime(
    inventory: ClusterInventory,
    system: SystemSpec,
    *,
    source_node_id: str,
    target_node_id: str,
    command: tuple[str, ...],
    cpu_request: float | None = None,
    timeout_s: float = 10.0,
) -> ClusterRuntimeExerciseReport:
    """Place, restart, migrate, and stop one ephemeral service on real Agents.

    The drill is intentionally active: unlike ``validate_cluster``, it starts a
    real remote process, restarts it in place, migrates the same canonical
    component instance to the target Agent, verifies the physical handoffs,
    then stops it cleanly.
    """

    if not command:
        raise ValueError("cluster runtime exercise command cannot be empty")
    if timeout_s <= 0 or timeout_s > 120:
        raise ValueError("timeout_s must be in (0, 120]")
    if source_node_id == target_node_id:
        raise ValueError("source and target nodes must be different")
    if cpu_request is not None and cpu_request <= 0:
        raise ValueError("cpu_request must be positive when provided")

    inventory_nodes = {node.id: node for node in inventory.nodes}
    system_nodes = {node.id for node in system.nodes}
    for node_id in (source_node_id, target_node_id):
        if node_id not in inventory_nodes:
            raise ValueError(f"cluster inventory does not contain node {node_id!r}")
        if node_id not in system_nodes:
            raise ValueError(f"system does not contain node {node_id!r}")

    preflight = await validate_cluster(inventory, system=system)
    if not preflight.ready:
        errors = tuple(preflight.errors)
        return ClusterRuntimeExerciseReport(
            ready=False,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            preflight=preflight,
            steps=(),
            errors=errors,
        )

    source_client = inventory.client(inventory_nodes[source_node_id])
    target_client = inventory.client(inventory_nodes[target_node_id])
    resources = (
        ()
        if cpu_request is None
        else (ResourceRequest("cpu", float(cpu_request)),)
    )
    application = ApplicationSpec(
        "darpan-cluster-runtime-exercise",
        components=(
            ComponentSpec(
                "service",
                kind="service",
                command=tuple(command),
                resources=resources,
                work_units=0,
            ),
        ),
    )
    instance_id = f"cluster-exercise-{uuid4().hex[:10]}"
    component_id = f"{instance_id}:service"
    session = session_from_inventory(
        inventory, system=system, monitor=False, link_probes=False
    )
    steps: list[RuntimeExerciseStep] = []
    errors: list[str] = []

    try:
        await session.start()
        await session.register_system(system)
        await session.submit_application(application, instance_id=instance_id)

        async def place_source() -> dict[str, Any]:
            after = session.event_count
            if not await session.apply(Action.place(component_id, source_node_id)):
                raise RuntimeError("source placement action was rejected")
            await session.wait_for(
                lambda event, state: (
                    event.kind == EventKind.COMPONENT_STARTED
                    and event.subject == component_id
                    and event.payload.get("node_id") == source_node_id
                ),
                after=after,
                timeout=timeout_s,
            )
            active = await _wait_active_executions(
                source_client,
                1,
                timeout_s=timeout_s,
            )
            return {
                "component_status": session.state.components[component_id].status,
                "node_id": session.state.components[component_id].node_id,
                **active,
            }

        steps.append(await _step("place_source", place_source))
        if not steps[-1].ok:
            raise RuntimeError(steps[-1].error or "source placement failed")

        async def restart_source() -> dict[str, Any]:
            after = session.event_count
            action = Action.restart(
                component_id,
                source="cluster.exercise",
                metadata={"reason": "physical-runtime-drill"},
            )
            if not await session.apply(action):
                raise RuntimeError("restart action was rejected")
            started = await session.wait_for(
                lambda event, state: (
                    event.kind == EventKind.COMPONENT_STARTED
                    and event.subject == component_id
                    and event.payload.get("node_id") == source_node_id
                    and event.causation_id == action.id
                ),
                after=after,
                timeout=timeout_s,
            )
            restarting = next(
                event
                for event in session.events_since(after)
                if event.kind == EventKind.COMPONENT_RESTARTING
                and event.subject == component_id
            )
            active = await _wait_active_executions(
                source_client,
                1,
                timeout_s=timeout_s,
            )
            component = session.state.components[component_id]
            return {
                "component_status": component.status,
                "node_id": component.node_id,
                "attempt": component.attempt,
                "restart_mode": restarting.payload.get("mode"),
                "canonical_downtime_s": max(
                    0.0,
                    started.event.event_time - restarting.event_time,
                ),
                **active,
            }

        steps.append(await _step("restart_source", restart_source))
        if not steps[-1].ok:
            raise RuntimeError(steps[-1].error or "restart failed")

        async def migrate_target() -> dict[str, Any]:
            after = session.event_count
            action = Action.migrate(
                component_id,
                target_node_id,
                source="cluster.exercise",
                metadata={"reason": "physical-runtime-drill"},
            )
            if not await session.apply(action):
                raise RuntimeError("migration action was rejected")
            started = await session.wait_for(
                lambda event, state: (
                    event.kind == EventKind.COMPONENT_STARTED
                    and event.subject == component_id
                    and event.payload.get("node_id") == target_node_id
                ),
                after=after,
                timeout=timeout_s,
            )
            migrating = next(
                event
                for event in session.events_since(after)
                if event.kind == EventKind.COMPONENT_MIGRATING
                and event.subject == component_id
            )
            source_active, target_active = await asyncio.gather(
                _wait_active_executions(source_client, 0, timeout_s=timeout_s),
                _wait_active_executions(target_client, 1, timeout_s=timeout_s),
            )
            component = session.state.components[component_id]
            return {
                "component_status": component.status,
                "node_id": component.node_id,
                "attempt": component.attempt,
                "migration_mode": migrating.payload.get("mode"),
                "canonical_downtime_s": max(
                    0.0,
                    started.event.event_time - migrating.event_time,
                ),
                "source_active_executions": source_active["active_executions"],
                "target_active_executions": target_active["active_executions"],
            }

        steps.append(await _step("migrate_target", migrate_target))
        if not steps[-1].ok:
            raise RuntimeError(steps[-1].error or "migration failed")

        async def stop_target() -> dict[str, Any]:
            if not await session.apply(
                Action.stop(component_id, source="cluster.exercise")
            ):
                raise RuntimeError("stop action was rejected")
            completion = await session.wait_for(
                lambda event, state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance_id
                ),
                timeout=timeout_s,
            )
            active = await _wait_active_executions(
                target_client,
                0,
                timeout_s=timeout_s,
            )
            if not bool(completion.event.payload.get("success", False)):
                raise RuntimeError("exercise application completed unsuccessfully")
            return {
                "application_success": True,
                "component_status": session.state.components[component_id].status,
                **active,
            }

        steps.append(await _step("stop_target", stop_target))
        if not steps[-1].ok:
            raise RuntimeError(steps[-1].error or "stop failed")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            await session.close()
        except Exception as exc:
            errors.append(f"session close failed: {type(exc).__name__}: {exc}")

    return ClusterRuntimeExerciseReport(
        ready=not errors and all(step.ok for step in steps) and len(steps) == 4,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        preflight=preflight,
        steps=tuple(steps),
        errors=tuple(errors),
    )
