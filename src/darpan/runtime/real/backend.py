"""Physical runtime backend executing canonical component placement actions."""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter

from darpan.core.action import Action, ActionKind
from darpan.core.event import Event, EventKind
from darpan.core.protocols.backend import BackendContext
from darpan.core.protocols.executor import Executor
from darpan.core.serialization import to_primitive
from darpan.core.state import FlowRouteBinding
from darpan.runtime.action_plan import scale_feasibility
from darpan.runtime.replicas import (
    active_replicas,
    logical_instance_id,
    next_replica_indices,
    replica_instance_id,
)
from darpan.runtime.routing import flow_route_feasibility

from .executors.local import LocalExecutor
from .transport import MAX_AGENT_BINARY_PAYLOAD_BYTES


@dataclass(slots=True)
class _PlacementResources:
    node_id: str
    correlation_id: str
    reserved: tuple[object, ...]
    emitted: list[object] = field(default_factory=list)


class RealBackend:
    supported_action_kinds = frozenset(
        {
            ActionKind.PLACE,
            ActionKind.MIGRATE,
            ActionKind.RESTART,
            ActionKind.SCALE,
            ActionKind.STOP,
        }
    )

    def __init__(
        self,
        executors: Mapping[str, Executor] | None = None,
        *,
        default_executor: Executor | None = None,
        artifact_chunk_size: int = 256 * 1024,
        require_explicit_executor: bool = False,
        network_driver=None,
        physical_control_driver=None,
    ) -> None:
        if not 1 <= artifact_chunk_size <= MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact_chunk_size must be in [1, 4194304]")
        self.executors = dict(executors or {})
        self.default_executor = (
            default_executor if default_executor is not None else LocalExecutor()
        )
        self.require_explicit_executor = require_explicit_executor
        self.network_driver = network_driver
        self.physical_control_driver = physical_control_driver
        self._physical_control_report: dict[str, object] | None = None
        self._physical_control_restored = False
        self.supported_action_kinds = frozenset(
            {
                ActionKind.PLACE,
                ActionKind.MIGRATE,
                ActionKind.RESTART,
                ActionKind.SCALE,
                ActionKind.STOP,
                *(() if network_driver is None else (ActionKind.ROUTE,)),
            }
        )
        self.allowed_node_ids = (
            frozenset(self.executors) if require_explicit_executor else None
        )
        self.artifact_chunk_size = artifact_chunk_size
        self.context: BackendContext | None = None
        self._tasks: set[asyncio.Task] = set()
        self._placements: dict[str, asyncio.Task] = {}
        self._placement_resources: dict[str, _PlacementResources] = {}
        self._placement_failure_reasons: dict[str, dict[str, str]] = {}
        self._resource_condition = asyncio.Condition()
        self._reserved_resources: dict[str, dict[str, float]] = {}
        self._contention_lock = asyncio.Lock()
        self._active_fair_compute: dict[str, dict[str, float]] = {}
        self._contended_compute: set[str] = set()
        self._active_artifact_transfers: dict[str, frozenset[str] | None] = {}
        self._artifact_transfer_nodes: dict[str, tuple[str, str, str]] = {}
        self._contended_artifact_transfers: set[str] = set()

    def _executor_for(self, node_id: str) -> Executor:
        executor = self.executors.get(node_id)
        if executor is not None:
            return executor
        if self.require_explicit_executor:
            raise RuntimeError(
                f"no physical executor configured for cluster node: {node_id}"
            )
        return self.default_executor

    async def start(self, context: BackendContext) -> None:
        self.context = context
        if self.physical_control_driver is not None:
            starter = getattr(self.physical_control_driver, "start", None)
            if starter is not None:
                result = starter()
                if asyncio.iscoroutine(result):
                    await result

    async def prepare_event(self, event: Event) -> None:
        """Enforce a controlled Physical scenario before its canonical event commits."""

        if self.physical_control_driver is None or self.context is None:
            return
        await self.physical_control_driver.prepare_event(event, self.context.state())

    @property
    def physical_control_report(self) -> dict[str, object] | None:
        if self._physical_control_report is not None:
            return dict(self._physical_control_report)
        if self.physical_control_driver is None:
            return None
        return dict(self.physical_control_driver.report())

    async def restore_physical_control(self) -> dict[str, object] | None:
        """Restore and freeze Physical control evidence exactly once."""

        if self.physical_control_driver is None:
            return None
        if self._physical_control_restored:
            return self.physical_control_report
        try:
            await self.physical_control_driver.restore()
        finally:
            self._physical_control_report = dict(self.physical_control_driver.report())
            self._physical_control_restored = True
        return self.physical_control_report

    async def apply(self, action: Action) -> None:
        if self.context is None:
            raise RuntimeError("backend has not started")
        if action.kind == ActionKind.PLACE:
            await self._start_placement(action)
            return
        if action.kind == ActionKind.MIGRATE:
            await self._migrate_long_running(action)
            return
        if action.kind == ActionKind.RESTART:
            await self._restart_long_running(action)
            return
        if action.kind == ActionKind.SCALE:
            await self._scale_long_running(action)
            return
        if action.kind == ActionKind.ROUTE:
            await self._route_flow(action)
            return
        if action.kind == ActionKind.STOP:
            await self._stop_long_running(action)
            return
        raise NotImplementedError(f"real backend does not handle {action.kind}")

    async def _route_flow(self, action: Action) -> None:
        """Apply a future-transfer flow binding through a real network driver."""

        assert self.context is not None
        if self.network_driver is None:
            raise RuntimeError("physical ROUTE requires a NetworkControlDriver")
        state = self.context.state()
        app_instance = str(action.payload["application_instance_id"])
        source_component = str(action.payload["source_component_id"])
        target_component = str(action.payload["target_component_id"])
        path = tuple(str(item) for item in action.payload.get("path", ()))
        requested_links = tuple(str(item) for item in action.payload.get("links", ()))
        feasible, reason, links = flow_route_feasibility(
            state,
            app_instance,
            source_component,
            target_component,
            path,
            requested_links,
        )
        if not feasible:
            raise RuntimeError(reason or "invalid route request")
        application_id = state.application_instances[app_instance]
        now = self.context.now()
        common = {
            "application_id": application_id,
            "application_instance_id": app_instance,
            "source_component_id": source_component,
            "target_component_id": target_component,
            "path": list(path),
            "links": list(links),
            "scope": "future_transfers",
        }
        await self.context.emit(
            Event(
                kind=EventKind.FLOW_ROUTING,
                event_time=now,
                source="runtime.real",
                subject=action.target,
                correlation_id=app_instance,
                causation_id=action.id,
                payload=common,
            )
        )
        if path:
            binding = FlowRouteBinding(
                application_id=application_id,
                application_instance_id=app_instance,
                source_component_id=source_component,
                target_component_id=target_component,
                path=path,
                links=links,
                updated_at=now,
                source=action.source,
            )
            await self.network_driver.bind_route(binding, state)
        else:
            await self.network_driver.clear_route(
                app_instance,
                source_component,
                target_component,
                state,
            )
        await self.context.emit(
            Event(
                kind=EventKind.FLOW_ROUTED,
                event_time=self.context.now(),
                source="runtime.real",
                subject=action.target,
                correlation_id=app_instance,
                causation_id=action.id,
                payload=common,
            )
        )

    async def _scale_long_running(self, action: Action) -> None:
        """Reconcile a stable long-running replica set to a desired count."""

        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        desired = int(action.payload["replicas"])
        feasible, reason = scale_feasibility(state, instance_id, desired)
        if not feasible:
            raise RuntimeError(reason or "invalid scale request")
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        logical_id = logical_instance_id(
            instance.application_instance_id,
            instance.component_id,
        )
        current = active_replicas(state, instance_id)
        now = self.context.now()
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_SCALING,
                event_time=now,
                source="runtime.real",
                subject=logical_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": logical_id,
                    "application_id": app.id,
                    "component_id": instance.component_id,
                    "from_replicas": len(current),
                    "desired_replicas": desired,
                },
            )
        )

        if desired > len(current):
            indices = next_replica_indices(state, instance_id, desired - len(current))
            for replica_index in indices:
                replica_id = replica_instance_id(
                    instance.application_instance_id,
                    instance.component_id,
                    replica_index,
                )
                await self.context.emit(
                    Event(
                        kind=EventKind.COMPONENT_CREATED,
                        event_time=self.context.now(),
                        source="runtime.real",
                        subject=replica_id,
                        correlation_id=instance.application_instance_id,
                        causation_id=action.id,
                        payload={
                            "instance_id": replica_id,
                            "application_id": app.id,
                            "application_instance_id": instance.application_instance_id,
                            "component_id": instance.component_id,
                            "replica_index": replica_index,
                            "scaled_from": logical_id,
                        },
                    )
                )
                await self.context.emit(
                    Event(
                        kind=EventKind.COMPONENT_READY,
                        event_time=self.context.now(),
                        source="runtime.real",
                        subject=replica_id,
                        correlation_id=instance.application_instance_id,
                        causation_id=action.id,
                        payload={
                            "instance_id": replica_id,
                            "replica_index": replica_index,
                            "scaled": True,
                        },
                    )
                )
        else:
            victims = sorted(
                (item for item in current if item.replica_index != 0),
                key=lambda item: item.replica_index,
                reverse=True,
            )[: len(current) - desired]
            if len(victims) != len(current) - desired:
                raise RuntimeError("cannot scale below the primary replica")
            for victim in victims:
                stop = replace(
                    action,
                    kind=ActionKind.STOP,
                    target=victim.id,
                    payload={"instance_id": victim.id, "scaled_in": True},
                )
                await self._stop_long_running(stop)


    async def _start_placement(self, action: Action) -> None:
        """Start one physical placement and return after it is canonically scheduled."""

        instance_id = str(action.payload["instance_id"])
        scheduled = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._execute_placement(action, scheduled=scheduled))
        self._tasks.add(task)
        self._placements[instance_id] = task

        def placement_done(completed: asyncio.Task) -> None:
            self._tasks.discard(completed)
            if self._placements.get(instance_id) is completed:
                self._placements.pop(instance_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(placement_done)
        await scheduled

    async def _migrate_long_running(self, action: Action) -> None:
        """Move a running service/stream without terminally completing it."""

        assert self.context is not None
        instance_id = str(action.payload["instance_id"])
        target_node_id = str(action.payload["node_id"])
        state = self.context.state()
        instance = state.components[instance_id]
        source_node_id = instance.node_id
        if source_node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)

        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_MIGRATING,
                event_time=self.context.now(),
                source="runtime.real",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "from_node_id": source_node_id,
                    "to_node_id": target_node_id,
                    "attempt": instance.attempt + 1,
                    "mode": str(action.payload.get("mode", "restart")),
                },
            )
        )

        task = self._placements.get(instance_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # The normal cancellation path releases reservations and canonical
        # allocations. Keep this fallback for custom executors/tasks.
        await self._release_placement_resources(instance_id)
        self._placement_failure_reasons.pop(instance_id, None)
        await self._start_placement(action)

    async def _restart_long_running(self, action: Action) -> None:
        """Restart a running service/stream on its current physical node."""

        assert self.context is not None
        instance_id = str(action.payload["instance_id"])
        state = self.context.state()
        instance = state.components[instance_id]
        node_id = instance.node_id
        if node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_RESTARTING,
                event_time=self.context.now(),
                source="runtime.real",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "node_id": node_id,
                    "attempt": instance.attempt + 1,
                    "mode": "process",
                },
            )
        )
        task = self._placements.get(instance_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._release_placement_resources(instance_id)
        self._placement_failure_reasons.pop(instance_id, None)
        placement = replace(
            action,
            payload={**dict(action.payload), "node_id": node_id},
        )
        await self._start_placement(placement)

    async def _execute_placement(
        self,
        action: Action,
        *,
        scheduled: asyncio.Future[None] | None = None,
    ) -> None:
        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        node_id = str(action.payload["node_id"])
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        executor = self._executor_for(node_id)
        workspace_id = self._workspace_id(node_id, instance_id)
        tracking_fair_compute = False

        try:
            await self.context.emit(
                Event(
                    kind=EventKind.COMPONENT_SCHEDULED,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=instance_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={"instance_id": instance_id, "node_id": node_id},
                )
            )
        except Exception as exc:
            if scheduled is not None and not scheduled.done():
                scheduled.set_exception(exc)
            raise
        else:
            if scheduled is not None and not scheduled.done():
                scheduled.set_result(None)
        try:
            # A retry reuses the logical component workspace on the same node.
            # Remove declared outputs before staging/execution so a partial
            # failed-attempt artifact can never masquerade as fresh output.
            if instance.attempt > 0 and not component.is_long_running:
                await self._clear_output_artifacts(
                    app,
                    component.id,
                    executor=executor,
                    workspace_id=workspace_id,
                )
            # Input transfer is part of scheduling, not component execution.
            # Keeping it before allocation mirrors Twin semantics and avoids
            # holding scarce compute resources while bytes are still moving.
            await self._stage_input_artifacts(
                app,
                instance,
                node_id=node_id,
                executor=executor,
                workspace_id=workspace_id,
            )
            queue_delay_s = await self._acquire_resources(
                component.resources,
                node_id=node_id,
            )
            self._placement_resources[instance_id] = _PlacementResources(
                node_id=node_id,
                correlation_id=instance.application_instance_id,
                reserved=tuple(component.resources),
            )
            tracking_fair_compute = await self._enter_fair_compute(
                instance_id,
                node_id=node_id,
                resources=component.resources,
            )
            for request in component.resources:
                await self.context.emit(
                    Event(
                        kind=EventKind.RESOURCE_ALLOCATED,
                        event_time=self.context.now(),
                        source="runtime.real",
                        subject=node_id,
                        correlation_id=instance.application_instance_id,
                        payload={
                            "node_id": node_id,
                            "resource": request.name,
                            "amount": request.amount,
                        },
                    )
                )
                allocation = self._placement_resources.get(instance_id)
                if allocation is not None:
                    allocation.emitted.append(request)
            remote_handle = None
            start_execution = getattr(executor, "start_execution", None)
            if start_execution is not None and (component.command or component.image):
                remote_handle = await start_execution(
                    component,
                    workspace_id=workspace_id,
                )
            await self.context.emit(
                Event(
                    kind=EventKind.COMPONENT_STARTED,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=instance_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "instance_id": instance_id,
                        "node_id": node_id,
                        "application_id": app.id,
                        "component_id": component.id,
                        "long_running": component.is_long_running,
                        "queue_delay_s": queue_delay_s,
                    },
                )
            )
            execute_in_workspace = getattr(executor, "execute_in_workspace", None)
            if component.is_long_running and not component.command and not component.image:
                await asyncio.Event().wait()
                raise AssertionError("unreachable long-running component wait")
            if remote_handle is not None:
                result = await remote_handle.wait()
            elif execute_in_workspace is None:
                if self._component_uses_file_artifacts(app, component.id):
                    raise RuntimeError(
                        "file artifact flows require an artifact-capable executor"
                    )
                result = await executor.execute(component)
            else:
                result = await execute_in_workspace(component, workspace_id)
            event_kind = (
                EventKind.COMPONENT_COMPLETED
                if result.success
                else EventKind.COMPONENT_FAILED
            )
            artifact_bytes = None
            if result.success:
                artifact_bytes = await self._output_artifact_size(
                    app,
                    component.id,
                    executor=executor,
                    workspace_id=workspace_id,
                )
            compute_contention = (
                await self._leave_fair_compute(instance_id, node_id=node_id)
                if tracking_fair_compute
                else False
            )
            tracking_fair_compute = False
            await self._release_placement_resources(instance_id)
            await self.context.emit(
                Event(
                    kind=event_kind,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=instance_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "instance_id": instance_id,
                        "node_id": node_id,
                        "application_id": app.id,
                        "component_id": component.id,
                        "duration_s": result.duration_s,
                        "compute_contention_observed": compute_contention,
                        "execution_calibration_eligible": not compute_contention,
                        "return_code": result.return_code,
                        "output_bytes": (
                            result.output_bytes
                            if artifact_bytes is None
                            else artifact_bytes
                        ),
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "measurements": to_primitive(result.measurements),
                        **(
                            {"failure_kind": "execution_failed"}
                            if not result.success
                            else {}
                        ),
                    },
                )
            )
        except asyncio.CancelledError:
            if tracking_fair_compute:
                await self._leave_fair_compute(instance_id, node_id=node_id)
                tracking_fair_compute = False
            await self._release_placement_resources(instance_id)
            failure = self._placement_failure_reasons.pop(instance_id, None)
            if failure is not None:
                await self._emit_placement_failure(
                    instance_id=instance_id,
                    node_id=node_id,
                    application_id=app.id,
                    component_id=component.id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    error=failure["error"],
                    failure_kind=failure["failure_kind"],
                )
            raise
        except Exception as exc:
            if tracking_fair_compute:
                await self._leave_fair_compute(instance_id, node_id=node_id)
                tracking_fair_compute = False
            await self._release_placement_resources(instance_id)
            self._placement_failure_reasons.pop(instance_id, None)
            await self._emit_placement_failure(
                instance_id=instance_id,
                node_id=node_id,
                application_id=app.id,
                component_id=component.id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _emit_placement_failure(
        self,
        *,
        instance_id: str,
        node_id: str,
        application_id: str,
        component_id: str,
        correlation_id: str,
        causation_id: str | None,
        error: str,
        failure_kind: str | None = None,
    ) -> None:
        assert self.context is not None
        current = self.context.state().components.get(instance_id)
        if current is None or current.status in {"completed", "failed"}:
            return
        payload = {
            "instance_id": instance_id,
            "node_id": node_id,
            "application_id": application_id,
            "component_id": component_id,
            "error": error,
        }
        if failure_kind is not None:
            payload["failure_kind"] = failure_kind
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_FAILED,
                event_time=self.context.now(),
                source="runtime.real",
                subject=instance_id,
                correlation_id=correlation_id,
                causation_id=causation_id,
                payload=payload,
            )
        )

    async def node_unavailable(self, node_id: str, *, reason: str = "node offline") -> None:
        """Cancel physical work whose compute or active data path uses a failed node."""

        if self.context is None:
            return
        state = self.context.state()
        affected = {
            instance_id
            for instance_id in self._placements
            if (instance := state.components.get(instance_id)) is not None
            and instance.status in {"scheduled", "running"}
            and instance.node_id == node_id
        }
        affected.update(
            placement_id
            for source, target, placement_id in self._artifact_transfer_nodes.values()
            if node_id in {source, target}
        )
        tasks = []
        for instance_id in affected:
            task = self._placements.get(instance_id)
            if task is None or task.done():
                continue
            self._placement_failure_reasons[instance_id] = {
                "error": f"execution/data-path node {node_id} became unavailable: {reason}",
                "failure_kind": "node_offline",
            }
            task.cancel()
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Defensive fallback for a custom executor/task that completed its
        # cancellation path without producing a terminal component event.
        for instance_id in affected:
            current = self.context.state().components.get(instance_id)
            if current is None or current.status in {"completed", "failed"}:
                self._placement_failure_reasons.pop(instance_id, None)
                continue
            await self._release_placement_resources(instance_id)
            failure = self._placement_failure_reasons.pop(instance_id, None)
            if failure is None:
                continue
            app = self.context.state().applications[current.application_id]
            component = app.component(current.component_id)
            await self._emit_placement_failure(
                instance_id=instance_id,
                node_id=current.node_id or node_id,
                application_id=app.id,
                component_id=component.id,
                correlation_id=current.application_instance_id,
                causation_id=None,
                error=failure["error"],
                failure_kind=failure["failure_kind"],
            )

    async def _enter_fair_compute(
        self,
        instance_id: str,
        *,
        node_id: str,
        resources,
    ) -> bool:
        """Track overlap that can contaminate solo execution calibration."""

        assert self.context is not None
        cpu = self.context.state().nodes[node_id].resources.get("cpu")
        if cpu is None or not cpu.is_shareable:
            return False
        cpu_request = sum(
            float(request.amount)
            for request in resources
            if request.name == "cpu" and request.amount > 0
        )
        if cpu_request <= 0:
            return False
        async with self._contention_lock:
            active = self._active_fair_compute.setdefault(node_id, {})
            active[instance_id] = cpu_request
            if sum(active.values()) > float(cpu.capacity) + 1e-12:
                self._contended_compute.update(active)
        return True

    async def _leave_fair_compute(self, instance_id: str, *, node_id: str) -> bool:
        async with self._contention_lock:
            active = self._active_fair_compute.get(node_id)
            if active is not None:
                active.pop(instance_id, None)
                if not active:
                    self._active_fair_compute.pop(node_id, None)
            contended = instance_id in self._contended_compute
            self._contended_compute.discard(instance_id)
            return contended

    @staticmethod
    def _transfer_paths_overlap(
        left: frozenset[str] | None,
        right: frozenset[str] | None,
    ) -> bool:
        # Empty paths are node-local copies and therefore do not consume a
        # continuum link. ``None`` means the physical/logical route is unknown;
        # stay conservative in that case rather than contaminating calibration.
        if left == frozenset() or right == frozenset():
            return False
        if left is None or right is None:
            return True
        return bool(left.intersection(right))

    async def _enter_artifact_transfer(
        self,
        transfer_id: str,
        *,
        logical_links: frozenset[str] | None,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        placement_instance_id: str | None = None,
    ) -> None:
        async with self._contention_lock:
            for active_id, active_links in self._active_artifact_transfers.items():
                if not self._transfer_paths_overlap(logical_links, active_links):
                    continue
                self._contended_artifact_transfers.add(active_id)
                self._contended_artifact_transfers.add(transfer_id)
            self._active_artifact_transfers[transfer_id] = logical_links
            if (
                source_node_id is not None
                and target_node_id is not None
                and placement_instance_id is not None
            ):
                self._artifact_transfer_nodes[transfer_id] = (
                    source_node_id,
                    target_node_id,
                    placement_instance_id,
                )

    async def _leave_artifact_transfer(self, transfer_id: str) -> bool:
        async with self._contention_lock:
            self._active_artifact_transfers.pop(transfer_id, None)
            self._artifact_transfer_nodes.pop(transfer_id, None)
            contended = transfer_id in self._contended_artifact_transfers
            self._contended_artifact_transfers.discard(transfer_id)
            return contended

    def _logical_transfer_links(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        size_bytes: int,
    ) -> frozenset[str] | None:
        """Resolve the best declared continuum path for calibration auditing.

        This is deliberately only an *audit* path. Real bytes still travel via
        the configured executors/agents; the path is used to decide whether two
        measured transfers plausibly shared a Darpan-declared link. If no route
        can be resolved we return ``None`` and contention tracking falls back to
        the conservative behavior used by earlier releases.
        """

        assert self.context is not None
        if source_node_id == target_node_id:
            return frozenset()
        state = self.context.state()
        graph: dict[str, list[tuple[float, str, str]]] = {}
        for link in state.links.values():
            if link.status != "up":
                continue
            latency = link.measurements.get("network.latency_ms")
            bandwidth = link.measurements.get("network.bandwidth_mbps")
            latency_ms = (
                float(link.spec.latency_ms)
                if latency is None
                else max(0.0, float(latency.value))
            )
            bandwidth_mbps = (
                float(link.spec.bandwidth_mbps)
                if bandwidth is None
                else max(1e-9, float(bandwidth.value))
            )
            serialization_s = (
                0.0
                if size_bytes <= 0 or bandwidth_mbps == float("inf")
                else size_bytes * 8 / (bandwidth_mbps * 1_000_000)
            )
            cost = latency_ms / 1000.0 + serialization_s
            graph.setdefault(link.spec.source, []).append(
                (cost, link.spec.target, link.spec.id)
            )
            if link.spec.bidirectional:
                graph.setdefault(link.spec.target, []).append(
                    (cost, link.spec.source, link.spec.id)
                )

        queue: list[tuple[float, str, tuple[str, ...]]] = [
            (0.0, source_node_id, ())
        ]
        best = {source_node_id: 0.0}
        while queue:
            cost, node_id, links = heapq.heappop(queue)
            if node_id == target_node_id:
                return frozenset(links)
            if cost > best.get(node_id, float("inf")):
                continue
            for edge_cost, neighbor, link_id in graph.get(node_id, ()):
                candidate = cost + edge_cost
                if candidate >= best.get(neighbor, float("inf")):
                    continue
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor, (*links, link_id)))
        return None

    async def _acquire_resources(self, requests, *, node_id: str) -> float:
        """Reserve physical-node capacity without overcommitting concurrent placements."""

        assert self.context is not None
        requests = tuple(requests)
        if not requests:
            return 0.0
        started = perf_counter()
        async with self._resource_condition:
            while True:
                state = self.context.state()
                node = state.nodes[node_id]
                if node.status != "online":
                    raise RuntimeError(f"node {node_id} is not online")
                reserved = self._reserved_resources.setdefault(node_id, {})
                missing = [
                    request.name
                    for request in requests
                    if request.name not in node.resources
                ]
                if missing:
                    raise RuntimeError(
                        f"node {node_id} has no resource {', '.join(sorted(set(missing)))}"
                    )
                impossible = [
                    request
                    for request in requests
                    if float(request.amount)
                    > node.effective_resource_capacity(request.name) + 1e-12
                ]
                if impossible:
                    request = impossible[0]
                    raise RuntimeError(
                        f"request exceeds effective {request.name} capacity on "
                        f"{node_id}: need {request.amount}, capacity "
                        f"{node.effective_resource_capacity(request.name)}"
                    )
                if all(
                    (
                        True
                        if node.resources[request.name].is_shareable
                        else reserved.get(request.name, 0.0) + float(request.amount)
                        <= node.effective_resource_capacity(request.name) + 1e-12
                    )
                    for request in requests
                ):
                    for request in requests:
                        reserved[request.name] = (
                            reserved.get(request.name, 0.0) + float(request.amount)
                        )
                    return max(0.0, perf_counter() - started)
                await self._resource_condition.wait()

    async def resource_capacity_changed(self, node_id: str) -> None:
        """Wake placements waiting on an updated physical capacity measurement."""

        del node_id
        async with self._resource_condition:
            self._resource_condition.notify_all()

    async def _emit_resource_releases(
        self,
        requests,
        *,
        node_id: str,
        correlation_id: str,
    ) -> None:
        assert self.context is not None
        for request in tuple(requests):
            await self.context.emit(
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=node_id,
                    correlation_id=correlation_id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                    },
                )
            )

    async def _release_reserved_capacity(self, requests, *, node_id: str) -> None:
        requests = tuple(requests)
        if not requests:
            return
        async with self._resource_condition:
            reserved = self._reserved_resources.setdefault(node_id, {})
            for request in requests:
                remaining = reserved.get(request.name, 0.0) - float(request.amount)
                if remaining <= 1e-12:
                    reserved.pop(request.name, None)
                else:
                    reserved[request.name] = remaining
            self._resource_condition.notify_all()

    async def _release_resources(
        self,
        requests,
        *,
        node_id: str,
        correlation_id: str,
    ) -> None:
        requests = tuple(requests)
        await self._emit_resource_releases(
            requests,
            node_id=node_id,
            correlation_id=correlation_id,
        )
        await self._release_reserved_capacity(requests, node_id=node_id)

    async def _release_placement_resources(self, instance_id: str) -> None:
        allocation = self._placement_resources.pop(instance_id, None)
        if allocation is None:
            return
        await self._emit_resource_releases(
            allocation.emitted,
            node_id=allocation.node_id,
            correlation_id=allocation.correlation_id,
        )
        await self._release_reserved_capacity(
            allocation.reserved,
            node_id=allocation.node_id,
        )

    async def _stop_long_running(self, action: Action) -> None:
        assert self.context is not None
        instance_id = str(action.payload["instance_id"])
        state = self.context.state()
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        node_id = instance.node_id
        if node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")

        task = self._placements.get(instance_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        current = self.context.state().components[instance_id]
        if current.status in {"completed", "failed"}:
            return
        now = self.context.now()
        await self._release_placement_resources(instance_id)
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_COMPLETED,
                event_time=now,
                source="runtime.real",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "node_id": node_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "duration_s": max(
                        0.0,
                        now - (current.started_at if current.started_at is not None else now),
                    ),
                    "return_code": 0,
                    "output_bytes": 0,
                    "stopped": True,
                },
            )
        )

    @staticmethod
    def _workspace_id(node_id: str, instance_id: str) -> str:
        return f"{node_id}|{instance_id}"

    @staticmethod
    def _component_uses_file_artifacts(app, component_id: str) -> bool:
        return any(
            flow.artifact is not None
            and (flow.source == component_id or flow.target == component_id)
            for flow in app.flows
        )

    async def _clear_output_artifacts(
        self,
        app,
        component_id: str,
        *,
        executor,
        workspace_id: str,
    ) -> None:
        outputs = {
            flow.artifact
            for flow in app.flows
            if flow.source == component_id and flow.artifact is not None
        }
        if not outputs:
            return
        delete_file = getattr(executor, "delete_file", None)
        if delete_file is None:
            raise RuntimeError("executor cannot clean declared output artifacts")
        for path in sorted(outputs):
            await delete_file(workspace_id, path)

    async def _output_artifact_size(
        self,
        app,
        component_id: str,
        *,
        executor,
        workspace_id: str,
    ) -> int | None:
        paths = {
            flow.artifact
            for flow in app.flows
            if flow.source == component_id and flow.artifact is not None
        }
        if not paths:
            return None
        stat_file = getattr(executor, "stat_file", None)
        if stat_file is None:
            raise RuntimeError("executor cannot inspect declared output artifacts")
        total = 0
        for path in sorted(paths):
            total += int(await stat_file(workspace_id, path))
        return total

    async def _stage_input_artifacts(
        self,
        app,
        instance,
        *,
        node_id: str,
        executor,
        workspace_id: str,
    ) -> None:
        assert self.context is not None
        incoming = [
            flow
            for flow in app.flows
            if flow.target == instance.component_id and flow.artifact is not None
        ]
        if not incoming:
            return
        put_file = getattr(executor, "put_file", None)
        write_chunk = getattr(executor, "write_chunk", None)
        if put_file is None and write_chunk is None:
            raise RuntimeError("target executor cannot receive declared input artifacts")

        state = self.context.state()
        for flow in incoming:
            predecessor_id = f"{instance.application_instance_id}:{flow.source}"
            predecessor = state.components[predecessor_id]
            if predecessor.node_id is None:
                raise RuntimeError(f"artifact source has no node: {predecessor_id}")
            source_executor = self._executor_for(predecessor.node_id)
            get_file = getattr(source_executor, "get_file", None)
            read_chunk = getattr(source_executor, "read_chunk", None)
            stat_file = getattr(source_executor, "stat_file", None)
            if get_file is None and (read_chunk is None or stat_file is None):
                raise RuntimeError("source executor cannot provide declared output artifacts")
            source_workspace = self._workspace_id(predecessor.node_id, predecessor_id)
            target_path = flow.artifact_target
            assert flow.artifact is not None and target_path is not None

            transfer_id = f"{predecessor_id}->{instance.id}"
            transfer_size = (
                int(await stat_file(source_workspace, flow.artifact))
                if stat_file is not None
                else None
            )
            transfer_mode = self._artifact_transfer_mode(source_executor, executor)
            path_representative = self._network_path_representative(
                source_executor,
                executor,
                transfer_mode=transfer_mode,
            )
            route = state.flow_route(
                instance.application_instance_id,
                flow.source,
                flow.target,
            )
            route_metadata = (
                {}
                if route is None
                else {
                    "route_bound": True,
                    "route_path": list(route.path),
                    "route_links": list(route.links),
                }
            )
            logical_links = (
                frozenset(route.links)
                if route is not None
                else self._logical_transfer_links(
                    predecessor.node_id,
                    node_id,
                    size_bytes=0 if transfer_size is None else transfer_size,
                )
            )
            contention_scope = (
                "conservative-global" if logical_links is None else "logical-path"
            )
            started = perf_counter()
            await self._enter_artifact_transfer(
                transfer_id,
                logical_links=logical_links,
                source_node_id=predecessor.node_id,
                target_node_id=node_id,
                placement_instance_id=instance.id,
            )
            await self.context.emit(
                Event(
                    kind=EventKind.DATA_TRANSFER_STARTED,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=transfer_id,
                    correlation_id=instance.application_instance_id,
                    payload={
                        "source_instance_id": predecessor_id,
                        "target_instance_id": instance.id,
                        "application_id": app.id,
                        "source_component_id": flow.source,
                        "target_component_id": flow.target,
                        "source_node_id": predecessor.node_id,
                        "target_node_id": node_id,
                        "artifact": flow.artifact,
                        "target_path": target_path,
                        "contention_scope": contention_scope,
                        "transport": transfer_mode,
                        "network_path_representative": path_representative,
                        **route_metadata,
                        **(
                            {}
                            if logical_links is None
                            else {"logical_links": sorted(logical_links)}
                        ),
                        **({} if transfer_size is None else {"size_bytes": transfer_size}),
                    },
                )
            )
            try:
                size_bytes, direct_duration_s = await self._copy_artifact(
                    source_executor,
                    source_workspace,
                    flow.artifact,
                    executor,
                    workspace_id,
                    target_path,
                    transfer_mode=transfer_mode,
                    expected_size=transfer_size,
                )
                control_duration_s = max(0.0, perf_counter() - started)
                transfer_duration_s = (
                    control_duration_s
                    if direct_duration_s is None
                    else max(0.0, direct_duration_s)
                )
            except asyncio.CancelledError:
                transfer_contended = await self._leave_artifact_transfer(transfer_id)
                failure = self._placement_failure_reasons.get(instance.id)
                error = (
                    "artifact transfer cancelled"
                    if failure is None
                    else failure["error"]
                )
                payload = {
                    "source_instance_id": predecessor_id,
                    "target_instance_id": instance.id,
                    "application_id": app.id,
                    "source_component_id": flow.source,
                    "target_component_id": flow.target,
                    "source_node_id": predecessor.node_id,
                    "target_node_id": node_id,
                    "artifact": flow.artifact,
                    "contention_scope": contention_scope,
                    "transport": transfer_mode,
                    "network_contention_observed": transfer_contended,
                    "network_path_representative": path_representative,
                    "network_calibration_eligible": False,
                    **route_metadata,
                    "error": error,
                }
                if failure is not None:
                    payload["failure_kind"] = failure["failure_kind"]
                if logical_links is not None:
                    payload["logical_links"] = sorted(logical_links)
                await self.context.emit(
                    Event(
                        kind=EventKind.DATA_TRANSFER_FAILED,
                        event_time=self.context.now(),
                        source="runtime.real",
                        subject=transfer_id,
                        correlation_id=instance.application_instance_id,
                        payload=payload,
                    )
                )
                raise
            except Exception as exc:
                transfer_contended = await self._leave_artifact_transfer(transfer_id)
                await self.context.emit(
                    Event(
                        kind=EventKind.DATA_TRANSFER_FAILED,
                        event_time=self.context.now(),
                        source="runtime.real",
                        subject=transfer_id,
                        correlation_id=instance.application_instance_id,
                        payload={
                            "source_instance_id": predecessor_id,
                            "target_instance_id": instance.id,
                            "application_id": app.id,
                            "source_component_id": flow.source,
                            "target_component_id": flow.target,
                            "artifact": flow.artifact,
                            "contention_scope": contention_scope,
                            "transport": transfer_mode,
                            **route_metadata,
                            **(
                                {}
                                if logical_links is None
                                else {"logical_links": sorted(logical_links)}
                            ),
                            "network_contention_observed": transfer_contended,
                            "network_path_representative": path_representative,
                            "network_calibration_eligible": (
                                not transfer_contended and path_representative
                            ),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
                raise
            transfer_contended = await self._leave_artifact_transfer(transfer_id)
            await self.context.emit(
                Event(
                    kind=EventKind.DATA_TRANSFER_COMPLETED,
                    event_time=self.context.now(),
                    source="runtime.real",
                    subject=transfer_id,
                    correlation_id=instance.application_instance_id,
                    payload={
                        "source_instance_id": predecessor_id,
                        "target_instance_id": instance.id,
                        "application_id": app.id,
                        "source_component_id": flow.source,
                        "target_component_id": flow.target,
                        "source_node_id": predecessor.node_id,
                        "target_node_id": node_id,
                        "artifact": flow.artifact,
                        "target_path": target_path,
                        "size_bytes": size_bytes,
                        "duration_s": transfer_duration_s,
                        "control_duration_s": control_duration_s,
                        "control_overhead_s": max(
                            0.0, control_duration_s - transfer_duration_s
                        ),
                        "measurement_scope": (
                            "source-agent-data-plane"
                            if direct_duration_s is not None
                            else (
                                "agent-direct-completion-recovery"
                                if transfer_mode == "agent-direct"
                                else "controller-stream"
                            )
                        ),
                        "completion_recovered": (
                            transfer_mode == "agent-direct"
                            and direct_duration_s is None
                        ),
                        "contention_scope": contention_scope,
                        "transport": transfer_mode,
                        **route_metadata,
                        **(
                            {}
                            if logical_links is None
                            else {"logical_links": sorted(logical_links)}
                        ),
                        "network_contention_observed": transfer_contended,
                        "network_path_representative": path_representative,
                        "network_calibration_eligible": (
                            not transfer_contended
                            and path_representative
                            and not (
                                transfer_mode == "agent-direct"
                                and direct_duration_s is None
                            )
                        ),
                    },
                )
            )

    @staticmethod
    def _artifact_transfer_mode(source_executor, target_executor) -> str:
        can_direct = getattr(source_executor, "can_direct_transfer_to", None)
        if can_direct is not None and bool(can_direct(target_executor)):
            return "agent-direct"
        return "controller-stream"

    @staticmethod
    def _network_path_representative(
        source_executor,
        target_executor,
        *,
        transfer_mode: str,
    ) -> bool:
        if transfer_mode == "agent-direct":
            return True
        # Remote-to-remote fallback transfers are deliberately controller
        # hairpins, so their wall time must not train a source->target link
        # model. Local/custom executors retain the historical behavior because
        # they are commonly used as physical test harnesses.
        return not (
            bool(getattr(source_executor, "is_remote_agent", False))
            and bool(getattr(target_executor, "is_remote_agent", False))
        )

    async def _copy_artifact(
        self,
        source_executor,
        source_workspace: str,
        source_path: str,
        target_executor,
        target_workspace: str,
        target_path: str,
        *,
        transfer_mode: str,
        expected_size: int | None = None,
    ) -> tuple[int, float | None]:
        if transfer_mode == "agent-direct":
            copy_file_to = getattr(source_executor, "copy_file_to", None)
            if copy_file_to is None:
                raise RuntimeError("executor advertised direct transfer without implementation")
            size, duration_s = await copy_file_to(
                source_workspace,
                source_path,
                target_executor,
                target_workspace,
                target_path,
                chunk_size=self.artifact_chunk_size,
                expected_size=expected_size,
            )
            return int(size), None if duration_s is None else float(duration_s)

        read_chunk = getattr(source_executor, "read_chunk", None)
        stat_file = getattr(source_executor, "stat_file", None)
        write_chunk = getattr(target_executor, "write_chunk", None)
        if read_chunk is not None and stat_file is not None and write_chunk is not None:
            size = int(await stat_file(source_workspace, source_path))
            if size == 0:
                await write_chunk(
                    target_workspace,
                    target_path,
                    offset=0,
                    data=b"",
                    truncate=True,
                )
                return 0, None
            offset = 0
            while offset < size:
                chunk, reported_size = await read_chunk(
                    source_workspace,
                    source_path,
                    offset=offset,
                    limit=min(self.artifact_chunk_size, size - offset),
                )
                if int(reported_size) != size:
                    raise OSError("artifact changed size during transfer")
                if not chunk:
                    raise OSError("empty artifact chunk before EOF")
                await write_chunk(
                    target_workspace,
                    target_path,
                    offset=offset,
                    data=chunk,
                    truncate=offset == 0,
                )
                offset += len(chunk)
            return size, None

        get_file = getattr(source_executor, "get_file", None)
        put_file = getattr(target_executor, "put_file", None)
        if get_file is None or put_file is None:
            raise RuntimeError("artifact executors do not share a compatible transfer API")
        data = await get_file(source_workspace, source_path)
        await put_file(target_workspace, target_path, data)
        return len(data), None

    async def close(self) -> None:
        errors: list[BaseException] = []
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks.clear()
        self._placements.clear()
        self._placement_resources.clear()
        self._placement_failure_reasons.clear()
        self._reserved_resources.clear()
        self._active_fair_compute.clear()
        self._contended_compute.clear()
        self._artifact_transfer_nodes.clear()
        self._active_artifact_transfers.clear()
        self._contended_artifact_transfers.clear()
        if self.physical_control_driver is not None:
            try:
                await self.restore_physical_control()
            except BaseException as exc:
                errors.append(exc)
        # BackendContext owns bound Session callables. Dropping it here breaks
        # Session -> Backend -> Context -> Session cycles deterministically.
        self.context = None
        if errors:
            raise ExceptionGroup("errors while restoring physical control", errors)
