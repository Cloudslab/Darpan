"""Default in-memory Digital Twin runtime backend."""

from __future__ import annotations

from dataclasses import replace

from darpan.core.action import Action, ActionKind
from darpan.core.event import Event, EventKind
from darpan.core.protocols.backend import BackendContext
from darpan.core.protocols.simulation import SimulationKernel
from darpan.runtime.action_plan import scale_feasibility
from darpan.runtime.clock import VirtualClock
from darpan.runtime.replicas import (
    active_replicas,
    logical_instance_id,
    next_replica_indices,
    replica_instance_id,
)
from darpan.runtime.routing import flow_route_feasibility

from .compute_scheduler import MaxMinComputeScheduler, TwinComputeJob
from .kernel import DiscreteEventQueue
from .models.artifact import ArtifactSizeModel
from .models.execution import ExecutionTimeModel
from .models.network import NetworkDelayModel
from .models.queue import QueueDelayModel
from .models.registry import ModelRegistry
from .network_scheduler import MaxMinNetworkScheduler, TwinTransfer
from .uncertainty import combine_uncertainties

_NETWORK_TICK_KIND = "runtime.twin.network_tick"
_COMPUTE_TICK_KIND = "runtime.twin.compute_tick"


class TwinBackend:
    """Discrete-event backend sharing exactly the same canonical events as Real."""

    supported_action_kinds = frozenset(
        {
            ActionKind.PLACE,
            ActionKind.MIGRATE,
            ActionKind.RESTART,
            ActionKind.SCALE,
            ActionKind.ROUTE,
            ActionKind.STOP,
        }
    )

    def __init__(
        self,
        *,
        clock: VirtualClock | None = None,
        models: ModelRegistry | None = None,
        injected_events: tuple[Event, ...] = (),
        kernel: SimulationKernel | None = None,
    ) -> None:
        self.clock = clock if clock is not None else VirtualClock()
        self.models = models if models is not None else ModelRegistry(
            [
                ExecutionTimeModel(),
                ArtifactSizeModel(),
                NetworkDelayModel(),
                QueueDelayModel(),
            ]
        )
        self.context: BackendContext | None = None
        self.kernel = kernel if kernel is not None else DiscreteEventQueue()
        # ``queue`` remains as a compatibility alias for v0.1 extensions.
        self.queue = self.kernel
        self._reservations: dict[str, list[dict[str, object]]] = {}
        self._network_scheduler = MaxMinNetworkScheduler()
        self._pending_network: dict[str, dict[str, object]] = {}
        self._compute_scheduler = MaxMinComputeScheduler()
        self._pending_compute: dict[str, dict[str, object]] = {}
        self._waiting_compute: dict[str, dict[str, object]] = {}
        self._component_event_ids: dict[str, set[str]] = {}
        self._event_component: dict[str, str] = {}
        self._cancelled_event_ids: set[str] = set()
        self._injected_events = tuple(injected_events)
        self._advance_depth = 0

    async def start(self, context: BackendContext) -> None:
        self.context = context
        for event in self._injected_events:
            if event.event_time <= self.clock.now():
                await self.context.emit(event)
            else:
                self.queue.schedule(event, event.event_time)

    async def apply(self, action: Action) -> None:
        if self.context is None:
            raise RuntimeError("Twin backend has not started")
        if action.kind == ActionKind.PLACE:
            await self._place(action)
        elif action.kind == ActionKind.MIGRATE:
            await self._migrate_long_running(action)
        elif action.kind == ActionKind.RESTART:
            await self._restart_long_running(action)
        elif action.kind == ActionKind.SCALE:
            await self._scale_long_running(action)
        elif action.kind == ActionKind.ROUTE:
            await self._route_flow(action)
        elif action.kind == ActionKind.STOP:
            await self._stop_long_running(action)
        else:
            raise NotImplementedError(f"Twin backend does not handle {action.kind}")
        if self._advance_depth == 0:
            await self._advance_until_decision()

    async def _route_flow(self, action: Action) -> None:
        """Bind future virtual transfers of one logical application flow."""

        assert self.context is not None
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
                event_time=self.clock.now(),
                source="runtime.twin",
                subject=action.target,
                correlation_id=app_instance,
                causation_id=action.id,
                payload=common,
            )
        )
        await self.context.emit(
            Event(
                kind=EventKind.FLOW_ROUTED,
                event_time=self.clock.now(),
                source="runtime.twin",
                subject=action.target,
                correlation_id=app_instance,
                causation_id=action.id,
                payload=common,
            )
        )

    async def _scale_long_running(self, action: Action) -> None:
        """Reconcile one long-running virtual replica set."""

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
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_SCALING,
                event_time=self.clock.now(),
                source="runtime.twin",
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
                        event_time=self.clock.now(),
                        source="runtime.twin",
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
                        event_time=self.clock.now(),
                        source="runtime.twin",
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

    def schedule_event(self, event: Event, at: float) -> None:
        if at < self.clock.now():
            raise ValueError("cannot schedule Twin event in the past")
        self.queue.schedule(event, at)

    async def _place(self, action: Action) -> None:
        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        node_id = str(action.payload["node_id"])
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        now = self.clock.now()

        cpu_request = next(
            (item.amount for item in component.resources if item.name == "cpu"), 1.0
        )
        execution = self.models.get("execution").predict(
            {
                "application_id": app.id,
                "component_id": component.id,
                "node_id": node_id,
                "work_units": component.work_units,
                "cpu_request": cpu_request,
            },
            state,
        )
        output_hint = max(
            (flow.data_size_bytes for flow in app.flows if flow.source == component.id),
            default=0,
        )
        output_size = self.models.get("artifact_size").predict(
            {
                "application_id": app.id,
                "component_id": component.id,
                "hint_bytes": output_hint,
            },
            state,
        )

        transfers: list[TwinTransfer] = []
        transfer_uncertainties: list[float | None] = []
        initial_transfer_delay = 0.0
        for predecessor_id in app.predecessors(instance.component_id):
            predecessor = state.components[
                f"{instance.application_instance_id}:{predecessor_id}"
            ]
            if predecessor.node_id is None or predecessor.node_id == node_id:
                continue
            flow = app.flow(predecessor_id, instance.component_id)
            route = state.flow_route(
                instance.application_instance_id,
                predecessor_id,
                instance.component_id,
            )
            hint_bytes = 0 if flow is None else flow.data_size_bytes
            artifact = self.models.get("artifact_size").predict(
                {
                    "application_id": app.id,
                    "component_id": predecessor_id,
                    "artifact": None if flow is None else flow.artifact,
                    "hint_bytes": hint_bytes,
                },
                state,
            )
            prediction = self.models.get("network").predict(
                {
                    "source": predecessor.node_id,
                    "target": node_id,
                    "size_bytes": int(artifact.estimate),
                    "earliest_start": now,
                    "reservations": (),
                    **({} if route is None else {"path": route.path, "links": route.links}),
                },
                state,
            )
            if prediction.estimate == float("inf"):
                raise RuntimeError(
                    f"no Twin network path {predecessor.node_id} -> {node_id}"
                )
            initial_transfer_delay = max(
                initial_transfer_delay,
                float(prediction.estimate),
            )
            transfer_uncertainty = combine_uncertainties(
                artifact.uncertainty, prediction.uncertainty
            )
            transfer_uncertainties.append(transfer_uncertainty)
            links = tuple(str(item) for item in prediction.metadata.get("links", ()))
            if not links:
                continue
            propagation_s = float(prediction.metadata.get("propagation_s", 0.0))
            serialization_s = max(
                0.0,
                float(prediction.metadata.get("serialization_s", prediction.estimate)),
            )
            bottleneck_mbps = float(
                prediction.metadata.get("bottleneck_mbps", float("inf"))
            )
            work_bits = (
                0.0
                if bottleneck_mbps == float("inf")
                else serialization_s * bottleneck_mbps * 1_000_000
            )
            transfer_id = f"{instance_id}<-{predecessor.id}"
            event_payload = {
                "source_instance_id": predecessor.id,
                "target_instance_id": instance.id,
                "application_id": app.id,
                "source_component_id": predecessor_id,
                "target_component_id": component.id,
                "source_node_id": predecessor.node_id,
                "target_node_id": node_id,
                "artifact": None if flow is None else flow.artifact,
                "target_path": None if flow is None else flow.artifact_target,
                "size_bytes": int(artifact.estimate),
                "model_uncertainty": transfer_uncertainty,
                "path": prediction.metadata.get("path", []),
                "links": list(links),
                "route_bound": bool(prediction.metadata.get("route_bound", False)),
                "predicted": True,
            }
            transfers.append(
                TwinTransfer(
                    id=transfer_id,
                    links=links,
                    size_bytes=int(artifact.estimate),
                    started_at=now,
                    ready_at=now + propagation_s,
                    work_bits=work_bits,
                    baseline_duration_s=float(prediction.estimate),
                    payload={
                        "placement_instance_id": instance_id,
                        "emit_events": flow is not None and flow.artifact is not None,
                        "event_payload": event_payload,
                        "uncertainty": transfer_uncertainty,
                        "propagation_s": propagation_s,
                    },
                )
            )

        preliminary_queue = self._predict_queue(
            instance_id=instance_id,
            node_id=node_id,
            duration_s=(
                float("inf") if component.is_long_running else float(execution.estimate)
            ),
            earliest_start=now + initial_transfer_delay,
        )
        model_uncertainty = combine_uncertainties(
            execution.uncertainty,
            output_size.uncertainty,
            preliminary_queue.uncertainty,
            *transfer_uncertainties,
        )
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_SCHEDULED,
                event_time=now,
                source="runtime.twin",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "node_id": node_id,
                    "predicted_duration_s": float(execution.estimate),
                    "predicted_input_transfer_s": initial_transfer_delay,
                    "model_uncertainty": model_uncertainty,
                    "predicted_output_bytes": int(output_size.estimate),
                    "predicted_queue_delay_s": float(preliminary_queue.estimate),
                },
            )
        )

        if not transfers:
            await self._schedule_compute(
                action,
                execution=execution,
                output_size=output_size,
                transfer_uncertainties=transfer_uncertainties,
            )
            return

        self._pending_network[instance_id] = {
            "action": action,
            "execution": execution,
            "output_size": output_size,
            "transfer_uncertainties": tuple(transfer_uncertainties),
            "remaining": {transfer.id for transfer in transfers},
        }
        for transfer in transfers:
            if bool(transfer.payload["emit_events"]):
                payload = dict(transfer.payload["event_payload"])
                payload.update(
                    {
                        "predicted_duration_s": transfer.baseline_duration_s,
                        "propagation_delay_s": float(transfer.payload["propagation_s"]),
                        "sharing_policy": "max-min",
                    }
                )
                await self.context.emit(
                    Event(
                        kind=EventKind.DATA_TRANSFER_STARTED,
                        event_time=now,
                        source="runtime.twin",
                        subject=transfer.id,
                        correlation_id=instance.application_instance_id,
                        causation_id=action.id,
                        payload=payload,
                    )
                )
            self._network_scheduler.add(transfer, now=now, state=self.context.state())
        self._schedule_network_tick()

    def _predict_queue(
        self,
        *,
        instance_id: str,
        node_id: str,
        duration_s: float,
        earliest_start: float,
    ):
        assert self.context is not None
        state = self.context.state()
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        now = self.clock.now()
        reservations = [
            item
            for item in self._reservations.get(node_id, [])
            if float(item["end"]) > now
        ]
        self._reservations[node_id] = reservations
        return self.models.get("queue").predict(
            {
                "node_id": node_id,
                "earliest_start": earliest_start,
                "duration_s": duration_s,
                "requests": {
                    item.name: float(item.amount) for item in component.resources
                },
                "reservations": reservations,
            },
            state,
        )

    def _fair_cpu_request(self, instance_id: str, node_id: str) -> float | None:
        """Return the component CPU cap when the target opts into fair sharing."""

        assert self.context is not None
        state = self.context.state()
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        cpu_requests = [
            float(item.amount)
            for item in component.resources
            if item.name == "cpu" and item.amount > 0
        ]
        if not cpu_requests:
            return None
        node = state.nodes[node_id]
        cpu_state = node.resources.get("cpu")
        if cpu_state is None or not cpu_state.is_shareable:
            return None
        cpu_request = sum(cpu_requests)
        if cpu_request > node.effective_resource_capacity("cpu") + 1e-12:
            return None
        return cpu_request

    def _schedule_component_event(
        self,
        instance_id: str,
        event: Event,
        at: float,
    ) -> None:
        self._component_event_ids.setdefault(instance_id, set()).add(event.id)
        self._event_component[event.id] = instance_id
        self.queue.schedule(event, at)

    def _consume_component_event(self, event: Event) -> bool:
        instance_id = self._event_component.pop(event.id, None)
        if instance_id is not None:
            ids = self._component_event_ids.get(instance_id)
            if ids is not None:
                ids.discard(event.id)
                if not ids:
                    self._component_event_ids.pop(instance_id, None)
        if event.id in self._cancelled_event_ids:
            self._cancelled_event_ids.discard(event.id)
            return True
        return False

    def _cancel_component_events(self, instance_id: str) -> None:
        ids = self._component_event_ids.pop(instance_id, set())
        self._cancelled_event_ids.update(ids)
        for event_id in ids:
            self._event_component.pop(event_id, None)

    async def _schedule_compute(
        self,
        action: Action,
        *,
        execution,
        output_size,
        transfer_uncertainties: list[float | None] | tuple[float | None, ...],
    ) -> None:
        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        node_id = str(action.payload["node_id"])
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        now = self.clock.now()
        duration_s = (
            float("inf") if component.is_long_running else float(execution.estimate)
        )
        fair_cpu = self._fair_cpu_request(instance_id, node_id)
        queue = self._predict_queue(
            instance_id=instance_id,
            node_id=node_id,
            duration_s=duration_s,
            earliest_start=now,
        )
        if queue.estimate == float("inf"):
            if fair_cpu is not None:
                self._waiting_compute[instance_id] = {
                    "action": action,
                    "execution": execution,
                    "output_size": output_size,
                    "transfer_uncertainties": tuple(transfer_uncertainties),
                }
                return
            raise RuntimeError(f"no Twin resource window available on {node_id}")
        self._waiting_compute.pop(instance_id, None)
        start_at = now + float(queue.estimate)
        finish_at = start_at + float(execution.estimate)
        reservation_end = float("inf") if component.is_long_running else finish_at
        requests = {item.name: float(item.amount) for item in component.resources}
        reservation_resources = (
            {
                name: amount
                for name, amount in requests.items()
                if not state.nodes[node_id].resources[name].is_shareable
            }
            if fair_cpu is not None
            else requests
        )
        if reservation_resources:
            self._reservations.setdefault(node_id, []).append(
                {
                    "instance_id": instance_id,
                    "start": start_at,
                    "end": (float("inf") if fair_cpu is not None else reservation_end),
                    "resources": reservation_resources,
                }
            )
        model_uncertainty = combine_uncertainties(
            execution.uncertainty,
            output_size.uncertainty,
            queue.uncertainty,
            *transfer_uncertainties,
        )

        if fair_cpu is not None:
            self._pending_compute[instance_id] = {
                "action": action,
                "execution": execution,
                "output_size": output_size,
                "model_uncertainty": model_uncertainty,
                "cpu_request": fair_cpu,
                "queue_delay_s": float(queue.estimate),
                "long_running": component.is_long_running,
            }
            for request in component.resources:
                self._schedule_component_event(
                    instance_id,
                    Event(
                        kind=EventKind.RESOURCE_ALLOCATED,
                        event_time=start_at,
                        source="runtime.twin",
                        subject=node_id,
                        correlation_id=instance.application_instance_id,
                        payload={
                            "node_id": node_id,
                            "resource": request.name,
                            "amount": request.amount,
                            "scheduling": "fair",
                        },
                    ),
                    start_at,
                )
            self._schedule_component_event(
                instance_id,
                Event(
                    kind=EventKind.COMPONENT_STARTED,
                    event_time=start_at,
                    source="runtime.twin",
                    subject=instance_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "instance_id": instance_id,
                        "node_id": node_id,
                        "application_id": app.id,
                        "component_id": component.id,
                        "long_running": component.is_long_running,
                        "queue_delay_s": float(queue.estimate),
                        "compute_sharing_policy": "max-min",
                        "cpu_request": fair_cpu,
                        "predicted_solo_duration_s": float(execution.estimate),
                    },
                ),
                start_at,
            )
            return

        for request in component.resources:
            self._schedule_component_event(
                instance_id,
                Event(
                    kind=EventKind.RESOURCE_ALLOCATED,
                    event_time=start_at,
                    source="runtime.twin",
                    subject=node_id,
                    correlation_id=instance.application_instance_id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                    },
                ),
                start_at,
            )
        self._schedule_component_event(
            instance_id,
            Event(
                kind=EventKind.COMPONENT_STARTED,
                event_time=start_at,
                source="runtime.twin",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "node_id": node_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "long_running": component.is_long_running,
                    "queue_delay_s": float(queue.estimate),
                },
            ),
            start_at,
        )
        if component.is_long_running:
            return
        for request in component.resources:
            self._schedule_component_event(
                instance_id,
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=finish_at,
                    source="runtime.twin",
                    subject=node_id,
                    correlation_id=instance.application_instance_id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                    },
                ),
                finish_at,
            )
        self._schedule_component_event(
            instance_id,
            Event(
                kind=EventKind.COMPONENT_COMPLETED,
                event_time=finish_at,
                source="runtime.twin",
                subject=instance_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": instance_id,
                    "node_id": node_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "duration_s": float(execution.estimate),
                    "model_uncertainty": model_uncertainty,
                    "output_bytes": int(output_size.estimate),
                },
            ),
            finish_at,
        )

    async def _activate_fair_compute(self, instance_id: str) -> None:
        assert self.context is not None
        pending = self._pending_compute.get(instance_id)
        if pending is None:
            return
        cpu_request = float(pending["cpu_request"])
        execution = pending["execution"]
        baseline = float(execution.estimate)
        work = (
            float("inf")
            if bool(pending["long_running"])
            else baseline * cpu_request
        )
        self._compute_scheduler.add(
            TwinComputeJob(
                id=instance_id,
                node_id=str(pending["action"].payload["node_id"]),
                max_cpu=cpu_request,
                work_cpu_seconds=work,
                started_at=self.clock.now(),
                payload={"baseline_duration_s": baseline},
            ),
            now=self.clock.now(),
            state=self.context.state(),
        )
        self._schedule_compute_tick()

    def _schedule_compute_tick(self) -> None:
        next_at = self._compute_scheduler.next_event_at(now=self.clock.now())
        if next_at is None:
            return
        self.queue.schedule(
            Event(
                kind=_COMPUTE_TICK_KIND,
                event_time=next_at,
                source="runtime.twin.internal",
                payload={"generation": self._compute_scheduler.generation},
            ),
            next_at,
        )

    async def _handle_compute_tick(self) -> None:
        assert self.context is not None
        completed = self._compute_scheduler.advance(
            now=self.clock.now(),
            state=self.context.state(),
        )
        for job in completed:
            await self._complete_fair_compute(job)
        self._schedule_compute_tick()

    async def _complete_fair_compute(self, job: TwinComputeJob) -> None:
        assert self.context is not None
        pending = self._pending_compute.pop(job.id, None)
        if pending is None:
            return
        action = pending["action"]
        assert isinstance(action, Action)
        state = self.context.state()
        instance = state.components[job.id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        node_id = str(action.payload["node_id"])
        actual_duration = max(0.0, self.clock.now() - job.started_at)
        baseline = float(job.payload["baseline_duration_s"])
        for request in component.resources:
            await self.context.emit(
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=self.clock.now(),
                    source="runtime.twin",
                    subject=node_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                        "scheduling": "fair",
                    },
                )
            )
        output_size = pending["output_size"]
        self._reservations[node_id] = [
            item
            for item in self._reservations.get(node_id, [])
            if item.get("instance_id") != job.id
        ]
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_COMPLETED,
                event_time=self.clock.now(),
                source="runtime.twin",
                subject=job.id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": job.id,
                    "node_id": node_id,
                    "application_id": app.id,
                    "component_id": component.id,
                    "duration_s": actual_duration,
                    "predicted_solo_duration_s": baseline,
                    "compute_sharing_delay_s": max(0.0, actual_duration - baseline),
                    "compute_sharing_policy": "max-min",
                    "model_uncertainty": pending["model_uncertainty"],
                    "output_bytes": int(output_size.estimate),
                },
            )
        )
        await self._retry_waiting_compute()

    async def _fail_compute_on_unavailable_nodes(self) -> None:
        """Fail scheduled/running compute bound to nodes that went offline."""

        assert self.context is not None
        state = self.context.state()
        failed = [
            instance
            for instance in state.components.values()
            if instance.node_id is not None
            and instance.status in {"scheduled", "running"}
            and (
                instance.node_id not in state.nodes
                or state.nodes[instance.node_id].status != "online"
            )
        ]
        for instance in failed:
            node_id = str(instance.node_id)
            self._cancel_component_events(instance.id)
            self._waiting_compute.pop(instance.id, None)
            if instance.id in self._pending_compute:
                self._compute_scheduler.cancel(
                    {instance.id},
                    now=self.clock.now(),
                    state=state,
                )
                self._pending_compute.pop(instance.id, None)
            self._reservations[node_id] = [
                item
                for item in self._reservations.get(node_id, [])
                if item.get("instance_id") != instance.id
            ]
            app = state.applications[instance.application_id]
            component = app.component(instance.component_id)
            if instance.status == "running":
                for request in component.resources:
                    await self.context.emit(
                        Event(
                            kind=EventKind.RESOURCE_RELEASED,
                            event_time=self.clock.now(),
                            source="runtime.twin",
                            subject=node_id,
                            correlation_id=instance.application_instance_id,
                            payload={
                                "node_id": node_id,
                                "resource": request.name,
                                "amount": request.amount,
                                "reason": "node_offline",
                            },
                        )
                    )
            await self.context.emit(
                Event(
                    kind=EventKind.COMPONENT_FAILED,
                    event_time=self.clock.now(),
                    source="runtime.twin",
                    subject=instance.id,
                    correlation_id=instance.application_instance_id,
                    payload={
                        "instance_id": instance.id,
                        "node_id": node_id,
                        "application_id": app.id,
                        "component_id": component.id,
                        "error": "execution node became unavailable",
                        "failure_kind": "node_offline",
                    },
                )
            )
        if failed:
            self._schedule_compute_tick()
            await self._retry_waiting_compute()

    async def _retry_waiting_compute(self) -> None:
        """Retry fair-CPU jobs blocked only by strict secondary resources."""

        for instance_id in tuple(self._waiting_compute):
            pending = self._waiting_compute.get(instance_id)
            if pending is None:
                continue
            await self._schedule_compute(
                pending["action"],
                execution=pending["execution"],
                output_size=pending["output_size"],
                transfer_uncertainties=pending["transfer_uncertainties"],
            )
            if instance_id not in self._waiting_compute:
                # Starting one waiter may consume a strict resource. Recompute
                # every following waiter against the new reservation state.
                continue

    def _schedule_network_tick(self) -> None:
        next_at = self._network_scheduler.next_event_at(now=self.clock.now())
        if next_at is None:
            return
        self.queue.schedule(
            Event(
                kind=_NETWORK_TICK_KIND,
                event_time=next_at,
                source="runtime.twin.internal",
                payload={"generation": self._network_scheduler.generation},
            ),
            next_at,
        )

    async def _handle_network_tick(self) -> None:
        assert self.context is not None
        now = self.clock.now()
        completed = self._network_scheduler.advance(
            now=now,
            state=self.context.state(),
        )
        for transfer in completed:
            await self._complete_transfer(transfer)
        self._schedule_network_tick()

    async def _complete_transfer(self, transfer: TwinTransfer) -> None:
        assert self.context is not None
        placement_id = str(transfer.payload["placement_instance_id"])
        pending = self._pending_network.get(placement_id)
        if pending is None:
            return
        action = pending["action"]
        assert isinstance(action, Action)
        instance = self.context.state().components[placement_id]
        actual_duration = max(0.0, self.clock.now() - transfer.started_at)
        if bool(transfer.payload["emit_events"]):
            payload = dict(transfer.payload["event_payload"])
            payload.update(
                {
                    "duration_s": actual_duration,
                    "predicted_duration_s": transfer.baseline_duration_s,
                    "propagation_delay_s": float(transfer.payload["propagation_s"]),
                    "sharing_delay_s": max(
                        0.0, actual_duration - transfer.baseline_duration_s
                    ),
                    "queue_delay_s": 0.0,
                    "sharing_policy": "max-min",
                }
            )
            await self.context.emit(
                Event(
                    kind=EventKind.DATA_TRANSFER_COMPLETED,
                    event_time=self.clock.now(),
                    source="runtime.twin",
                    subject=transfer.id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload=payload,
                )
            )
        remaining = pending["remaining"]
        assert isinstance(remaining, set)
        remaining.discard(transfer.id)
        if remaining:
            return
        self._pending_network.pop(placement_id, None)
        await self._schedule_compute(
            action,
            execution=pending["execution"],
            output_size=pending["output_size"],
            transfer_uncertainties=pending["transfer_uncertainties"],
        )

    async def _fail_network_placement(self, placement_id: str, *, reason: str) -> None:
        assert self.context is not None
        pending = self._pending_network.pop(placement_id, None)
        if pending is None:
            return
        remaining = pending["remaining"]
        assert isinstance(remaining, set)
        cancelled = self._network_scheduler.cancel(set(remaining))
        action = pending["action"]
        assert isinstance(action, Action)
        instance = self.context.state().components[placement_id]
        app = self.context.state().applications[instance.application_id]
        component = app.component(instance.component_id)
        for transfer in cancelled:
            if not bool(transfer.payload["emit_events"]):
                continue
            payload = dict(transfer.payload["event_payload"])
            payload.update(
                {
                    "duration_s": max(0.0, self.clock.now() - transfer.started_at),
                    "error": reason,
                    "sharing_policy": "max-min",
                }
            )
            await self.context.emit(
                Event(
                    kind=EventKind.DATA_TRANSFER_FAILED,
                    event_time=self.clock.now(),
                    source="runtime.twin",
                    subject=transfer.id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload=payload,
                )
            )
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_FAILED,
                event_time=self.clock.now(),
                source="runtime.twin",
                subject=placement_id,
                correlation_id=instance.application_instance_id,
                causation_id=action.id,
                payload={
                    "instance_id": placement_id,
                    "node_id": str(action.payload["node_id"]),
                    "application_id": app.id,
                    "component_id": component.id,
                    "error": reason,
                },
            )
        )

    async def _fail_blocked_network_transfers(self) -> None:
        assert self.context is not None
        blocked = self._network_scheduler.blocked(self.context.state())
        placements = {
            str(transfer.payload["placement_instance_id"]) for transfer in blocked
        }
        for placement_id in placements:
            await self._fail_network_placement(
                placement_id,
                reason="active Twin transfer path became unavailable",
            )
        if placements and self._network_scheduler.transfers:
            self._network_scheduler.topology_changed(
                now=self.clock.now(),
                state=self.context.state(),
            )


    async def _migrate_long_running(self, action: Action) -> None:
        """Move a running long-lived component while preserving its identity."""

        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        target_node_id = str(action.payload["node_id"])
        instance = state.components[instance_id]
        source_node_id = instance.node_id
        if source_node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        now = self.clock.now()

        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_MIGRATING,
                event_time=now,
                source="runtime.twin",
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

        self._cancel_component_events(instance_id)
        self._waiting_compute.pop(instance_id, None)
        if instance_id in self._pending_compute:
            self._compute_scheduler.cancel(
                {instance_id},
                now=now,
                state=self.context.state(),
            )
            self._pending_compute.pop(instance_id, None)
            self._schedule_compute_tick()
        self._reservations[source_node_id] = [
            item
            for item in self._reservations.get(source_node_id, [])
            if item.get("instance_id") != instance_id
        ]
        for request in component.resources:
            await self.context.emit(
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=now,
                    source="runtime.twin",
                    subject=source_node_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "node_id": source_node_id,
                        "resource": request.name,
                        "amount": request.amount,
                        "reason": "migration",
                    },
                )
            )
        await self._place(action)

    async def _restart_long_running(self, action: Action) -> None:
        """Restart a running service/stream on its current virtual node."""

        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        instance = state.components[instance_id]
        node_id = instance.node_id
        if node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        now = self.clock.now()
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_RESTARTING,
                event_time=now,
                source="runtime.twin",
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
        self._cancel_component_events(instance_id)
        self._waiting_compute.pop(instance_id, None)
        if instance_id in self._pending_compute:
            self._compute_scheduler.cancel(
                {instance_id},
                now=now,
                state=self.context.state(),
            )
            self._pending_compute.pop(instance_id, None)
            self._schedule_compute_tick()
        self._reservations[node_id] = [
            item
            for item in self._reservations.get(node_id, [])
            if item.get("instance_id") != instance_id
        ]
        for request in component.resources:
            await self.context.emit(
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=now,
                    source="runtime.twin",
                    subject=node_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                        "reason": "restart",
                    },
                )
            )
        placement = replace(
            action,
            payload={**dict(action.payload), "node_id": node_id},
        )
        await self._place(placement)

    async def _stop_long_running(self, action: Action) -> None:
        assert self.context is not None
        state = self.context.state()
        instance_id = str(action.payload["instance_id"])
        instance = state.components[instance_id]
        app = state.applications[instance.application_id]
        component = app.component(instance.component_id)
        node_id = instance.node_id
        if node_id is None:
            raise RuntimeError(f"running component has no node: {instance_id}")
        now = self.clock.now()
        self._waiting_compute.pop(instance_id, None)
        if instance_id in self._pending_compute:
            self._compute_scheduler.cancel(
                {instance_id},
                now=now,
                state=self.context.state(),
            )
            self._pending_compute.pop(instance_id, None)
            self._schedule_compute_tick()
        self._reservations[node_id] = [
            item
            for item in self._reservations.get(node_id, [])
            if item.get("instance_id") != instance_id
        ]
        for request in component.resources:
            await self.context.emit(
                Event(
                    kind=EventKind.RESOURCE_RELEASED,
                    event_time=now,
                    source="runtime.twin",
                    subject=node_id,
                    correlation_id=instance.application_instance_id,
                    causation_id=action.id,
                    payload={
                        "node_id": node_id,
                        "resource": request.name,
                        "amount": request.amount,
                    },
                )
            )
        await self.context.emit(
            Event(
                kind=EventKind.COMPONENT_COMPLETED,
                event_time=now,
                source="runtime.twin",
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
                        now - (instance.started_at if instance.started_at is not None else now),
                    ),
                    "model_uncertainty": 0.0,
                    "output_bytes": 0,
                    "stopped": True,
                },
            )
        )

    async def _advance_until_decision(self) -> None:
        await self.advance(stop_on_decision=True)

    async def advance(
        self,
        *,
        until: float | None = None,
        stop_on_decision: bool = False,
    ) -> None:
        if self.context is None:
            raise RuntimeError("Twin backend has not started")
        self._advance_depth += 1
        try:
            while self.queue:
                if stop_on_decision and self.context.state().ready_components():
                    return
                at, event = self.queue.pop()
                if self._consume_component_event(event):
                    continue
                if event.kind == _NETWORK_TICK_KIND:
                    generation = int(event.payload.get("generation", -1))
                    if generation != self._network_scheduler.generation:
                        continue
                    if until is not None and at > until:
                        self.queue.schedule(event, at)
                        break
                    previous = self.clock.now()
                    self.clock.advance_to(at)
                    self.models.advance(previous, at, self.context.state())
                    await self._handle_network_tick()
                    continue
                if event.kind == _COMPUTE_TICK_KIND:
                    generation = int(event.payload.get("generation", -1))
                    if generation != self._compute_scheduler.generation:
                        continue
                    if until is not None and at > until:
                        self.queue.schedule(event, at)
                        break
                    previous = self.clock.now()
                    self.clock.advance_to(at)
                    self.models.advance(previous, at, self.context.state())
                    await self._handle_compute_tick()
                    continue
                if until is not None and at > until:
                    self.queue.schedule(event, at)
                    break
                previous = self.clock.now()
                self.clock.advance_to(at)
                self.models.advance(previous, at, self.context.state())
                await self.context.emit(event)
                if event.kind in {EventKind.NODE_OFFLINE, EventKind.NODE_REMOVED}:
                    await self._fail_compute_on_unavailable_nodes()
                if (
                    event.kind == EventKind.COMPONENT_STARTED
                    and event.source == "runtime.twin"
                    and event.subject in self._pending_compute
                ):
                    await self._activate_fair_compute(str(event.subject))
                if self._compute_scheduler.jobs and event.kind in {
                    EventKind.NODE_OFFLINE,
                    EventKind.NODE_RECOVERED,
                    EventKind.NODE_REMOVED,
                    EventKind.MEASUREMENT_OBSERVED,
                }:
                    self._compute_scheduler.capacity_changed(
                        now=at,
                        state=self.context.state(),
                    )
                    self._schedule_compute_tick()
                if self._network_scheduler.transfers and event.kind in {
                    EventKind.LINK_REGISTERED,
                    EventKind.LINK_CHANGED,
                    EventKind.LINK_REMOVED,
                    EventKind.NODE_OFFLINE,
                    EventKind.NODE_RECOVERED,
                    EventKind.NODE_REMOVED,
                    EventKind.MEASUREMENT_OBSERVED,
                }:
                    self._network_scheduler.topology_changed(
                        now=at,
                        state=self.context.state(),
                    )
                    await self._fail_blocked_network_transfers()
                    self._schedule_network_tick()
            if until is not None and self.clock.now() < until:
                previous = self.clock.now()
                self.clock.advance_to(until)
                self.models.advance(previous, until, self.context.state())
                await self.context.emit(
                    Event(
                        kind=EventKind.TIME_ADVANCED,
                        event_time=until,
                        source="runtime.twin",
                        payload={"from": previous, "to": until},
                    )
                )
        finally:
            self._advance_depth -= 1

    async def close(self) -> None:
        self._pending_network.clear()
        self._component_event_ids.clear()
        self._event_component.clear()
        self._cancelled_event_ids.clear()
        self._network_scheduler.clear()
        self._pending_compute.clear()
        self._waiting_compute.clear()
        self._compute_scheduler.clear()
        self._reservations.clear()
        self.queue.clear()
        # BackendContext owns bound Session callables. Dropping it here breaks
        # Session -> Backend -> Context -> Session cycles deterministically.
        self.context = None
