"""Backend-independent application lifecycle orchestration."""

from __future__ import annotations

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState

from .replicas import (
    logical_instance_id,
    next_replica_indices,
    replica_instance_id,
)


class ApplicationOrchestrator:
    def generated_events(self, event: Event, state: ContinuumState) -> list[Event]:
        if event.kind == EventKind.APPLICATION_SUBMITTED:
            return self._create_application(event, state)
        if event.kind == EventKind.COMPONENT_STARTED:
            return [
                *self._after_component_started(event, state),
                *self._scale_convergence(event, state),
            ]
        if event.kind in {EventKind.COMPONENT_COMPLETED, EventKind.COMPONENT_FAILED}:
            return [
                *self._after_component(event, state),
                *self._scale_convergence(event, state),
            ]
        return []

    def _scale_convergence(
        self,
        event: Event,
        state: ContinuumState,
    ) -> list[Event]:
        instance_id = str(event.payload.get("instance_id", ""))
        if instance_id not in state.components or not state.scale_pending(instance_id):
            return []
        instance = state.components[instance_id]
        replicas = state.component_replicas(instance_id, include_terminal=False)
        desired = state.desired_replicas(instance_id)
        if len(replicas) != desired or any(item.status != "running" for item in replicas):
            return []
        logical_id = logical_instance_id(
            instance.application_instance_id,
            instance.component_id,
        )
        return [
            Event(
                kind=EventKind.COMPONENT_SCALED,
                event_time=event.event_time,
                source="runtime.orchestrator",
                subject=logical_id,
                correlation_id=instance.application_instance_id,
                causation_id=event.id,
                payload={
                    "instance_id": logical_id,
                    "application_id": instance.application_id,
                    "component_id": instance.component_id,
                    "desired_replicas": desired,
                    "running_replicas": desired,
                },
            )
        ]

    def _create_application(self, event: Event, state: ContinuumState) -> list[Event]:
        app_id = str(event.payload["application_id"])
        app_instance_id = str(event.payload["instance_id"])
        app = state.applications[app_id]
        generated: list[Event] = []
        for component in app.components:
            instance_id = logical_instance_id(app_instance_id, component.id)
            generated.append(
                Event(
                    kind=EventKind.COMPONENT_CREATED,
                    event_time=event.event_time,
                    source="runtime.orchestrator",
                    subject=instance_id,
                    correlation_id=event.correlation_id or app_instance_id,
                    causation_id=event.id,
                    payload={
                        "instance_id": instance_id,
                        "application_id": app_id,
                        "application_instance_id": app_instance_id,
                        "component_id": component.id,
                        "replica_index": 0,
                    },
                )
            )
        for component_id in app.roots():
            generated.append(
                Event(
                    kind=EventKind.COMPONENT_READY,
                    event_time=event.event_time,
                    source="runtime.orchestrator",
                    subject=logical_instance_id(app_instance_id, component_id),
                    correlation_id=event.correlation_id or app_instance_id,
                    causation_id=event.id,
                    payload={
                        "instance_id": logical_instance_id(app_instance_id, component_id)
                    },
                )
            )
        return generated

    def _successor_ready(
        self,
        app,
        app_instance: str,
        successor: str,
        state: ContinuumState,
    ) -> bool:
        for predecessor_id in app.predecessors(successor):
            primary_id = logical_instance_id(app_instance, predecessor_id)
            replicas = state.component_replicas(primary_id)
            flow = app.flow(predecessor_id, successor)
            if flow is not None and flow.kind == "stream":
                if not any(
                    item.status in {"running", "completed"} for item in replicas
                ):
                    return False
            elif not any(item.status == "completed" for item in replicas):
                return False
        return True

    def _ready_successors(
        self,
        completed,
        app,
        state: ContinuumState,
        *,
        causation_id: str,
        event_time: float,
    ) -> list[Event]:
        generated: list[Event] = []
        app_instance = completed.application_instance_id
        for successor in app.successors(completed.component_id):
            successor_id = logical_instance_id(app_instance, successor)
            successor_state = state.components.get(successor_id)
            if successor_state is None or successor_state.status != "created":
                continue
            if self._successor_ready(app, app_instance, successor, state):
                generated.append(
                    Event(
                        kind=EventKind.COMPONENT_READY,
                        event_time=event_time,
                        source="runtime.orchestrator",
                        subject=successor_id,
                        correlation_id=app_instance,
                        causation_id=causation_id,
                        payload={"instance_id": successor_id},
                    )
                )
        return generated

    def _after_component_started(
        self,
        event: Event,
        state: ContinuumState,
    ) -> list[Event]:
        started = state.components[str(event.payload["instance_id"])]
        app = state.applications[started.application_id]
        return self._ready_successors(
            started,
            app,
            state,
            causation_id=event.id,
            event_time=event.event_time,
        )

    def _after_component(self, event: Event, state: ContinuumState) -> list[Event]:
        completed = state.components[str(event.payload["instance_id"])]
        app = state.applications[completed.application_id]
        app_instance = completed.application_instance_id
        generated: list[Event] = []

        if event.kind == EventKind.COMPONENT_FAILED:
            component_spec = app.component(completed.component_id)
            siblings = tuple(
                item
                for item in state.component_replicas(completed.id)
                if item.id != completed.id
                and item.status not in {"completed", "failed"}
            )
            scale_key = state.component_scale_key(completed.id)
            if (
                component_spec.is_long_running
                and scale_key in state.component_scale_targets
            ):
                active = state.component_replicas(
                    completed.id, include_terminal=False
                )
                desired = state.desired_replicas(completed.id)
                missing = max(0, desired - len(active))
                if missing:
                    logical_id = logical_instance_id(
                        completed.application_instance_id,
                        completed.component_id,
                    )
                    replacement_events = [
                        Event(
                            kind=EventKind.COMPONENT_SCALING,
                            event_time=event.event_time,
                            source="runtime.orchestrator",
                            subject=logical_id,
                            correlation_id=app_instance,
                            causation_id=event.id,
                            payload={
                                "instance_id": logical_id,
                                "application_id": completed.application_id,
                                "component_id": completed.component_id,
                                "from_replicas": len(active),
                                "desired_replicas": desired,
                                "reason": "replica_replacement",
                            },
                        )
                    ]
                    for replica_index in next_replica_indices(
                        state, completed.id, missing
                    ):
                        replica_id = replica_instance_id(
                            completed.application_instance_id,
                            completed.component_id,
                            replica_index,
                        )
                        replacement_events.extend(
                            [
                                Event(
                                    kind=EventKind.COMPONENT_CREATED,
                                    event_time=event.event_time,
                                    source="runtime.orchestrator",
                                    subject=replica_id,
                                    correlation_id=app_instance,
                                    causation_id=event.id,
                                    payload={
                                        "instance_id": replica_id,
                                        "application_id": completed.application_id,
                                        "application_instance_id": app_instance,
                                        "component_id": completed.component_id,
                                        "replica_index": replica_index,
                                        "scaled_from": logical_id,
                                        "replacement_for": completed.id,
                                    },
                                ),
                                Event(
                                    kind=EventKind.COMPONENT_READY,
                                    event_time=event.event_time,
                                    source="runtime.orchestrator",
                                    subject=replica_id,
                                    correlation_id=app_instance,
                                    causation_id=event.id,
                                    payload={
                                        "instance_id": replica_id,
                                        "replica_index": replica_index,
                                        "scaled": True,
                                        "replacement": True,
                                    },
                                ),
                            ]
                        )
                    return replacement_events
                if siblings:
                    return []
            elif component_spec.is_long_running and siblings:
                # One unmanaged replica failing is evidence, but the logical
                # service is still alive while a sibling remains active.
                return []
            failure_kind = str(event.payload.get("failure_kind", "execution_error"))
            retry = component_spec.retry
            if retry.allows(failure_kind) and completed.attempt < retry.max_retries:
                retry_event = Event(
                    kind=EventKind.COMPONENT_RETRYING,
                    event_time=event.event_time,
                    source="runtime.orchestrator",
                    subject=completed.id,
                    correlation_id=app_instance,
                    causation_id=event.id,
                    payload={
                        "instance_id": completed.id,
                        "attempt": completed.attempt + 1,
                        "failure_kind": failure_kind,
                        "previous_node_id": completed.node_id,
                        "reason": str(
                            event.payload.get("error", event.payload.get("reason", ""))
                        ),
                    },
                )
                ready_event = Event(
                    kind=EventKind.COMPONENT_READY,
                    event_time=event.event_time + retry.backoff_s,
                    source="runtime.orchestrator",
                    subject=completed.id,
                    correlation_id=app_instance,
                    causation_id=retry_event.id,
                    payload={
                        "instance_id": completed.id,
                        "attempt": completed.attempt + 1,
                        "retry": True,
                        "backoff_s": retry.backoff_s,
                    },
                )
                return [retry_event, ready_event]

            descendants: set[str] = set()
            frontier = list(app.successors(completed.component_id))
            while frontier:
                component_id = frontier.pop()
                if component_id in descendants:
                    continue
                descendants.add(component_id)
                frontier.extend(app.successors(component_id))
            for component_id in sorted(descendants):
                descendant_id = logical_instance_id(app_instance, component_id)
                descendant = state.components.get(descendant_id)
                if descendant is not None and descendant.status in {"created", "ready"}:
                    generated.append(
                        Event(
                            kind=EventKind.COMPONENT_FAILED,
                            event_time=event.event_time,
                            source="runtime.orchestrator",
                            subject=descendant_id,
                            correlation_id=app_instance,
                            causation_id=event.id,
                            payload={
                                "instance_id": descendant_id,
                                "reason": "dependency_failed",
                            },
                        )
                    )

        if event.kind == EventKind.COMPONENT_COMPLETED:
            generated.extend(
                self._ready_successors(
                    completed,
                    app,
                    state,
                    causation_id=event.id,
                    event_time=event.event_time,
                )
            )

        terminal = {"completed", "failed"}
        logical_groups = [
            state.component_replicas(logical_instance_id(app_instance, component.id))
            for component in app.components
        ]
        if all(
            group and all(item.status in terminal for item in group)
            for group in logical_groups
        ):
            successful = True
            for component, group in zip(
                app.components, logical_groups, strict=True
            ):
                logical_id = logical_instance_id(app_instance, component.id)
                managed = logical_id in state.component_scale_targets
                if component.is_long_running and managed:
                    desired = state.desired_replicas(logical_id)
                    completed_count = sum(
                        item.status == "completed" for item in group
                    )
                    successful = successful and completed_count >= desired
                else:
                    successful = successful and all(
                        item.status == "completed" for item in group
                    )
            generated.append(
                Event(
                    kind=EventKind.APPLICATION_COMPLETED,
                    event_time=event.event_time,
                    source="runtime.orchestrator",
                    subject=app_instance,
                    correlation_id=app_instance,
                    causation_id=event.id,
                    payload={
                        "instance_id": app_instance,
                        "application_id": app.id,
                        "success": successful,
                    },
                )
            )
        return generated
