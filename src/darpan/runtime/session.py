"""Central Darpan session: EventLog -> State -> observers -> Actions -> Backend."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from darpan.core.action import Action
from darpan.core.application import ApplicationSpec
from darpan.core.event import Event, EventKind
from darpan.core.protocols.backend import BackendContext, RuntimeBackend
from darpan.core.serialization import to_primitive
from darpan.core.state import ContinuumState
from darpan.core.topology import SystemSpec

from .action_plan import (
    BackendCapabilityValidator,
    BackendNodeValidator,
    DataReachabilityValidator,
    DefaultActionValidator,
    PriorityArbiter,
    ResourceFeasibilityValidator,
    validate_actions,
)
from .clock import Clock, WallClock
from .event_log import EventLog, InMemoryEventLog
from .orchestrator import ApplicationOrchestrator
from .state_store import StateStore


@dataclass(frozen=True, slots=True)
class SessionEvent:
    index: int
    event: Event
    state: ContinuumState


class Session:
    """A single deterministic or physical Continuum execution session."""

    def __init__(
        self,
        backend: RuntimeBackend,
        *,
        clock: Clock | None = None,
        event_log: EventLog | None = None,
        validators: list[Any] | None = None,
        arbiter: Any | None = None,
        initial_state: ContinuumState | None = None,
    ) -> None:
        self.backend = backend
        backend_clock = getattr(backend, "clock", None)
        self.clock = (
            clock
            if clock is not None
            else backend_clock
            if backend_clock is not None
            else WallClock()
        )
        self.event_log = event_log if event_log is not None else InMemoryEventLog()
        self.state_store = StateStore(initial_state)
        self.validators = validators if validators is not None else [
            BackendCapabilityValidator(backend),
            BackendNodeValidator(backend),
            DefaultActionValidator(),
            ResourceFeasibilityValidator(),
            DataReachabilityValidator(),
        ]
        self.arbiter = arbiter if arbiter is not None else PriorityArbiter()
        self.orchestrator = ApplicationOrchestrator()
        self._listeners: list[Callable[[Event, ContinuumState], Any]] = []
        self._condition = asyncio.Condition()
        self._emit_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._event_index = 0
        self._history: list[SessionEvent] = []
        self._service_tasks: set[asyncio.Task[Any]] = set()
        self._services: list[Any] = []

    @property
    def state(self) -> ContinuumState:
        return self.state_store.state

    @property
    def event_count(self) -> int:
        return self._event_index

    @property
    def supported_action_kinds(self) -> frozenset[str] | None:
        """Action kinds explicitly supported by the active backend.

        ``None`` means the backend did not declare a capability set. The
        canonical :class:`ActionKind` vocabulary is intentionally broader than
        any one backend's implementation.
        """

        supported = getattr(self.backend, "supported_action_kinds", None)
        return None if supported is None else frozenset(supported)

    def supports_action(self, kind: str) -> bool:
        supported = self.supported_action_kinds
        return supported is None or kind in supported

    @property
    def history(self) -> tuple[SessionEvent, ...]:
        return tuple(self._history)

    async def start(self) -> Session:
        if self._closed:
            raise RuntimeError("cannot start a closed Session")
        if self._started:
            return self
        context = BackendContext(
            emit=self.emit,
            state=lambda: self.state,
            now=self.clock.now,
        )
        backend_started = False
        started_services: list[Any] = []
        try:
            await self.backend.start(context)
            backend_started = True
            for service in self._services:
                starter = getattr(service, "start", None)
                if starter is None:
                    raise TypeError(f"session service has no start(): {service!r}")
                result = starter()
                if inspect.isawaitable(result):
                    await result
                started_services.append(service)
        except BaseException:
            for service in reversed(started_services):
                closer = getattr(service, "close", None)
                if closer is None:
                    continue
                try:
                    result = closer()
                    if inspect.isawaitable(result):
                        await result
                except BaseException:
                    pass
            if backend_started:
                try:
                    await self.backend.close()
                except BaseException:
                    pass
            raise
        self._started = True
        return self

    def add_service(self, service: Any) -> Any:
        """Register a managed background service before the session starts."""

        if self._started:
            raise RuntimeError("add services before Session.start()")
        self._services.append(service)
        return service

    def subscribe(self, listener: Callable[[Event, ContinuumState], Any]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[Event, ContinuumState], Any]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def emit(self, event: Event) -> None:
        prepare_event = getattr(self.backend, "prepare_event", None)
        if prepare_event is not None:
            prepared = prepare_event(event)
            if inspect.isawaitable(prepared):
                await prepared
        async with self._emit_lock:
            appended = self.event_log.append(event)
            if not appended:
                return
            state = self.state_store.apply(event)
            self._event_index += 1
            self._history.append(SessionEvent(self._event_index, event, state))
            generated = self.orchestrator.generated_events(event, state)
            listeners = tuple(self._listeners)
            async with self._condition:
                self._condition.notify_all()

        for listener in listeners:
            result = listener(event, state)
            if inspect.isawaitable(result):
                await result
        for generated_event in generated:
            # Causal children at the parent's logical time must commit directly.
            # A WallClock timer can wake a few microseconds before its declared
            # timestamp; rescheduling same-time COMPONENT_CREATED/READY children
            # as detached tasks can then reverse their canonical order.
            if generated_event.event_time > max(event.event_time, self.clock.now()):
                await self.schedule_event(generated_event, at=generated_event.event_time)
            else:
                await self.emit(generated_event)

    async def register_system(self, system: SystemSpec) -> None:
        for node in system.nodes:
            await self.emit(
                Event(
                    kind=EventKind.NODE_REGISTERED,
                    event_time=self.clock.now(),
                    source="session",
                    subject=node.id,
                    payload={"node": to_primitive(node)},
                )
            )
        for link in system.links:
            await self.emit(
                Event(
                    kind=EventKind.LINK_REGISTERED,
                    event_time=self.clock.now(),
                    source="session",
                    subject=link.id,
                    payload={"link": to_primitive(link)},
                )
            )

    async def register_application(self, app: ApplicationSpec) -> None:
        await self.emit(
            Event(
                kind=EventKind.APPLICATION_REGISTERED,
                event_time=self.clock.now(),
                source="session",
                subject=app.id,
                payload={"application": to_primitive(app)},
            )
        )

    async def submit_application(
        self, app: ApplicationSpec, *, instance_id: str | None = None
    ) -> str:
        if app.id not in self.state.applications:
            await self.register_application(app)
        instance_id = instance_id or f"{app.id}-{uuid4().hex[:8]}"
        await self.emit(
            Event(
                kind=EventKind.APPLICATION_SUBMITTED,
                event_time=self.clock.now(),
                source="session",
                subject=instance_id,
                correlation_id=instance_id,
                payload={"application_id": app.id, "instance_id": instance_id},
            )
        )
        return instance_id

    async def schedule_event(self, event: Event, *, at: float) -> None:
        """Schedule one canonical event using runtime-appropriate time semantics."""

        now = self.clock.now()
        scheduler = getattr(self.backend, "schedule_event", None)
        if at < now:
            if scheduler is not None:
                raise ValueError("cannot schedule an event in the past")
            at = now
        if event.event_time != at:
            event = Event(
                kind=event.kind,
                event_time=at,
                source=event.source,
                payload=event.payload,
                subject=event.subject,
                id=event.id,
                schema_version=event.schema_version,
                ingest_time=event.ingest_time,
                source_sequence=event.source_sequence,
                correlation_id=event.correlation_id,
                causation_id=event.causation_id,
                metadata=event.metadata,
            )
        if scheduler is not None:
            result = scheduler(event, at)
            if inspect.isawaitable(result):
                await result
            return

        async def emit_later() -> None:
            # Some platform/event-loop combinations may wake a timer slightly
            # early. Do not commit a future event until the declared session
            # time has actually been reached.
            while (remaining := at - self.clock.now()) > 0.0:
                await self.clock.sleep(remaining)
            await self.emit(event)

        task = asyncio.create_task(emit_later())
        self._service_tasks.add(task)
        task.add_done_callback(self._service_tasks.discard)

    async def schedule_application(
        self,
        app: ApplicationSpec,
        *,
        at: float,
        instance_id: str | None = None,
    ) -> str:
        """Schedule an application submission at an absolute session time.

        Twin backends place the event directly on their deterministic event
        queue. Physical backends use the session clock and a tracked service
        task.  The caller does not need to know which runtime it is using.
        """

        now = self.clock.now()
        scheduler = getattr(self.backend, "schedule_event", None)
        if at < now:
            if scheduler is not None:
                raise ValueError("cannot schedule an application in the past")
            # Wall-clock setup itself consumes time. An arrival at t=0 should
            # be submitted immediately rather than rejected because a few
            # microseconds elapsed while registering the workload.
            at = now
        if app.id not in self.state.applications:
            await self.register_application(app)
        instance_id = instance_id or f"{app.id}-{uuid4().hex[:8]}"
        event = Event(
            kind=EventKind.APPLICATION_SUBMITTED,
            event_time=at,
            source="session",
            subject=instance_id,
            correlation_id=instance_id,
            payload={"application_id": app.id, "instance_id": instance_id},
        )
        await self.schedule_event(event, at=at)
        return instance_id

    async def advance(self, *, until: float | None = None) -> None:
        """Advance a runtime that exposes deterministic time control.

        This is primarily used by workload/scenario orchestration.  Physical
        runtimes intentionally do not pretend that they can be fast-forwarded.
        """

        advance = getattr(self.backend, "advance", None)
        if advance is None:
            raise RuntimeError("this runtime does not support explicit time advancement")
        result = advance(until=until, stop_on_decision=False)
        if inspect.isawaitable(result):
            await result

    async def apply(self, action: Action) -> bool:
        results = await self.apply_many([action])
        return bool(results)

    async def apply_many(self, actions: list[Action]) -> list[Action]:
        if not actions:
            return []
        for action in actions:
            await self.emit(
                Event(
                    kind=EventKind.ACTION_REQUESTED,
                    event_time=self.clock.now(),
                    source=action.source,
                    subject=action.target,
                    correlation_id=action.correlation_id,
                    payload={
                        "action_id": action.id,
                        "kind": action.kind,
                        "action_payload": to_primitive(action.payload),
                        "priority": action.priority,
                        "metadata": to_primitive(action.metadata),
                    },
                )
            )
        accepted, validation_results = validate_actions(
            actions, self.validators, self.state
        )
        for result in validation_results:
            await self.emit(
                Event(
                    kind=(
                        EventKind.ACTION_ACCEPTED
                        if result.accepted
                        else EventKind.ACTION_REJECTED
                    ),
                    event_time=self.clock.now(),
                    source="runtime.validator",
                    subject=result.action.target,
                    correlation_id=result.action.correlation_id,
                    payload={
                        "action_id": result.action.id,
                        "kind": result.action.kind,
                        "reason": result.reason,
                    },
                )
            )
        selected = self.arbiter.select(accepted, self.state)
        for action in selected:
            await self.emit(
                Event(
                    kind=EventKind.ACTION_STARTED,
                    event_time=self.clock.now(),
                    source="runtime.session",
                    subject=action.target,
                    correlation_id=action.correlation_id,
                    causation_id=action.id,
                    payload={"action_id": action.id, "kind": action.kind},
                )
            )
            try:
                await self.backend.apply(action)
            except Exception as exc:
                await self.emit(
                    Event(
                        kind=EventKind.ACTION_FAILED,
                        event_time=self.clock.now(),
                        source="runtime.session",
                        subject=action.target,
                        correlation_id=action.correlation_id,
                        causation_id=action.id,
                        payload={
                            "action_id": action.id,
                            "kind": action.kind,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
                raise
            await self.emit(
                Event(
                    kind=EventKind.ACTION_COMPLETED,
                    event_time=self.clock.now(),
                    source="runtime.session",
                    subject=action.target,
                    correlation_id=action.correlation_id,
                    causation_id=action.id,
                    payload={"action_id": action.id, "kind": action.kind},
                )
            )
        return selected

    def events_since(self, index: int) -> tuple[Event, ...]:
        events = tuple(self.event_log)
        return events[index:]

    async def wait_for(
        self,
        predicate: Callable[[Event, ContinuumState], bool],
        *,
        after: int = 0,
        timeout: float | None = None,
    ) -> SessionEvent:
        async def find_existing() -> SessionEvent | None:
            for item in self._history:
                if item.index > after and predicate(item.event, item.state):
                    return item
            return None

        existing = await find_existing()
        if existing is not None:
            return existing

        future: asyncio.Future[SessionEvent] = asyncio.get_running_loop().create_future()

        def listener(event: Event, state: ContinuumState) -> None:
            if self._event_index <= after or future.done():
                return
            if predicate(event, state):
                future.set_result(SessionEvent(self._event_index, event, state))

        self.subscribe(listener)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.unsubscribe(listener)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for service in reversed(self._services):
            closer = getattr(service, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                errors.append(exc)
        for task in tuple(self._service_tasks):
            task.cancel()
        if self._service_tasks:
            results = await asyncio.gather(
                *tuple(self._service_tasks), return_exceptions=True
            )
            self._service_tasks.clear()
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )
        if self._started:
            try:
                await self.backend.close()
            except BaseException as exc:
                errors.append(exc)
        self._started = False
        # Closed sessions never emit again. Remove references to listeners and
        # managed services so controller/monitor objects cannot keep Session
        # cycles alive until process-wide garbage collection.
        self._listeners.clear()
        self._services.clear()
        if errors:
            raise ExceptionGroup("errors while closing Session", errors)
