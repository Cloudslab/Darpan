from __future__ import annotations

import asyncio

from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.event import Event, EventKind
from darpan.core.protocols.backend import BackendContext
from darpan.runtime.clock import VirtualClock
from darpan.runtime.session import Session


class PassiveBackend:
    supported_action_kinds = frozenset()

    async def start(self, context: BackendContext) -> None:
        self.context = context

    async def apply(self, action) -> None:
        raise AssertionError(f"unexpected action: {action}")

    async def close(self) -> None:
        return None


class EarlyWakeClock:
    def __init__(self) -> None:
        self.time = 0.0
        self.early_wake = True

    def now(self) -> float:
        return self.time

    async def sleep(self, seconds: float) -> None:
        if self.early_wake:
            self.time += max(0.0, seconds - 0.001)
            self.early_wake = False
        else:
            self.time += max(0.0, seconds)


def test_same_time_causal_children_are_not_detached_when_clock_is_early():
    async def run():
        clock = VirtualClock(0.999)
        session = Session(PassiveBackend(), clock=clock)
        await session.start()
        app = ApplicationSpec("job", (ComponentSpec("task", work_units=0.1),))
        await session.register_application(app)

        await session.emit(
            Event(
                kind=EventKind.APPLICATION_SUBMITTED,
                event_time=1.0,
                source="test",
                subject="job-1",
                correlation_id="job-1",
                payload={"application_id": "job", "instance_id": "job-1"},
            )
        )

        lifecycle = [
            event.kind
            for event in session.event_log
            if event.subject == "job-1:task"
        ]
        assert lifecycle == [EventKind.COMPONENT_CREATED, EventKind.COMPONENT_READY]
        assert session.state.components["job-1:task"].status == "ready"
        await session.close()

    asyncio.run(run())


def test_scheduled_event_waits_again_after_an_early_clock_wake():
    async def run():
        clock = EarlyWakeClock()
        session = Session(PassiveBackend(), clock=clock)
        await session.start()
        app = ApplicationSpec("job", (ComponentSpec("task", work_units=0.1),))
        await session.register_application(app)
        instance_id = await session.schedule_application(app, at=1.0, instance_id="job-1")
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_SUBMITTED
            and event.subject == instance_id,
            timeout=1.0,
        )
        assert clock.now() >= 1.0
        await session.close()

    asyncio.run(run())
