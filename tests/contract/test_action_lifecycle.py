from __future__ import annotations

import asyncio

import pytest

from darpan import Action, Darpan
from darpan.core.action import ActionKind
from darpan.core.event import EventKind
from darpan.runtime.clock import VirtualClock
from darpan.runtime.session import Session


def test_accepted_action_has_complete_canonical_lifecycle(small_system, small_app):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        action = Action.place(f"{instance}:a", "edge-1")

        assert await session.apply(action)
        lifecycle = [
            event.kind
            for event in session.event_log
            if event.payload.get("action_id") == action.id
        ]

        assert lifecycle == [
            EventKind.ACTION_REQUESTED,
            EventKind.ACTION_ACCEPTED,
            EventKind.ACTION_STARTED,
            EventKind.ACTION_COMPLETED,
        ]
        await session.close()

    asyncio.run(run())


def test_rejected_action_never_starts(small_system, small_app):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        await session.submit_application(small_app)
        action = Action(kind=ActionKind.ROUTE, source="test")

        assert not await session.apply(action)
        lifecycle = [
            event.kind
            for event in session.event_log
            if event.payload.get("action_id") == action.id
        ]
        assert lifecycle == [EventKind.ACTION_REQUESTED, EventKind.ACTION_REJECTED]
        await session.close()

    asyncio.run(run())


class FailingBackend:
    supported_action_kinds = frozenset({"test.custom"})

    async def start(self, context):
        self.context = context

    async def apply(self, action):
        raise RuntimeError("backend exploded")

    async def close(self):
        return None


def test_backend_exception_emits_action_failed_before_propagating():
    async def run():
        clock = VirtualClock()
        session = Session(FailingBackend(), clock=clock)
        await session.start()
        action = Action(kind="test.custom", source="test")

        with pytest.raises(RuntimeError, match="backend exploded"):
            await session.apply(action)

        lifecycle = [
            event.kind
            for event in session.event_log
            if event.payload.get("action_id") == action.id
        ]
        assert lifecycle == [
            EventKind.ACTION_REQUESTED,
            EventKind.ACTION_ACCEPTED,
            EventKind.ACTION_STARTED,
            EventKind.ACTION_FAILED,
        ]
        await session.close()

    asyncio.run(run())
