from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.core.event import EventKind


def test_local_real_runtime_executes_application(small_system, small_app):
    async def run():
        session = Darpan.real()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        await session.apply(Action.place(f"{instance}:a", "edge-1"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{instance}:b",
            timeout=5,
        )
        await session.apply(Action.place(f"{instance}:b", "fog-1"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=5,
        )
        assert all(item.status == "completed" for item in session.state.components.values())
        await session.close()

    asyncio.run(run())
