from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.core.event import EventKind


def test_twin_executes_same_canonical_lifecycle(small_system, small_app):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        assert [item.component_id for item in session.state.ready_components()] == ["a"]
        await session.apply(Action.place(f"{instance}:a", "edge-1"))
        assert [item.component_id for item in session.state.ready_components()] == ["b"]
        await session.apply(Action.place(f"{instance}:b", "cloud-1"))
        assert all(item.status == "completed" for item in session.state.components.values())
        kinds = [event.kind for event in session.event_log]
        assert EventKind.APPLICATION_COMPLETED in kinds
        assert EventKind.COMPONENT_STARTED in kinds
        assert EventKind.COMPONENT_COMPLETED in kinds
        await session.close()

    asyncio.run(run())
