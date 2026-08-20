from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.core.event import EventKind


def test_real_place_returns_only_after_component_is_scheduled(small_system, small_app):
    async def run():
        session = Darpan.real()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        component_id = f"{instance}:a"

        assert session.state.components[component_id].status == "ready"
        assert await session.apply(Action.place(component_id, "edge-1"))
        assert session.state.components[component_id].status != "ready"
        assert any(
            event.kind == EventKind.COMPONENT_SCHEDULED
            and event.subject == component_id
            for event in session.event_log
        )
        await session.close()

    asyncio.run(run())
