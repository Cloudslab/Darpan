from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.core.action import ActionKind
from darpan.core.event import EventKind


def test_stable_action_vocabulary_is_deliberately_small():
    exported = {
        value
        for name, value in vars(ActionKind).items()
        if name.isupper() and isinstance(value, str)
    }
    assert exported == {
        ActionKind.PLACE,
        ActionKind.STOP,
        ActionKind.RESTART,
        ActionKind.MIGRATE,
        ActionKind.SCALE,
        ActionKind.ROUTE,
    }


def test_unsupported_action_is_rejected_before_backend_dispatch(small_system):
    async def run():
        session = Darpan.real()
        await session.start()
        await session.register_system(small_system)
        action = Action(
            kind=ActionKind.ROUTE,
            source="test",
            target="edge-fog",
            payload={"path": ["edge-1", "fog-1"]},
        )
        accepted = await session.apply(action)
        assert accepted is False
        rejected = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("action_id") == action.id
        ]
        assert len(rejected) == 1
        assert "does not support action kind" in rejected[0].payload["reason"]
        await session.close()

    asyncio.run(run())


def test_session_exposes_backend_action_capabilities():
    from darpan import Darpan
    from darpan.core.action import ActionKind

    real = Darpan.real()
    twin = Darpan.twin()
    base = frozenset(
        {
            ActionKind.PLACE,
            ActionKind.MIGRATE,
            ActionKind.RESTART,
            ActionKind.SCALE,
            ActionKind.STOP,
        }
    )
    assert real.supported_action_kinds == base
    assert twin.supported_action_kinds == base | {ActionKind.ROUTE}
    assert not real.supports_action(ActionKind.ROUTE)
    assert twin.supports_action(ActionKind.ROUTE)


def test_rl_env_can_use_real_network_driver(small_system, small_app):
    from darpan.adapters.rl.env import AsyncDarpanEnv

    class Driver:
        async def bind_route(self, binding, state):
            pass

        async def clear_route(self, application, source, target, state):
            pass

    env = AsyncDarpanEnv(
        system=small_system,
        application=small_app,
        runtime="real",
        network_driver=Driver(),
    )
    session = env.session_factory()
    assert session.supports_action(ActionKind.ROUTE)
