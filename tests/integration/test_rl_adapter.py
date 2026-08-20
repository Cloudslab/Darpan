from __future__ import annotations

from darpan import Darpan
from darpan.adapters.rl import DarpanEnv


def test_third_party_rl_only_needs_darpan_env(small_system, small_app):
    env = DarpanEnv(
        session_factory=Darpan.twin,
        system=small_system,
        application=small_app,
    )
    observation, info = env.reset()
    assert observation.ndim == 1
    while True:
        valid = [i for i, allowed in enumerate(info["action_mask"]) if allowed]
        observation, reward, terminated, truncated, info = env.step(valid[0])
        if terminated or truncated:
            break
    assert terminated
    env.close()


def test_async_rl_waits_for_feasible_action_after_node_recovery() -> None:
    import asyncio

    from darpan import ApplicationSpec, ComponentSpec, Darpan, ResourceRequest, ResourceSpec
    from darpan.adapters.rl.env import AsyncDarpanEnv
    from darpan.core.event import Event, EventKind
    from darpan.core.topology import NodeSpec, SystemSpec

    async def run() -> None:
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),)
        )
        app = ApplicationSpec(
            "rl-recovery",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                ),
            ),
        )
        env = AsyncDarpanEnv(system=system, application=app, runtime="twin", timeout=2)
        session = Darpan.twin()
        await session.start()
        try:
            await session.register_system(system)
            await session.emit(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=0.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                )
            )
            instance = await session.submit_application(app)
            env.session = session
            env.application_instance_id = instance
            waiter = asyncio.create_task(
                env._advance_to_controlled_decision(after=session.event_count)
            )
            await asyncio.sleep(0)
            assert waiter.done() is False
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_RECOVERED,
                    event_time=1.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                1.0,
            )
            await session.advance(until=1.0)
            decision = await asyncio.wait_for(waiter, timeout=1)
            assert decision is not None
            assert decision.id == f"{instance}:task"
            assert any(env.action_adapter.action_mask(session.state, decision))
        finally:
            env.session = None
            await session.close()

    asyncio.run(run())
