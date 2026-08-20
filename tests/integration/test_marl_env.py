from __future__ import annotations

from darpan import ApplicationSpec, ComponentSpec, FlowSpec
from darpan.adapters.rl import MultiAgentDarpanEnv, RLProblem
from darpan.experiment.baselines import FirstFitPolicy


def _selector(state, decision):
    del state
    return {"left": "agent-left", "right": "agent-right"}.get(
        decision.component_id
    )


def _two_root_app():
    return ApplicationSpec(
        "parallel",
        components=(ComponentSpec("left"), ComponentSpec("right")),
    )


def test_marl_exposes_parallel_ready_decisions(small_system):
    env = MultiAgentDarpanEnv(
        system=small_system,
        application=_two_root_app(),
        agents={
            "agent-left": RLProblem.placement(),
            "agent-right": RLProblem.placement(),
        },
        selector=_selector,
        runtime="twin",
    )
    try:
        observations, infos = env.reset()
        assert set(observations) == {"agent-left", "agent-right"}
        actions = {
            agent_id: next(
                index
                for index, allowed in enumerate(infos[agent_id]["action_mask"])
                if allowed
            )
            for agent_id in observations
        }
        observations, rewards, terminated, truncated, infos = env.step(actions)
        assert observations == {}
        assert terminated["__all__"] is True
        assert truncated["__all__"] is False
        assert set(rewards) == {"agent-left", "agent-right"}
        assert all(info["decision_instance_id"] is None for info in infos.values())
    finally:
        env.close()


def test_marl_can_delegate_unowned_ready_components(small_system):
    app = ApplicationSpec(
        "mixed-control",
        components=(
            ComponentSpec("left"),
            ComponentSpec("background"),
            ComponentSpec("tail"),
        ),
        flows=(FlowSpec("left", "tail"), FlowSpec("background", "tail")),
    )

    def selector(state, decision):
        del state
        return "agent-left" if decision.component_id == "left" else None

    env = MultiAgentDarpanEnv(
        system=small_system,
        application=app,
        agents={"agent-left": RLProblem.placement()},
        selector=selector,
        fallback_policy=FirstFitPolicy(),
        runtime="twin",
    )
    try:
        observations, infos = env.reset()
        assert set(observations) == {"agent-left"}
        action = next(
            index
            for index, allowed in enumerate(infos["agent-left"]["action_mask"])
            if allowed
        )
        _, _, terminated, _, _ = env.step({"agent-left": action})
        assert terminated["__all__"] is True
    finally:
        env.close()


def test_marl_real_runtime_uses_same_contract(small_system):
    env = MultiAgentDarpanEnv(
        system=small_system,
        application=_two_root_app(),
        agents={
            "agent-left": RLProblem.placement(),
            "agent-right": RLProblem.placement(),
        },
        selector=_selector,
        runtime="real",
        timeout=3,
    )
    try:
        observations, infos = env.reset()
        actions = {
            agent_id: next(
                index
                for index, allowed in enumerate(infos[agent_id]["action_mask"])
                if allowed
            )
            for agent_id in observations
        }
        _, _, terminated, _, _ = env.step(actions)
        assert terminated["__all__"] is True
    finally:
        env.close()


def test_async_marl_waits_for_feasible_agent_action_after_node_recovery() -> None:
    import asyncio

    from darpan import Darpan, ResourceRequest, ResourceSpec
    from darpan.adapters.rl.marl import AsyncMultiAgentDarpanEnv
    from darpan.core.event import Event, EventKind
    from darpan.core.topology import NodeSpec, SystemSpec

    async def run() -> None:
        system = SystemSpec(
            nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),)
        )
        app = ApplicationSpec(
            "marl-recovery",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                ),
            ),
        )
        env = AsyncMultiAgentDarpanEnv(
            system=system,
            application=app,
            agents={"agent": RLProblem.placement()},
            selector=lambda state, decision: "agent",
            runtime="twin",
            timeout=2,
        )
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
                env._advance_to_agent_decisions(after=session.event_count)
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
            decisions = await asyncio.wait_for(waiter, timeout=1)
            assert set(decisions) == {"agent"}
            assert decisions["agent"].id == f"{instance}:task"
            assert any(
                env.agents["agent"].action.action_mask(
                    session.state, decisions["agent"]
                )
            )
        finally:
            env.session = None
            await session.close()

    asyncio.run(run())
