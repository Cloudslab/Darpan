"""Multi-agent RL adapters over the same canonical Darpan session model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from darpan.core.application import ApplicationSpec
from darpan.core.codec import load_application, load_system
from darpan.core.event import Event, EventKind
from darpan.core.protocols.policy import Policy
from darpan.core.state import ComponentInstanceState, ContinuumState
from darpan.core.topology import SystemSpec
from darpan.runtime.session import Session

from .env import _AsyncBridge, _decision_event_relevant, _default_session_factory
from .problem import RLProblem
from .reward import Transition


@dataclass(frozen=True, slots=True)
class AgentView:
    agent_id: str
    observation: Any
    action_mask: Any | None = None


@dataclass(frozen=True, slots=True)
class MultiAgentStep:
    observations: Mapping[str, Any]
    rewards: Mapping[str, float]
    terminated: Mapping[str, bool]


AgentSelector = Callable[[ContinuumState, ComponentInstanceState], str | None]


class AsyncMultiAgentDarpanEnv:
    """Parallel-ready multi-agent environment.

    Each ready component is assigned to at most one agent by ``selector``.
    Agents may use different :class:`RLProblem` adapters while sharing one
    canonical application execution.  Ready decisions that belong to no agent
    can be delegated to a normal Darpan policy with ``fallback_policy``.

    The API follows the common parallel MARL shape: ``reset`` returns mappings
    of active-agent observations/infos and ``step`` accepts one action for each
    currently active agent.
    """

    def __init__(
        self,
        *,
        system: SystemSpec | str,
        application: ApplicationSpec | str,
        agents: Mapping[str, RLProblem],
        selector: AgentSelector,
        runtime: str = "twin",
        session_factory: Callable[[], Session] | None = None,
        cluster: str | None = None,
        fallback_policy: Policy | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not agents:
            raise ValueError("multi-agent environment requires at least one agent")
        self.system = load_system(system) if isinstance(system, str) else system
        self.application = (
            load_application(application) if isinstance(application, str) else application
        )
        self.agents = dict(agents)
        self.selector = selector
        self.fallback_policy = fallback_policy
        self.session_factory = (
            session_factory
            if session_factory is not None
            else _default_session_factory(runtime, cluster)
        )
        self.timeout = timeout
        self.session: Session | None = None
        self.application_instance_id: str | None = None
        self._decisions: dict[str, ComponentInstanceState] = {}
        self._terminated = False

    @property
    def action_sizes(self) -> dict[str, int]:
        return {
            agent_id: problem.action_count(self.system)
            for agent_id, problem in self.agents.items()
        }

    async def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del seed
        if self.session is not None:
            await self.session.close()
        self.session = self.session_factory()
        await self.session.start()
        await self.session.register_system(self.system)
        self.application_instance_id = await self.session.submit_application(
            self.application
        )
        self._terminated = False
        self._decisions = await self._advance_to_agent_decisions(
            after=self.session.event_count
        )
        self._terminated = self._application_done(self.session.state)
        return self._observations(), self._infos()

    async def step(
        self,
        actions: Mapping[str, Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self.session is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset()")
        active = set(self._decisions)
        supplied = set(actions)
        if supplied != active:
            missing = sorted(active - supplied)
            extra = sorted(supplied - active)
            raise ValueError(
                f"actions must match active agents; missing={missing}, extra={extra}"
            )

        before = self.session.state
        event_index = self.session.event_count
        canonical = {}
        for agent_id, decision in self._decisions.items():
            canonical[agent_id] = self.agents[agent_id].action.decode(
                actions[agent_id],
                before,
                decision,
            )
        selected = await self.session.apply_many(list(canonical.values()))
        selected_ids = {action.id for action in selected}
        invalid = {
            agent_id: action.id not in selected_ids
            for agent_id, action in canonical.items()
        }

        self._decisions = await self._advance_to_agent_decisions(after=event_index)
        self._terminated = self._application_done(self.session.state)
        events = self.session.events_since(event_index)
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id, action in canonical.items():
            transition = Transition(
                before,
                action,
                self.session.state,
                events,
                self._terminated,
            )
            rewards[agent_id] = float(self.agents[agent_id].reward.compute(transition))

        terminated = {agent_id: self._terminated for agent_id in self.agents}
        terminated["__all__"] = self._terminated
        truncated = {agent_id: False for agent_id in self.agents}
        truncated["__all__"] = False
        infos = self._infos(invalid=invalid)
        return self._observations(), rewards, terminated, truncated, infos

    def _ready(self, state: ContinuumState) -> list[ComponentInstanceState]:
        return sorted(
            (
                item
                for item in state.ready_components()
                if item.application_instance_id == self.application_instance_id
            ),
            key=lambda item: item.id,
        )

    def _partition(
        self,
        state: ContinuumState,
    ) -> tuple[dict[str, ComponentInstanceState], list[ComponentInstanceState]]:
        assigned: dict[str, ComponentInstanceState] = {}
        unowned: list[ComponentInstanceState] = []
        for decision in self._ready(state):
            agent_id = self.selector(state, decision)
            if agent_id is None:
                unowned.append(decision)
                continue
            if agent_id not in self.agents:
                raise ValueError(
                    f"selector returned unknown MARL agent {agent_id!r} for {decision.id}"
                )
            if not self.agents[agent_id].is_controlled(state, decision):
                raise ValueError(
                    f"agent {agent_id!r} does not control selected decision {decision.id}"
                )
            if agent_id in assigned:
                raise RuntimeError(
                    f"agent {agent_id!r} has multiple simultaneous decisions: "
                    f"{assigned[agent_id].id}, {decision.id}"
                )
            assigned[agent_id] = decision
        return assigned, unowned

    async def _apply_fallback(
        self,
        state: ContinuumState,
        decisions: list[ComponentInstanceState],
    ) -> None:
        if self.session is None:
            raise RuntimeError("multi-agent environment has no active session")
        if self.fallback_policy is None:
            ids = ", ".join(item.id for item in decisions)
            raise RuntimeError(
                f"ready decision(s) {ids} are not assigned to an agent and "
                "no fallback_policy is configured"
            )
        for decision in decisions:
            current = self.session.state.components.get(decision.id)
            if current is None or current.status != "ready":
                continue
            trigger = Event(
                kind=EventKind.COMPONENT_READY,
                event_time=self.session.state.time,
                source="adapter.marl",
                subject=current.id,
                correlation_id=current.application_instance_id,
                payload={"instance_id": current.id},
            )
            result = self.fallback_policy.decide(self.session.state, trigger)
            if result is None:
                raise RuntimeError(
                    f"fallback_policy returned no action for ready decision {current.id}"
                )
            actions = result if isinstance(result, list) else [result]
            selected = await self.session.apply_many(actions)
            if not selected:
                raise RuntimeError(
                    f"fallback_policy action was rejected for ready decision {current.id}"
                )

    async def _advance_to_agent_decisions(
        self,
        *,
        after: int,
    ) -> dict[str, ComponentInstanceState]:
        if self.session is None:
            raise RuntimeError("multi-agent environment has no active session")
        cursor = after
        while True:
            state = self.session.state
            assigned, unowned = self._partition(state)
            if unowned:
                await self._apply_fallback(state, unowned)
                cursor = self.session.event_count
                continue
            actionable = {
                agent_id: decision
                for agent_id, decision in assigned.items()
                if any(self.agents[agent_id].action.action_mask(state, decision))
            }
            if actionable:
                return actionable
            if self._application_done(self.session.state):
                return {}
            await self.session.wait_for(
                lambda event, state: _decision_event_relevant(
                    event, self.application_instance_id
                ),
                after=cursor,
                timeout=self.timeout,
            )
            cursor = self.session.event_count

    def _application_done(self, state: ContinuumState) -> bool:
        if self.application_instance_id is None:
            return False
        relevant = [
            item
            for item in state.components.values()
            if item.application_instance_id == self.application_instance_id
        ]
        return bool(relevant) and all(
            item.status in {"completed", "failed"} for item in relevant
        )

    def _observations(self) -> dict[str, np.ndarray]:
        if self.session is None:
            return {}
        return {
            agent_id: problem.observation.encode(
                self.session.state,
                self._decisions[agent_id],
            )
            for agent_id, problem in self.agents.items()
            if agent_id in self._decisions
        }

    def _infos(
        self,
        *,
        invalid: Mapping[str, bool] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if self.session is None:
            return {}
        invalid = invalid or {}
        infos: dict[str, dict[str, Any]] = {}
        sizes = self.action_sizes
        for agent_id, problem in self.agents.items():
            decision = self._decisions.get(agent_id)
            mask = (
                problem.action.action_mask(self.session.state, decision)
                if decision is not None
                else [False] * sizes[agent_id]
            )
            infos[agent_id] = {
                "action_mask": np.asarray(mask, dtype=np.bool_),
                "decision_instance_id": None if decision is None else decision.id,
                "application_instance_id": self.application_instance_id,
                "invalid_action": bool(invalid.get(agent_id, False)),
                "time": self.session.state.time,
            }
        return infos

    def episode_uncertainty(self) -> float:
        if self.session is None:
            return 0.0
        values = [
            float(event.payload["model_uncertainty"])
            for event in self.session.event_log
            if event.payload.get("model_uncertainty") is not None
        ]
        return max(values) if values else 0.0

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None


class MultiAgentDarpanEnv:
    """Synchronous bridge for :class:`AsyncMultiAgentDarpanEnv`."""

    def __init__(self, **kwargs) -> None:
        self._bridge = _AsyncBridge()
        self._async = AsyncMultiAgentDarpanEnv(**kwargs)

    @property
    def action_sizes(self) -> dict[str, int]:
        return self._async.action_sizes

    def reset(self, *, seed: int | None = None):
        return self._bridge.run(self._async.reset(seed=seed))

    def step(self, actions: Mapping[str, Any]):
        return self._bridge.run(self._async.step(actions))

    @property
    def episode_uncertainty(self) -> float:
        return self._async.episode_uncertainty()

    def close(self) -> None:
        try:
            self._bridge.run(self._async.close())
        finally:
            self._bridge.close()
