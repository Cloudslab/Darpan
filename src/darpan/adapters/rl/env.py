"""Gymnasium-shaped DarpanEnv without requiring Gymnasium as a core dependency."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from threading import Thread
from typing import Any

import numpy as np

from darpan.core.application import ApplicationSpec
from darpan.core.codec import load_application, load_system
from darpan.core.event import EventKind
from darpan.core.protocols.metric import Metric
from darpan.core.state import ComponentInstanceState, ContinuumState
from darpan.core.topology import SystemSpec
from darpan.runtime.session import Session

from .action import PlacementActionAdapter, RLActionAdapter
from .observation import ObservationAdapter, PlacementObservation
from .problem import RLProblem
from .reward import CompletionTimeReward, RewardFunction, Transition

_DECISION_REEVALUATION_KINDS = frozenset(
    {
        EventKind.NODE_REGISTERED,
        EventKind.NODE_RECOVERED,
        EventKind.LINK_REGISTERED,
        EventKind.LINK_CHANGED,
        EventKind.RESOURCE_RELEASED,
        EventKind.MEASUREMENT_OBSERVED,
    }
)


def _decision_event_relevant(event, application_instance_id: str | None) -> bool:
    if event.kind in _DECISION_REEVALUATION_KINDS:
        return True
    if event.kind not in {EventKind.COMPONENT_READY, EventKind.APPLICATION_COMPLETED}:
        return False
    return (
        event.correlation_id == application_instance_id
        or event.payload.get("instance_id") == application_instance_id
    )


def _default_session_factory(
    runtime: str,
    cluster: str | None = None,
    system: SystemSpec | None = None,
    network_driver=None,
    physical_control: bool = False,
) -> Callable[[], Session]:
    from darpan.api import Darpan

    if runtime == "twin":
        if cluster is not None:
            raise ValueError("cluster is only valid with runtime='real'")
        if network_driver is not None:
            raise ValueError("network_driver is only valid with runtime='real'")
        if physical_control:
            raise ValueError("physical_control is only valid with runtime='real'")
        return Darpan.twin
    if runtime != "real":
        raise ValueError("runtime must be 'real' or 'twin'")
    if cluster is None:
        if physical_control:
            raise ValueError("physical_control requires a real cluster inventory")
        if network_driver is None:
            return Darpan.real

        from darpan.runtime.real.backend import RealBackend

        return lambda: Darpan.real(backend=RealBackend(network_driver=network_driver))

    from darpan.runtime.real.cluster.inventory import ClusterInventory
    from darpan.runtime.real.cluster.session import session_from_inventory

    inventory = ClusterInventory.load(cluster)

    def cluster_session() -> Session:
        return session_from_inventory(
            inventory,
            system=system,
            network_driver=network_driver,
            physical_control=physical_control,
        )

    return cluster_session


class AsyncDarpanEnv:
    """Research-facing RL environment over any Darpan Session factory."""

    def __init__(
        self,
        *,
        system: SystemSpec | str,
        application: ApplicationSpec | str,
        runtime: str = "twin",
        session_factory: Callable[[], Session] | None = None,
        cluster: str | None = None,
        network_driver=None,
        physical_control: bool = False,
        observation: ObservationAdapter | None = None,
        action: RLActionAdapter | None = None,
        reward: RewardFunction | None = None,
        problem: RLProblem | None = None,
        timeout: float = 120.0,
        listeners: tuple[Callable[[Any, ContinuumState], Any], ...] = (),
        metrics: tuple[Metric, ...] = (),
    ) -> None:
        self.system = load_system(system) if isinstance(system, str) else system
        self.application = (
            load_application(application) if isinstance(application, str) else application
        )
        self.session_factory = (
            session_factory
            if session_factory is not None
            else _default_session_factory(
                runtime, cluster, self.system, network_driver, physical_control
            )
        )
        if problem is not None and any(
            item is not None for item in (observation, action, reward)
        ):
            raise ValueError(
                "problem cannot be combined with observation/action/reward adapters"
            )
        self.problem = problem if problem is not None else RLProblem.placement(
            observation=(
                observation if observation is not None else PlacementObservation()
            ),
            action=action if action is not None else PlacementActionAdapter(),
            reward=reward if reward is not None else CompletionTimeReward(),
        )
        self.observation = self.problem.observation
        self.action_adapter = self.problem.action
        self.reward = self.problem.reward
        self.timeout = timeout
        self.listeners = tuple(listeners)
        self.metrics = tuple(metrics)
        self.session: Session | None = None
        self.application_instance_id: str | None = None
        self._decision: ComponentInstanceState | None = None
        self._terminated = False

    @property
    def action_size(self) -> int:
        return self.problem.action_count(self.system)

    async def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        if self.session is not None:
            await self.session.close()
        self.session = self.session_factory()
        for listener in self.listeners:
            self.session.subscribe(listener)
        for metric in self.metrics:
            metric.reset()
            self.session.subscribe(metric.observe)
        await self.session.start()
        await self.session.register_system(self.system)
        self.application_instance_id = await self.session.submit_application(
            self.application
        )
        self._terminated = False
        self._decision = await self._advance_to_controlled_decision(
            after=self.session.event_count
        )
        self._terminated = self._application_done(self.session.state)
        observation = self.observation.encode(self.session.state, self._decision)
        return observation, self._info()

    async def step(
        self, action_value: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.session is None or self._decision is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset()")

        before = self.session.state
        event_index = self.session.event_count
        canonical_action = self.action_adapter.decode(
            action_value, before, self._decision
        )
        accepted = await self.session.apply(canonical_action)
        if not accepted:
            transition = Transition(
                before,
                canonical_action,
                self.session.state,
                self.session.events_since(event_index),
                False,
            )
            reward = self.reward.compute(transition)
            return (
                self.observation.encode(self.session.state, self._decision),
                reward,
                False,
                False,
                self._info(invalid=True),
            )

        self._decision = await self._advance_to_controlled_decision(after=event_index)
        self._terminated = self._application_done(self.session.state)
        events = self.session.events_since(event_index)
        transition = Transition(
            before,
            canonical_action,
            self.session.state,
            events,
            self._terminated,
        )
        reward = float(self.reward.compute(transition))
        observation = self.observation.encode(self.session.state, self._decision)
        return observation, reward, self._terminated, False, self._info()

    def _ready(self, state: ContinuumState) -> list[ComponentInstanceState]:
        return sorted(
            (
                item
                for item in state.ready_components()
                if item.application_instance_id == self.application_instance_id
            ),
            key=lambda item: item.id,
        )

    def _next_ready(self, state: ContinuumState) -> ComponentInstanceState | None:
        ready = self._ready(state)
        return ready[0] if ready else None

    async def _advance_to_controlled_decision(
        self,
        *,
        after: int,
    ) -> ComponentInstanceState | None:
        if self.session is None:
            raise RuntimeError("RL environment has no active session")

        cursor = after
        while True:
            state = self.session.state
            ready = self._ready(state)
            controlled = [
                item for item in ready if self.problem.is_controlled(state, item)
            ]
            actionable = [
                item
                for item in controlled
                if any(self.action_adapter.action_mask(state, item))
            ]
            if actionable:
                return actionable[0]

            uncontrolled = [item for item in ready if item not in controlled]
            if uncontrolled:
                if self.problem.fallback_policy is None:
                    ids = ", ".join(item.id for item in uncontrolled)
                    raise RuntimeError(
                        "RLProblem does not control ready decision(s) "
                        f"{ids} and has no fallback_policy"
                    )
                progressed = False
                for item in uncontrolled:
                    state = self.session.state
                    current = state.components.get(item.id)
                    if current is None or current.status != "ready":
                        continue
                    actions = self.problem.fallback_actions(state, current)
                    if actions is None:
                        raise RuntimeError(
                            "fallback_policy returned no action for ready decision "
                            f"{current.id}"
                        )
                    if not isinstance(actions, list):
                        actions = [actions]
                    accepted = await self.session.apply_many(actions)
                    if not accepted:
                        raise RuntimeError(
                            "fallback_policy action was rejected for ready decision "
                            f"{current.id}"
                        )
                    progressed = True
                if progressed:
                    cursor = self.session.event_count
                    continue

            if self._application_done(self.session.state):
                return None

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

    def _info(self, *, invalid: bool = False) -> dict[str, Any]:
        if self.session is None:
            return {}
        mask = (
            self.action_adapter.action_mask(self.session.state, self._decision)
            if self._decision is not None
            else [False] * self.action_size
        )
        return {
            "action_mask": np.asarray(mask, dtype=np.bool_),
            "decision_instance_id": None if self._decision is None else self._decision.id,
            "application_instance_id": self.application_instance_id,
            "invalid_action": invalid,
            "time": self.session.state.time,
            "metrics": {metric.name: metric.result() for metric in self.metrics},
        }

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


class _AsyncBridge:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run, daemon=True, name="DarpanEnvLoop")
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine):
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


class DarpanEnv:
    """Synchronous RL API: reset()/step() compatible with common RL workflows."""

    def __init__(self, **kwargs) -> None:
        self._bridge = _AsyncBridge()
        self._async = AsyncDarpanEnv(**kwargs)

    @property
    def action_size(self) -> int:
        return self._async.action_size

    def reset(self, *, seed: int | None = None):
        return self._bridge.run(self._async.reset(seed=seed))

    def step(self, action):
        return self._bridge.run(self._async.step(action))

    @property
    def episode_uncertainty(self) -> float:
        return self._async.episode_uncertainty()

    def close(self) -> None:
        try:
            self._bridge.run(self._async.close())
        finally:
            self._bridge.close()
