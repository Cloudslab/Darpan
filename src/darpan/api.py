"""High-level Darpan facade kept intentionally small."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from random import Random

from .core.application import ApplicationSpec
from .core.protocols.simulation import SimulationKernel
from .core.topology import SystemSpec
from .runtime.clock import VirtualClock, WallClock
from .runtime.real.backend import RealBackend
from .runtime.session import Session
from .twin.backend import TwinBackend
from .twin.kernel import DiscreteEventQueue
from .twin.mirror import TwinMirror
from .twin.models.artifact import ArtifactSizeModel
from .twin.models.execution import ExecutionTimeModel
from .twin.models.network import NetworkDelayModel
from .twin.models.queue import QueueDelayModel
from .twin.models.registry import ModelRegistry
from .twin.result import SimulationResult
from .twin.scenario import Scenario
from .twin.snapshot import TwinSnapshot


@dataclass(slots=True)
class Darpan:
    """Factory facade for common physical and digital Darpan sessions."""

    @staticmethod
    def real(*, backend: RealBackend | None = None) -> Session:
        return Session(backend if backend is not None else RealBackend(), clock=WallClock())

    @staticmethod
    def twin(
        *,
        models: ModelRegistry | None = None,
        kernel: SimulationKernel | None = None,
    ) -> Session:
        clock = VirtualClock()
        backend = TwinBackend(clock=clock, models=models, kernel=kernel)
        return Session(backend, clock=clock)


class DigitalTwin:
    """Continuously calibrated Twin attached to a physical Darpan session."""

    def __init__(
        self,
        models: ModelRegistry | None = None,
        *,
        seed: int = 0,
        kernel_factory: Callable[[], SimulationKernel] = DiscreteEventQueue,
    ) -> None:
        self.models = models if models is not None else ModelRegistry(
            [
                ExecutionTimeModel(),
                ArtifactSizeModel(),
                NetworkDelayModel(),
                QueueDelayModel(),
            ]
        )
        self.mirror = TwinMirror(self.models)
        self.random = Random(seed)
        self.kernel_factory = kernel_factory
        self._attached_sessions: set[int] = set()

    def attach(self, real_session: Session, *, replay_history: bool = True) -> DigitalTwin:
        identity = id(real_session)
        if identity in self._attached_sessions:
            return self
        if replay_history:
            for item in real_session.history:
                self.mirror.observe(item.event, item.state)
        real_session.subscribe(self.mirror.observe)
        self._attached_sessions.add(identity)
        return self

    def snapshot(self, state=None, *, at: float | None = None) -> TwinSnapshot:
        state = state if state is not None else self.mirror.last_state
        virtual_time = state.time if at is None else at
        return TwinSnapshot(
            state=deepcopy(state),
            virtual_time=virtual_time,
            model_states=deepcopy(self.models.snapshot()),
            rng_state=self.random.getstate(),
            model_versions={name: "1" for name in self.models.names()},
        )

    def scenario(self, state=None, *, seed: int = 0) -> Scenario:
        return Scenario(self.snapshot(state), seed=seed)

    async def simulate(
        self,
        scenario: Scenario,
        *,
        actions=(),
        until: float | None = None,
    ) -> SimulationResult:
        session = self.session(scenario)
        await session.start()
        try:
            for action in actions:
                await session.apply(action)
            backend = session.backend
            if hasattr(backend, "advance"):
                horizon = until
                if horizon is None and scenario.horizon_s is not None:
                    horizon = scenario.snapshot.virtual_time + scenario.horizon_s
                await backend.advance(until=horizon, stop_on_decision=False)
            return SimulationResult(session.state, tuple(session.event_log))
        finally:
            await session.close()

    def session(self, scenario: Scenario) -> Session:
        models = deepcopy(self.models)
        models.restore(scenario.snapshot.model_states)
        clock = VirtualClock(scenario.snapshot.virtual_time)
        backend = TwinBackend(
            clock=clock,
            models=models,
            injected_events=tuple(scenario.injected_events),
            kernel=self.kernel_factory(),
        )
        return Session(
            backend,
            clock=clock,
            initial_state=scenario.materialize(),
        )


async def register_and_submit(
    session: Session, system: SystemSpec, application: ApplicationSpec
) -> str:
    await session.start()
    await session.register_system(system)
    return await session.submit_application(application)
