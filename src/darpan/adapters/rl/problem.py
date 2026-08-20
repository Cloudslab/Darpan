"""Composable RL problem definition independent of any specific algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from darpan.core.event import Event, EventKind
from darpan.core.protocols.policy import Policy
from darpan.core.state import ComponentInstanceState, ContinuumState
from darpan.core.topology import SystemSpec

from .action import PlacementActionAdapter, RLActionAdapter
from .observation import ObservationAdapter, PlacementObservation
from .reward import CompletionTimeReward, RewardFunction

ControlPredicate = Callable[[ContinuumState, ComponentInstanceState], bool]
ActionSize = int | Callable[[SystemSpec], int]


def _control_all(
    state: ContinuumState,
    decision: ComponentInstanceState,
) -> bool:
    del state, decision
    return True


@dataclass(frozen=True, slots=True)
class RLProblem:
    """Defines the learning-facing boundary of a Darpan control problem.

    The runtime, application and Digital Twin remain Darpan concerns.  An
    RLProblem only decides how a learner observes a decision, maps its action
    to canonical Darpan actions, computes reward, and which ready decisions the
    learner owns.  Ready decisions outside ``control`` can be delegated to a
    normal Darpan policy through ``fallback_policy``.  This makes partial
    control explicit instead of forcing every application decision through RL.
    """

    observation: ObservationAdapter
    action: RLActionAdapter
    reward: RewardFunction
    action_size: ActionSize
    control: ControlPredicate = _control_all
    fallback_policy: Policy | None = None

    def action_count(self, system: SystemSpec) -> int:
        value = self.action_size(system) if callable(self.action_size) else self.action_size
        count = int(value)
        if count <= 0:
            raise ValueError("RL action size must be positive")
        return count

    def is_controlled(
        self,
        state: ContinuumState,
        decision: ComponentInstanceState,
    ) -> bool:
        return bool(self.control(state, decision))

    def fallback_actions(
        self,
        state: ContinuumState,
        decision: ComponentInstanceState,
    ):
        if self.fallback_policy is None:
            return None
        trigger = Event(
            kind=EventKind.COMPONENT_READY,
            event_time=state.time,
            source="adapter.rl",
            subject=decision.id,
            correlation_id=decision.application_instance_id,
            payload={
                "instance_id": decision.id,
                "application_id": decision.application_id,
                "component_id": decision.component_id,
            },
        )
        return self.fallback_policy.decide(state, trigger)

    @classmethod
    def placement(
        cls,
        *,
        observation: ObservationAdapter | None = None,
        action: RLActionAdapter | None = None,
        reward: RewardFunction | None = None,
        control: ControlPredicate = _control_all,
        fallback_policy: Policy | None = None,
    ) -> RLProblem:
        """Build the standard placement problem used by Darpan v0.1."""

        return cls(
            observation=observation if observation is not None else PlacementObservation(),
            action=action if action is not None else PlacementActionAdapter(),
            reward=reward if reward is not None else CompletionTimeReward(),
            action_size=lambda system: len(system.nodes),
            control=control,
            fallback_policy=fallback_policy,
        )
