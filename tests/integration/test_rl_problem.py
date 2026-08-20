from __future__ import annotations

import pytest

from darpan import Darpan
from darpan.adapters.rl import DarpanEnv, RLProblem
from darpan.experiment.baselines import FirstFitPolicy


def _controls_component(component_id: str):
    def predicate(state, decision):
        del state
        return decision.component_id == component_id

    return predicate


def test_partial_control_delegates_uncontrolled_decision(small_system, small_app):
    problem = RLProblem.placement(
        control=_controls_component("b"),
        fallback_policy=FirstFitPolicy(),
    )
    env = DarpanEnv(
        session_factory=Darpan.twin,
        system=small_system,
        application=small_app,
        problem=problem,
    )
    try:
        _, info = env.reset()
        assert info["decision_instance_id"].endswith(":b")

        valid = [index for index, allowed in enumerate(info["action_mask"]) if allowed]
        _, _, terminated, truncated, info = env.step(valid[0])

        assert terminated
        assert not truncated
        assert info["decision_instance_id"] is None
    finally:
        env.close()


def test_partial_control_requires_fallback_for_unowned_decision(
    small_system,
    small_app,
):
    problem = RLProblem.placement(control=_controls_component("b"))
    env = DarpanEnv(
        session_factory=Darpan.twin,
        system=small_system,
        application=small_app,
        problem=problem,
    )
    try:
        with pytest.raises(RuntimeError, match="no fallback_policy"):
            env.reset()
    finally:
        env.close()


def test_problem_action_size_is_explicit(small_system, small_app):
    base = RLProblem.placement()
    problem = RLProblem(
        observation=base.observation,
        action=base.action,
        reward=base.reward,
        action_size=7,
    )
    env = DarpanEnv(
        session_factory=Darpan.twin,
        system=small_system,
        application=small_app,
        problem=problem,
    )
    try:
        assert env.action_size == 7
    finally:
        env.close()
