from __future__ import annotations

from darpan import Darpan, DigitalTwin
from darpan.adapters.rl.action import PlacementActionAdapter
from darpan.adapters.rl.observation import PlacementObservation
from darpan.adapters.rl.problem import RLProblem
from darpan.adapters.rl.reward import CompletionTimeReward
from darpan.runtime.event_log import InMemoryEventLog
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.executors.local import LocalExecutor
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend
from darpan.twin.models.registry import ModelRegistry


class FalsyLog(InMemoryEventLog):
    def __bool__(self):
        return False


class FalsyRealBackend(RealBackend):
    def __bool__(self):
        return False


class FalsyExecutor(LocalExecutor):
    def __bool__(self):
        return False


class FalsyModels(ModelRegistry):
    def __bool__(self):
        return False


class FalsyObservation(PlacementObservation):
    def __bool__(self):
        return False


class FalsyAction(PlacementActionAdapter):
    def __bool__(self):
        return False


class FalsyReward(CompletionTimeReward):
    def __bool__(self):
        return False


def test_session_preserves_empty_custom_log_and_explicit_empty_validators():
    log = FalsyLog()
    session = Session(TwinBackend(), event_log=log, validators=[])
    assert session.event_log is log
    assert session.validators == []


def test_facades_and_backends_preserve_falsy_injected_dependencies():
    backend = FalsyRealBackend()
    assert Darpan.real(backend=backend).backend is backend

    executor = FalsyExecutor()
    real = RealBackend(default_executor=executor)
    assert real.default_executor is executor

    models = FalsyModels()
    twin = TwinBackend(models=models)
    assert twin.models is models
    assert DigitalTwin(models=models).models is models


def test_rl_problem_preserves_falsy_custom_adapters():
    observation = FalsyObservation()
    action = FalsyAction()
    reward = FalsyReward()
    problem = RLProblem.placement(
        observation=observation,
        action=action,
        reward=reward,
    )
    assert problem.observation is observation
    assert problem.action is action
    assert problem.reward is reward
