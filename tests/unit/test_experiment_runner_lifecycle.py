from __future__ import annotations

import asyncio

import pytest

from darpan import Darpan
from darpan.experiment.baselines import RoundRobinPolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner
from darpan.runtime.dispatch import PolicyDispatcher


def test_single_application_runner_unsubscribes_dispatcher_after_success(
    small_system, small_app
):
    async def run():
        session = Darpan.twin()
        runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
        await runner.run(small_system, small_app, RoundRobinPolicy())
        assert not any(isinstance(item, PolicyDispatcher) for item in session._listeners)
        await session.close()

    asyncio.run(run())


def test_single_application_runner_unsubscribes_dispatcher_after_failure(
    small_system, small_app
):
    class FailingPolicy:
        def decide(self, event, state):
            raise RuntimeError("policy failed")

    async def run():
        session = Darpan.twin()
        runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
        with pytest.raises(RuntimeError, match="policy failed"):
            await runner.run(small_system, small_app, FailingPolicy())
        assert not any(isinstance(item, PolicyDispatcher) for item in session._listeners)
        await session.close()

    asyncio.run(run())
