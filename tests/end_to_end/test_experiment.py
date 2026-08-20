from __future__ import annotations

from darpan import Darpan
from darpan.experiment.baselines import RoundRobinPolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner


def test_experiment_runs_end_to_end(small_system, small_app):
    session = Darpan.twin()
    runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
    result = runner.run_sync(small_system, small_app, RoundRobinPolicy())
    assert result.metrics["application_latency_s"] > 0
