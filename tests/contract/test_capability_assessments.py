from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.experiment import (
    ProactiveSample,
    assess_adaptation,
    assess_explainability,
    assess_proactive,
    assess_proactive_benchmark,
    assess_robustness,
    assess_trustworthiness,
    explain_action,
    explore_scenarios,
)
from darpan.twin.comparison import Comparison
from darpan.twin.prediction import Prediction
from darpan.twin.result import SimulationResult


def test_trustworthy_and_adaptive_use_expected_vs_observed_comparisons():
    before = [Comparison(8.0, 10.0), Comparison(12.0, 10.0)]
    after = [Comparison(9.5, 10.0), Comparison(10.5, 10.0)]
    trustworthy = assess_trustworthiness(after)
    adaptive = assess_adaptation(before, after)
    assert trustworthy.samples == 2
    assert trustworthy.mean_absolute_error == 0.5
    assert adaptive.improved is True
    assert adaptive.absolute_improvement == 1.5


def test_proactive_preserves_prediction_uncertainty_and_horizon():
    report = assess_proactive(
        Prediction(
            estimate=0.92,
            uncertainty=0.08,
            interval=(0.80, 1.0),
            horizon_s=30.0,
        ),
        threshold=0.9,
        relation=">",
    )
    assert report.violation_predicted is True
    assert report.horizon_s == 30.0
    assert report.interval == (0.80, 1.0)


def test_explainable_action_report_is_derived_from_canonical_lifecycle(
    small_system, small_app
):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance_id = await session.submit_application(small_app)
        action = Action.place(
            f"{instance_id}:a",
            "edge-1",
            source="research-policy",
            metadata={"reason": "lowest predicted latency", "score": 0.91},
        )
        assert await session.apply(action)
        explanation = explain_action(session.event_log, action.id)
        assert explanation.requested_by == "research-policy"
        assert explanation.accepted is True
        assert explanation.outcome == "completed"
        assert explanation.request_payload["node_id"] == "edge-1"
        assert explanation.request_metadata["score"] == 0.91
        assert explanation.policy_reason == "lowest predicted latency"
        assert "action.started" in explanation.lifecycle
        await session.close()

    asyncio.run(run())


def test_explorable_report_compares_simulation_results(small_system):
    baseline = SimulationResult(state=Darpan.twin().state, events=())
    alternative = SimulationResult(state=Darpan.twin().state, events=())
    report = explore_scenarios(
        baseline,
        {"node-failure": alternative},
        metric="event_count",
        score=lambda result: float(len(result.events)),
    )
    assert report.baseline == 0.0
    assert report.deltas["node-failure"] == 0.0


def test_robustness_supports_lower_is_better_metrics():
    report = assess_robustness(
        10.0,
        {"link-slowdown": 11.0, "node-failure": 14.0},
        higher_is_better=False,
        tolerance=0.2,
    )
    assert report.worst_case_scenario == "node-failure"
    assert report.worst_case_degradation == 0.4
    assert report.within_tolerance_fraction == 0.5


def test_proactive_benchmark_reports_detection_quality_and_lead_time():
    report = assess_proactive_benchmark(
        [
            ProactiveSample(True, True, lead_time_s=12.0, uncertainty=0.1),
            ProactiveSample(True, False, uncertainty=0.2),
            ProactiveSample(False, True, uncertainty=0.3),
            ProactiveSample(False, False, uncertainty=0.4),
            ProactiveSample(True, True, lead_time_s=8.0, uncertainty=0.5),
        ]
    )
    assert report.true_positives == 2
    assert report.false_positives == 1
    assert report.true_negatives == 1
    assert report.false_negatives == 1
    assert report.precision == 2 / 3
    assert report.recall == 2 / 3
    assert report.f1_score == 2 / 3
    assert report.accuracy == 3 / 5
    assert report.mean_true_positive_lead_time_s == 10.0
    assert report.mean_uncertainty == 0.3


def test_explainability_benchmark_measures_canonical_audit_coverage(
    small_system, small_app
):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance_id = await session.submit_application(small_app)
        action = Action.place(
            f"{instance_id}:a",
            "edge-1",
            source="research-policy",
            metadata={"reason": "lowest predicted latency"},
        )
        assert await session.apply(action)
        report = assess_explainability(session.event_log)
        assert report.actions == 1
        assert report.decision_coverage == 1.0
        assert report.terminal_coverage == 1.0
        assert report.rationale_coverage == 1.0
        assert report.payload_coverage == 1.0
        assert report.negative_outcomes == 0
        assert report.negative_reason_coverage is None
        assert report.outcomes == {"completed": 1}
        await session.close()

    asyncio.run(run())
