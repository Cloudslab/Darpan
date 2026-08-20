from __future__ import annotations

import pytest

from darpan.experiment.learning import LearningRunSummary, validate_learning_budget


def _summary() -> LearningRunSummary:
    return LearningRunSummary(
        points=3,
        final_quality=0.9,
        best_quality=0.9,
        real_interactions=60,
        virtual_interactions=200,
        wall_time_s=8.0,
        threshold=0.8,
        threshold_reached=True,
        real_interactions_to_threshold=50,
        virtual_interactions_to_threshold=150,
        wall_time_to_threshold_s=6.0,
    )


def test_learning_budget_records_compliance() -> None:
    evidence = validate_learning_budget(
        _summary(),
        {
            "max_real_interactions": 70,
            "max_virtual_interactions": 250,
            "max_wall_time_s": 10,
            "min_points": 2,
            "require_threshold_reached": True,
        },
        label="candidate seed 7",
    )
    assert evidence["compliant"] is True
    assert evidence["observed"]["real_interactions"] == 60


def test_learning_budget_rejects_over_budget_trace() -> None:
    with pytest.raises(RuntimeError, match="max_real_interactions"):
        validate_learning_budget(
            _summary(),
            {"max_real_interactions": 50},
            label="candidate seed 7",
        )
