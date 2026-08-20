from __future__ import annotations

from darpan.experiment.learning import (
    LearningPoint,
    compare_learning_runs,
    summarize_learning_run,
)


def test_learning_summary_and_paired_real_interaction_efficiency() -> None:
    baseline = {}
    candidate = {}
    for seed in (7, 8):
        baseline[seed] = summarize_learning_run(
            (
                LearningPoint(0.2, 10, 0, 1.0),
                LearningPoint(0.8, 40, 0, 4.0),
                LearningPoint(0.9, 60, 0, 6.0),
            ),
            threshold=0.8,
        )
        candidate[seed] = summarize_learning_run(
            (
                LearningPoint(0.3, 10, 30, 1.5),
                LearningPoint(0.85, 20, 70, 3.0),
                LearningPoint(0.92, 30, 100, 4.0),
            ),
            threshold=0.8,
        )

    report = compare_learning_runs(
        baseline,
        candidate,
        higher_is_better=True,
        bootstrap_resamples=20,
    )
    assert report["threshold"]["baseline_reach_rate"] == 1.0
    assert report["threshold"]["candidate_reach_rate"] == 1.0
    real = report["threshold"]["real_interactions"]
    assert real["baseline_mean"] == 40
    assert real["candidate_mean"] == 20
    assert real["wins"] == 2
    assert report["threshold"]["candidate_virtual_interactions_mean"] == 70
    assert report["final_quality"]["wins"] == 2
