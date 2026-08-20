from __future__ import annotations

from darpan.experiment.benchmark import (
    FidelitySample,
    PairedSample,
    summarize_fidelity,
    summarize_paired,
)


def test_paired_summary_preserves_seed_matched_effects():
    summary = summarize_paired(
        [
            PairedSample(1, baseline=10.0, candidate=8.0),
            PairedSample(2, baseline=12.0, candidate=9.0),
            PairedSample(3, baseline=11.0, candidate=11.0),
        ],
        higher_is_better=False,
        bootstrap_resamples=200,
    )
    assert summary.samples == 3
    assert summary.wins == 2
    assert summary.ties == 1
    assert summary.losses == 0
    assert summary.mean_delta < 0
    assert summary.delta_ci95[0] <= summary.mean_delta <= summary.delta_ci95[1]


def test_fidelity_summary_reports_error_bias_and_interval_coverage():
    summary = summarize_fidelity(
        [
            FidelitySample(10.0, 11.0, lower=9.0, upper=12.0, uncertainty=0.2),
            FidelitySample(8.0, 10.0, lower=7.0, upper=9.0, uncertainty=0.3),
        ]
    )
    assert summary.samples == 2
    assert summary.mean_absolute_error == 1.5
    assert summary.mean_bias == 1.5
    assert summary.interval_coverage == 0.5
    assert summary.mean_uncertainty == 0.25


def test_calibration_progress_reports_sequential_error_reduction() -> None:
    from darpan.experiment.benchmark import summarize_calibration_progress

    samples = [
        FidelitySample(predicted=0.0, observed=4.0),
        FidelitySample(predicted=0.0, observed=3.0),
        FidelitySample(predicted=2.0, observed=3.0),
        FidelitySample(predicted=2.5, observed=3.0),
    ]
    report = summarize_calibration_progress(samples, window_size=2)
    assert report.initial_error == 3.5
    assert report.final_error == 0.75
    assert report.absolute_improvement == 2.75
    assert report.relative_improvement == 2.75 / 3.5
    assert report.improved
    assert [window.samples for window in report.windows] == [2, 2]
