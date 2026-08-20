"""Statistical building blocks for reproducible Darpan paper benchmarks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from random import Random
from statistics import mean
from typing import Any

from .aggregation import aggregate


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    metric: str
    statistics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PairedSample:
    seed: int
    baseline: float
    candidate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


@dataclass(frozen=True, slots=True)
class PairedBenchmarkSummary:
    samples: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    mean_relative_change: float
    wins: int
    ties: int
    losses: int
    delta_ci95: tuple[float, float]
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class FidelitySample:
    predicted: float
    observed: float
    lower: float | None = None
    upper: float | None = None
    uncertainty: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def residual(self) -> float:
        return self.observed - self.predicted

    @property
    def relative_error(self) -> float:
        if abs(self.observed) < 1e-12:
            return 0.0 if abs(self.predicted) < 1e-12 else float("inf")
        return abs(self.predicted - self.observed) / abs(self.observed)

    @property
    def covered(self) -> bool | None:
        if self.lower is None or self.upper is None:
            return None
        return self.lower <= self.observed <= self.upper


@dataclass(frozen=True, slots=True)
class FidelityBenchmarkSummary:
    samples: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    mean_bias: float
    interval_coverage: float | None
    mean_uncertainty: float | None


@dataclass(frozen=True, slots=True)
class CalibrationWindow:
    start_index: int
    end_index: int
    samples: int
    mean_absolute_error: float
    root_mean_squared_error: float


@dataclass(frozen=True, slots=True)
class CalibrationProgressSummary:
    samples: int
    window_size: int
    initial_error: float
    final_error: float
    absolute_improvement: float
    relative_improvement: float | None
    windows: tuple[CalibrationWindow, ...]

    @property
    def improved(self) -> bool:
        return self.final_error < self.initial_error


def summarize(metric: str, values: list[float]) -> BenchmarkSummary:
    return BenchmarkSummary(metric, aggregate(values))


def _bootstrap_mean_ci95(
    values: tuple[float, ...],
    *,
    seed: int = 0,
    resamples: int = 2000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1:
        return values[0], values[0]
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = Random(seed)
    n = len(values)
    samples = sorted(
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    low = samples[int(0.025 * (resamples - 1))]
    high = samples[int(0.975 * (resamples - 1))]
    return low, high


def summarize_paired(
    samples: Iterable[PairedSample],
    *,
    higher_is_better: bool = True,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> PairedBenchmarkSummary:
    items = tuple(samples)
    if not items:
        raise ValueError("paired benchmark requires at least one sample")
    baselines = tuple(float(item.baseline) for item in items)
    candidates = tuple(float(item.candidate) for item in items)
    deltas = tuple(
        candidate - baseline
        for baseline, candidate in zip(baselines, candidates, strict=True)
    )
    relative = tuple(
        0.0 if abs(baseline) < 1e-12 and abs(delta) < 1e-12
        else math.copysign(float("inf"), delta)
        if abs(baseline) < 1e-12
        else delta / abs(baseline)
        for baseline, delta in zip(baselines, deltas, strict=True)
    )
    finite_relative = tuple(value for value in relative if math.isfinite(value))

    def outcome(delta: float) -> int:
        signed = delta if higher_is_better else -delta
        if abs(signed) <= 1e-12:
            return 0
        return 1 if signed > 0 else -1

    outcomes = tuple(outcome(delta) for delta in deltas)
    return PairedBenchmarkSummary(
        samples=len(items),
        baseline_mean=mean(baselines),
        candidate_mean=mean(candidates),
        mean_delta=mean(deltas),
        mean_relative_change=(
            mean(finite_relative) if finite_relative else float("nan")
        ),
        wins=sum(value > 0 for value in outcomes),
        ties=sum(value == 0 for value in outcomes),
        losses=sum(value < 0 for value in outcomes),
        delta_ci95=_bootstrap_mean_ci95(
            deltas,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
        higher_is_better=higher_is_better,
    )


def summarize_fidelity(samples: Iterable[FidelitySample]) -> FidelityBenchmarkSummary:
    items = tuple(samples)
    if not items:
        raise ValueError("fidelity benchmark requires at least one sample")
    residuals = tuple(item.residual for item in items)
    relative = tuple(item.relative_error for item in items)
    finite_relative = tuple(value for value in relative if math.isfinite(value))
    covered = tuple(item.covered for item in items if item.covered is not None)
    uncertainties = tuple(
        float(item.uncertainty)
        for item in items
        if item.uncertainty is not None
    )
    return FidelityBenchmarkSummary(
        samples=len(items),
        mean_absolute_error=mean(abs(value) for value in residuals),
        root_mean_squared_error=math.sqrt(mean(value * value for value in residuals)),
        mean_absolute_percentage_error=(
            mean(finite_relative) if finite_relative else float("inf")
        ),
        mean_bias=mean(residuals),
        interval_coverage=(
            None if not covered else sum(bool(value) for value in covered) / len(covered)
        ),
        mean_uncertainty=(None if not uncertainties else mean(uncertainties)),
    )


def summarize_calibration_progress(
    samples: Iterable[FidelitySample],
    *,
    window_size: int | None = None,
) -> CalibrationProgressSummary:
    """Summarize whether sequential out-of-sample fidelity improves over time."""

    items = tuple(samples)
    if len(items) < 2:
        raise ValueError("calibration progress requires at least two samples")
    if window_size is None:
        window_size = max(1, min(10, len(items) // 4 or 1))
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    window_size = min(window_size, len(items))
    windows = []
    for start in range(0, len(items), window_size):
        chunk = items[start : start + window_size]
        summary = summarize_fidelity(chunk)
        windows.append(
            CalibrationWindow(
                start_index=start,
                end_index=start + len(chunk) - 1,
                samples=len(chunk),
                mean_absolute_error=summary.mean_absolute_error,
                root_mean_squared_error=summary.root_mean_squared_error,
            )
        )
    initial = summarize_fidelity(items[:window_size]).mean_absolute_error
    final = summarize_fidelity(items[-window_size:]).mean_absolute_error
    improvement = initial - final
    relative = None if abs(initial) < 1e-12 else improvement / initial
    return CalibrationProgressSummary(
        samples=len(items),
        window_size=window_size,
        initial_error=initial,
        final_error=final,
        absolute_improvement=improvement,
        relative_improvement=relative,
        windows=tuple(windows),
    )


def benchmark_payload(
    *,
    samples: Iterable[PairedSample] | Iterable[FidelitySample],
    summary: PairedBenchmarkSummary | FidelityBenchmarkSummary,
) -> dict[str, Any]:
    return {
        "summary": asdict(summary),
        "samples": [asdict(item) for item in samples],
    }
