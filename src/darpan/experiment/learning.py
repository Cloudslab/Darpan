"""Algorithm-neutral learning-efficiency evidence for paper benchmarks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmark import PairedSample, summarize_paired


@dataclass(frozen=True, slots=True)
class LearningPoint:
    quality: float
    real_interactions: int
    virtual_interactions: int = 0
    wall_time_s: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.real_interactions < 0 or self.virtual_interactions < 0:
            raise ValueError("learning interaction counts cannot be negative")
        if self.wall_time_s < 0:
            raise ValueError("learning wall_time_s cannot be negative")


@dataclass(frozen=True, slots=True)
class LearningRunSummary:
    points: int
    final_quality: float
    best_quality: float
    real_interactions: int
    virtual_interactions: int
    wall_time_s: float
    threshold: float | None
    threshold_reached: bool | None
    real_interactions_to_threshold: int | None
    virtual_interactions_to_threshold: int | None
    wall_time_to_threshold_s: float | None


def load_learning_trace(path: str | Path) -> tuple[LearningPoint, ...]:
    source = Path(path)
    records: list[dict[str, Any]]
    if source.suffix.lower() == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{source}:{line_number} must contain a JSON object")
                records.append(item)
    else:
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            raw = raw.get("points", raw.get("samples"))
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise ValueError(f"{source} must contain a list of learning points")
        records = list(raw)
    points = tuple(
        LearningPoint(
            quality=float(item["quality"]),
            real_interactions=int(item["real_interactions"]),
            virtual_interactions=int(item.get("virtual_interactions", 0)),
            wall_time_s=float(item.get("wall_time_s", 0.0)),
            metadata=dict(item.get("metadata", {})),
        )
        for item in records
    )
    _validate_learning_trace(points)
    return points


def _validate_learning_trace(points: tuple[LearningPoint, ...]) -> None:
    if not points:
        raise ValueError("learning trace requires at least one point")
    previous = points[0]
    for point in points[1:]:
        if point.real_interactions < previous.real_interactions:
            raise ValueError("real_interactions must be cumulative and nondecreasing")
        if point.virtual_interactions < previous.virtual_interactions:
            raise ValueError("virtual_interactions must be cumulative and nondecreasing")
        if point.wall_time_s < previous.wall_time_s:
            raise ValueError("wall_time_s must be cumulative and nondecreasing")
        previous = point


def write_learning_trace(
    path: str | Path,
    points: Iterable[LearningPoint],
) -> Path:
    items = tuple(points)
    _validate_learning_trace(items)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), sort_keys=True))
            handle.write("\n")
    return target


def summarize_learning_run(
    points: Iterable[LearningPoint],
    *,
    threshold: float | None = None,
    higher_is_better: bool = True,
) -> LearningRunSummary:
    items = tuple(points)
    _validate_learning_trace(items)
    qualities = tuple(item.quality for item in items)
    best = max(qualities) if higher_is_better else min(qualities)
    threshold_point = None
    if threshold is not None:
        threshold_point = next(
            (
                item
                for item in items
                if (item.quality >= threshold if higher_is_better else item.quality <= threshold)
            ),
            None,
        )
    last = items[-1]
    return LearningRunSummary(
        points=len(items),
        final_quality=last.quality,
        best_quality=best,
        real_interactions=last.real_interactions,
        virtual_interactions=last.virtual_interactions,
        wall_time_s=last.wall_time_s,
        threshold=threshold,
        threshold_reached=None if threshold is None else threshold_point is not None,
        real_interactions_to_threshold=(
            None if threshold_point is None else threshold_point.real_interactions
        ),
        virtual_interactions_to_threshold=(
            None if threshold_point is None else threshold_point.virtual_interactions
        ),
        wall_time_to_threshold_s=(
            None if threshold_point is None else threshold_point.wall_time_s
        ),
    )



def validate_learning_budget(
    summary: LearningRunSummary,
    budget: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one learning trace against explicit study resource budgets."""

    limits = dict(budget or {})
    allowed = {
        "max_real_interactions",
        "max_virtual_interactions",
        "max_wall_time_s",
        "min_points",
        "require_threshold_reached",
    }
    unknown = sorted(set(limits) - allowed)
    if unknown:
        raise ValueError(f"unknown learning budget field(s): {', '.join(unknown)}")
    violations = []
    checks = (
        ("max_real_interactions", summary.real_interactions),
        ("max_virtual_interactions", summary.virtual_interactions),
        ("max_wall_time_s", summary.wall_time_s),
    )
    for key, observed in checks:
        if key in limits and float(observed) > float(limits[key]):
            violations.append(f"{key}={observed} exceeds {limits[key]}")
    if "min_points" in limits and summary.points < int(limits["min_points"]):
        violations.append(f"points={summary.points} is below {limits['min_points']}")
    if (
        bool(limits.get("require_threshold_reached", False))
        and summary.threshold_reached is not True
    ):
        violations.append("quality threshold was not reached")
    if violations:
        raise RuntimeError(f"learning budget violated for {label}: " + "; ".join(violations))
    return {
        "compliant": True,
        "limits": limits,
        "observed": {
            "points": summary.points,
            "real_interactions": summary.real_interactions,
            "virtual_interactions": summary.virtual_interactions,
            "wall_time_s": summary.wall_time_s,
            "threshold_reached": summary.threshold_reached,
        },
    }


def compare_learning_runs(
    baseline: dict[int, LearningRunSummary],
    candidate: dict[int, LearningRunSummary],
    *,
    higher_is_better: bool,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("learning comparison requires identical baseline/candidate seeds")
    seeds = tuple(sorted(baseline))
    final_samples = tuple(
        PairedSample(seed, baseline[seed].final_quality, candidate[seed].final_quality)
        for seed in seeds
    )
    final_summary = summarize_paired(
        final_samples,
        higher_is_better=higher_is_better,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    threshold_enabled = all(item.threshold is not None for item in baseline.values())
    payload: dict[str, Any] = {
        "seeds": list(seeds),
        "baseline": {str(seed): asdict(baseline[seed]) for seed in seeds},
        "candidate": {str(seed): asdict(candidate[seed]) for seed in seeds},
        "final_quality": asdict(final_summary),
    }
    if not threshold_enabled:
        return payload

    baseline_reached = [bool(baseline[seed].threshold_reached) for seed in seeds]
    candidate_reached = [bool(candidate[seed].threshold_reached) for seed in seeds]
    both = [
        seed
        for seed in seeds
        if baseline[seed].threshold_reached and candidate[seed].threshold_reached
    ]
    threshold_payload: dict[str, Any] = {
        "baseline_reach_rate": sum(baseline_reached) / len(seeds),
        "candidate_reach_rate": sum(candidate_reached) / len(seeds),
        "both_reached_seeds": both,
    }
    if both:
        real_samples = tuple(
            PairedSample(
                seed,
                float(baseline[seed].real_interactions_to_threshold),
                float(candidate[seed].real_interactions_to_threshold),
            )
            for seed in both
        )
        time_samples = tuple(
            PairedSample(
                seed,
                float(baseline[seed].wall_time_to_threshold_s),
                float(candidate[seed].wall_time_to_threshold_s),
            )
            for seed in both
        )
        threshold_payload["real_interactions"] = asdict(
            summarize_paired(
                real_samples,
                higher_is_better=False,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
        threshold_payload["wall_time_s"] = asdict(
            summarize_paired(
                time_samples,
                higher_is_better=False,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
        threshold_payload["candidate_virtual_interactions_mean"] = mean(
            float(candidate[seed].virtual_interactions_to_threshold) for seed in both
        )
    payload["threshold"] = threshold_payload
    return payload
