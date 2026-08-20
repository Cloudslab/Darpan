"""Execution of seed-matched paired experiment benchmarks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from typing import Any

from .benchmark import PairedSample, benchmark_payload, summarize_paired
from .benchmark_spec import PairedExperimentBenchmarkSpec
from .spec import ExperimentSpec

ExperimentExecutor = Callable[[ExperimentSpec], Awaitable[dict[str, Any]]]
SampleCallback = Callable[[PairedSample], Awaitable[None]]


def ensure_successful(output: dict[str, Any], *, label: str = "experiment") -> None:
    status = output.get("_experiment")
    if isinstance(status, dict) and status.get("successful") is False:
        failed = status.get("failed_instances", [])
        raise RuntimeError(f"{label} did not complete successfully; failed instances: {failed}")


def experiment_success(output: dict[str, Any]) -> bool:
    status = output.get("_experiment")
    if not isinstance(status, dict):
        return True
    return bool(status.get("successful", True))


def metric_value(output: dict[str, Any], metric: str) -> float:
    if metric not in output:
        raise KeyError(f"experiment result does not contain metric {metric!r}")
    value = output[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"benchmark metric must be numeric: {metric!r}")
    return float(value)


async def run_paired_benchmark(
    spec: PairedExperimentBenchmarkSpec,
    *,
    execute: ExperimentExecutor,
    initial_samples: Iterable[PairedSample] = (),
    on_sample: SampleCallback | None = None,
) -> tuple[dict[str, Any], tuple[PairedSample, ...]]:
    baseline_source = spec.resolve(spec.baseline)
    candidate_source = spec.resolve(spec.candidate)
    baseline_template = ExperimentSpec.load(baseline_source)
    candidate_template = ExperimentSpec.load(candidate_source)
    initial = {item.seed: item for item in initial_samples}
    unknown = set(initial) - set(spec.run_seeds)
    if unknown:
        raise ValueError(f"initial paired samples contain unexpected seeds: {sorted(unknown)}")
    samples: list[PairedSample] = []

    run_root = None if spec.output is None else spec.resolve(spec.output) / "runs"
    for seed in spec.run_seeds:
        if seed in initial:
            samples.append(initial[seed])
            continue
        seed_root = None if run_root is None else run_root / f"seed-{seed}"
        baseline_spec = replace(
            baseline_template,
            repeat=1,
            seed=seed,
            output=None if seed_root is None else str(seed_root / "baseline"),
        )
        candidate_spec = replace(
            candidate_template,
            repeat=1,
            seed=seed,
            output=None if seed_root is None else str(seed_root / "candidate"),
        )
        baseline = await execute(baseline_spec)
        candidate = await execute(candidate_spec)
        if spec.require_success:
            ensure_successful(baseline, label=f"baseline seed {seed}")
            ensure_successful(candidate, label=f"candidate seed {seed}")
        sample = PairedSample(
            seed=seed,
            baseline=metric_value(baseline, spec.metric),
            candidate=metric_value(candidate, spec.metric),
            metadata={
                "baseline_successful": experiment_success(baseline),
                "candidate_successful": experiment_success(candidate),
            },
        )
        samples.append(sample)
        if on_sample is not None:
            await on_sample(sample)

    items = tuple(samples)
    summary = summarize_paired(
        items,
        higher_is_better=spec.higher_is_better,
        bootstrap_seed=spec.seed,
        bootstrap_resamples=spec.bootstrap_resamples,
    )
    payload = benchmark_payload(samples=items, summary=summary)
    payload["metric"] = spec.metric
    payload["baseline"] = str(baseline_source)
    payload["candidate"] = str(candidate_source)
    payload["success"] = {
        "baseline_rate": sum(
            bool(item.metadata.get("baseline_successful", True)) for item in items
        )
        / len(items),
        "candidate_rate": sum(
            bool(item.metadata.get("candidate_successful", True)) for item in items
        )
        / len(items),
        "both_successful_rate": sum(
            bool(item.metadata.get("baseline_successful", True))
            and bool(item.metadata.get("candidate_successful", True))
            for item in items
        )
        / len(items),
    }
    return payload, items
