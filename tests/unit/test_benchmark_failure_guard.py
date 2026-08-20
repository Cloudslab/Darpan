from __future__ import annotations

import asyncio

import pytest

from darpan.experiment.benchmark_spec import PairedExperimentBenchmarkSpec
from darpan.experiment.paired_runner import run_paired_benchmark


async def _failed_candidate(spec):
    successful = "candidate" not in str(spec.source or "")
    return {
        "score": 1.0,
        "_experiment": {
            "successful": successful,
            "failed_instances": [] if successful else ["app-1"],
        },
    }


def test_paired_benchmark_rejects_failed_runs_by_default(tmp_path) -> None:
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    system = tmp_path / "system.yaml"
    app = tmp_path / "app.yaml"
    system.write_text("nodes: [{id: edge, resources: {cpu: 1}}]\n", encoding="utf-8")
    app.write_text("id: app\ncomponents: {task: {work_units: 1}}\n", encoding="utf-8")
    body = "system: system.yaml\napplication: app.yaml\nruntime: twin\n"
    baseline.write_text(body, encoding="utf-8")
    candidate.write_text(body, encoding="utf-8")
    spec = PairedExperimentBenchmarkSpec(
        baseline=baseline.name,
        candidate=candidate.name,
        metric="score",
        repeat=1,
        bootstrap_resamples=10,
        source=tmp_path / "benchmark.yaml",
    )

    async def execute(experiment_spec):
        failed = experiment_spec.source == candidate.resolve()
        return {
            "score": 0.1 if failed else 1.0,
            "_experiment": {
                "successful": not failed,
                "failed_instances": ["app-1"] if failed else [],
            },
        }

    with pytest.raises(RuntimeError, match="candidate seed 0 did not complete"):
        asyncio.run(run_paired_benchmark(spec, execute=execute))

    permissive = PairedExperimentBenchmarkSpec(
        baseline=baseline.name,
        candidate=candidate.name,
        metric="score",
        repeat=1,
        bootstrap_resamples=10,
        require_success=False,
        source=tmp_path / "benchmark.yaml",
    )
    payload, _ = asyncio.run(run_paired_benchmark(permissive, execute=execute))
    assert payload["summary"]["candidate_mean"] == 0.1
