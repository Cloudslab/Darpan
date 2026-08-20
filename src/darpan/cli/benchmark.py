"""Seed-matched paired experiment benchmark CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from darpan.experiment.benchmark_spec import PairedExperimentBenchmarkSpec
from darpan.experiment.paired_runner import run_paired_benchmark
from darpan.experiment.recorder import ResultRecorder

from .run import run_experiment_spec


def _recorder(spec: PairedExperimentBenchmarkSpec) -> ResultRecorder | None:
    if spec.output is None:
        return None
    return ResultRecorder(spec.resolve(spec.output))


async def _run_paired(spec: PairedExperimentBenchmarkSpec) -> dict:
    async def execute(experiment_spec):
        return await run_experiment_spec(experiment_spec, emit_output=False)

    payload, samples = await run_paired_benchmark(spec, execute=execute)
    recorder = _recorder(spec)
    if recorder is not None:
        baseline_source = spec.resolve(spec.baseline)
        candidate_source = spec.resolve(spec.candidate)
        recorder.write_json("benchmark.json", payload)
        recorder.write_jsonl("samples.jsonl", (asdict(item) for item in samples))
        recorder.copy(spec.source, "inputs/benchmark.yaml")
        recorder.copy(baseline_source, "inputs/baseline.yaml")
        recorder.copy(candidate_source, "inputs/candidate.yaml")
        recorder.write_json(
            "inputs/checksums.json",
            {
                "benchmark.yaml": recorder.sha256(spec.source),
                "baseline.yaml": recorder.sha256(baseline_source),
                "candidate.yaml": recorder.sha256(candidate_source),
            },
        )
    return payload


async def _run(args: argparse.Namespace) -> None:
    payload = await _run_paired(PairedExperimentBenchmarkSpec.load(args.benchmark))
    print(json.dumps(payload, indent=2))


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="run a seed-matched baseline/candidate benchmark",
    )
    parser.add_argument("benchmark", help="paired benchmark YAML")
    parser.set_defaults(func=lambda args: asyncio.run(_run(args)))
