from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from darpan.experiment.study import validate_physical_study
from darpan.experiment.study_binding import bind_study_deployment
from darpan.experiment.study_fidelity import build_study_fidelity_manifest
from darpan.experiment.study_run import (
    StudyPlanner,
    StudyRunner,
    StudySpec,
    verify_study,
)

from .run import run_experiment_spec


def _write_json(path: str | Path, payload) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def _validate(args: argparse.Namespace) -> None:
    report = await validate_physical_study(
        args.campaign,
        exercise_data_plane=args.exercise_data_plane,
        max_clock_offset_s=args.max_clock_offset,
        acceptance_receipt=args.acceptance_receipt,
        output=args.output,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if not report.ready:
        raise SystemExit(2)


def _plan(args: argparse.Namespace) -> None:
    payload = StudyPlanner(StudySpec.load(args.study)).plan()
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def _lock(args: argparse.Namespace) -> None:
    spec = StudySpec.load(args.study)
    payload = StudyPlanner(spec).lock_payload()
    destination = args.output
    if destination is None:
        destination = spec.resolve(spec.lock) if spec.lock is not None else None
    if destination is None:
        raise SystemExit("study lock requires --output or a top-level lock: path")
    _write_json(destination, payload)
    print(json.dumps(payload, indent=2))


async def _run(args: argparse.Namespace) -> None:
    spec = StudySpec.load(args.study)
    if args.output is not None:
        from dataclasses import replace

        spec = replace(spec, output=args.output)

    async def execute(experiment_spec):
        return await run_experiment_spec(experiment_spec, emit_output=False)

    payload = await StudyRunner(spec, execute=execute, resume=args.resume).run()
    print(json.dumps(payload, indent=2))


def _bind_deployment(args: argparse.Namespace) -> None:
    payload = bind_study_deployment(
        args.study,
        args.deployment_artifact,
        args.output,
        lock_output=args.lock_output,
    )
    print(json.dumps(payload, indent=2))


def _fidelity_plan(args: argparse.Namespace) -> None:
    payload = build_study_fidelity_manifest(args.mapping, args.output)
    print(json.dumps(payload, indent=2))


def _verify(args: argparse.Namespace) -> None:
    print(json.dumps(verify_study(args.directory), indent=2))


def add_parsers(subparsers) -> None:
    study = subparsers.add_parser(
        "study",
        help="plan, validate, run, resume, and verify frozen research studies",
    )
    commands = study.add_subparsers(dest="study_command", required=True)

    plan = commands.add_parser(
        "plan",
        help="resolve a Study Manifest into one immutable study fingerprint",
    )
    plan.add_argument("study", help="study YAML")
    plan.add_argument("--output", help="optional JSON plan path")
    plan.set_defaults(func=_plan)

    lock = commands.add_parser("lock", help="write the current frozen study plan lock")
    lock.add_argument("study", help="study YAML")
    lock.add_argument("--output", help="study lock JSON path")
    lock.set_defaults(func=_lock)

    run = commands.add_parser(
        "run",
        help="run readiness, campaign execution, and artifact sealing",
    )
    run.add_argument("study", help="study YAML")
    run.add_argument("--output", help="override study output directory")
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume the same frozen study without rerunning completed stages",
    )
    run.set_defaults(func=lambda args: asyncio.run(_run(args)))

    bind = commands.add_parser(
        "bind-deployment",
        help="bind a Physical Study to a verified cluster acceptance/first-run receipt",
    )
    bind.add_argument("study", help="source Physical Study YAML")
    bind.add_argument(
        "deployment_artifact",
        help="sealed cluster acceptance/first-run artifact directory",
    )
    bind.add_argument("--output", required=True, help="fresh bound Study YAML")
    bind.add_argument(
        "--lock-output",
        help="fresh Study lock JSON; defaults beside --output",
    )
    bind.set_defaults(func=_bind_deployment)

    fidelity_plan = commands.add_parser(
        "fidelity-plan",
        help="derive a sealed-input fidelity-batch manifest from two verified Studies",
    )
    fidelity_plan.add_argument("mapping", help="Study fidelity mapping YAML")
    fidelity_plan.add_argument(
        "--output", required=True, help="fresh generated fidelity-batch YAML"
    )
    fidelity_plan.set_defaults(func=_fidelity_plan)

    verify = commands.add_parser(
        "verify",
        help="verify the top-level study and every nested sealed artifact",
    )
    verify.add_argument("directory", help="completed sealed study directory")
    verify.set_defaults(func=_verify)

    validate = commands.add_parser(
        "validate",
        help="validate Physical campaign readiness without running workloads",
    )
    validate.add_argument("campaign", help="frozen campaign YAML")
    validate.add_argument(
        "--exercise-data-plane",
        action="store_true",
        help="also exercise Agent probes and direct artifact transfer",
    )
    validate.add_argument(
        "--max-clock-offset",
        type=float,
        default=1.0,
        help="maximum allowed clock offset in seconds after RTT uncertainty",
    )
    validate.add_argument(
        "--acceptance-receipt",
        help="sealed cluster accept/first-run artifact to bind readiness to prior acceptance",
    )
    validate.add_argument("--output", help="optional sealed readiness artifact directory")
    validate.set_defaults(func=lambda args: asyncio.run(_validate(args)))
