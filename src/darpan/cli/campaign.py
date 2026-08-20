"""Paper benchmark campaign CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from darpan.experiment.campaign import CampaignPlanner, CampaignRunner, CampaignSpec

from .run import run_experiment_spec


async def _run(args: argparse.Namespace) -> None:
    spec = CampaignSpec.load(args.campaign)
    if args.output is not None:
        spec = replace(spec, output=args.output)
    if args.plan:
        payload = CampaignPlanner(spec).plan()
        if args.plan_output is not None:
            destination = Path(args.plan_output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2))
        return

    async def execute(experiment_spec):
        return await run_experiment_spec(experiment_spec, emit_output=False)

    payload = await CampaignRunner(spec, execute=execute, resume=args.resume).run()
    print(json.dumps(payload, indent=2))


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "campaign",
        help="run a multi-job paper benchmark campaign",
    )
    parser.add_argument("campaign", help="campaign YAML")
    parser.add_argument("--output", help="override campaign output directory")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="validate and fingerprint the campaign without executing it",
    )
    parser.add_argument(
        "--plan-output",
        help="optional JSON path for --plan output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an incomplete sealed campaign output with the same frozen plan",
    )
    parser.set_defaults(func=lambda args: asyncio.run(_run(args)))
