"""Batch Real-vs-Twin fidelity CLI."""

from __future__ import annotations

import argparse
import json

from darpan.experiment.fidelity_batch import run_fidelity_batch


def _run(args: argparse.Namespace) -> None:
    payload = run_fidelity_batch(args.manifest, args.output)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "name": payload["name"],
                "pairs": payload["pair_count"],
                "samples": payload["diagnosis"]["samples"],
                "sealed_inputs_required": payload["require_sealed_inputs"],
                "refinement_policy_bound": payload["refinement_policy"] is not None,
                "dominant_metric": payload["diagnosis"]["dominant_metric"],
                "artifact_manifest_fingerprint": payload[
                    "artifact_manifest_fingerprint"
                ],
                "output": payload["output"],
            },
            indent=2,
        )
    )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "fidelity-batch",
        help="aggregate a frozen manifest of matched Real/Twin event traces",
    )
    parser.add_argument("manifest", help="YAML fidelity-batch manifest")
    parser.add_argument("--output", required=True, help="fresh sealed output directory")
    parser.set_defaults(func=_run)
