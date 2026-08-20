"""Residual-driven Twin refinement decision CLI."""

from __future__ import annotations

import argparse
import json

from darpan.experiment.refinement import export_refinement_decision


def _run(args: argparse.Namespace) -> None:
    payload = export_refinement_decision(args.diagnosis, args.policy, args.output)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "decision": payload["decision"],
                "samples": payload["samples"],
                "authorized_targets": payload["authorized_targets"],
                "policy_predeclared_with_evidence": payload[
                    "policy_predeclared_with_evidence"
                ],
                "reasons": payload["reasons"],
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
        "refinement",
        help="gate Twin model changes using sealed Physical residual evidence",
    )
    parser.add_argument(
        "diagnosis",
        help="sealed fidelity diagnosis or fidelity-batch artifact directory",
    )
    parser.add_argument(
        "--policy",
        help=(
            "refinement policy YAML; omit when the fidelity-batch manifest predeclared "
            "and sealed one"
        ),
    )
    parser.add_argument("--output", required=True, help="fresh sealed decision directory")
    parser.set_defaults(func=_run)
