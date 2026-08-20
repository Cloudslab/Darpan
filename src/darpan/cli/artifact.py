from __future__ import annotations

import json

from darpan.experiment.artifact import verify_artifact


def _verify(args) -> None:
    print(json.dumps(verify_artifact(args.directory), indent=2))


def add_parsers(subparsers) -> None:
    artifact = subparsers.add_parser(
        "artifact",
        help="verify durable Darpan experiment/campaign artifacts",
    )
    children = artifact.add_subparsers(dest="artifact_command", required=True)
    verify = children.add_parser("verify", help="verify artifact-manifest.json and files")
    verify.add_argument("directory", help="sealed experiment or campaign output directory")
    verify.set_defaults(func=_verify)
