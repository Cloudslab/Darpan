from __future__ import annotations

import argparse

from .artifact import add_parsers as add_artifact_parsers
from .benchmark import add_parser as add_benchmark_parser
from .campaign import add_parser as add_campaign_parser
from .cluster import add_parsers as add_cluster_parsers
from .fidelity import add_parser as add_fidelity_parser
from .fidelity_batch import add_parser as add_fidelity_batch_parser
from .refinement import add_parser as add_refinement_parser
from .replay import add_parser as add_replay_parser
from .run import add_parser as add_run_parser
from .study import add_parsers as add_study_parsers
from .suite import add_parsers as add_suite_parsers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darpan",
        description="Darpan Continuum Computing framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    add_artifact_parsers(subparsers)
    add_benchmark_parser(subparsers)
    add_campaign_parser(subparsers)
    add_fidelity_parser(subparsers)
    add_fidelity_batch_parser(subparsers)
    add_refinement_parser(subparsers)
    add_cluster_parsers(subparsers)
    add_replay_parser(subparsers)
    add_suite_parsers(subparsers)
    add_study_parsers(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
