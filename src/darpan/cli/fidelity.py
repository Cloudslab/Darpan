"""Offline Real-vs-Twin trace fidelity CLI."""

from __future__ import annotations

import argparse
import json

from darpan.experiment.fidelity_diagnosis import (
    diagnose_trace_fidelity,
    export_fidelity_diagnosis,
)
from darpan.experiment.recorder import ResultRecorder
from darpan.experiment.trace_fidelity import (
    compare_event_traces,
    load_event_trace,
    resolve_event_trace,
)


def _run(args: argparse.Namespace) -> None:
    real_path = resolve_event_trace(args.real)
    twin_path = resolve_event_trace(args.twin)
    report = compare_event_traces(
        load_event_trace(real_path),
        load_event_trace(twin_path),
    )
    payload = report.to_dict()
    if args.diagnose:
        payload["diagnosis"] = diagnose_trace_fidelity(report)
    payload["real_events"] = str(real_path)
    payload["twin_events"] = str(twin_path)
    if args.output:
        recorder = ResultRecorder(args.output)
        recorder.write_json("fidelity.json", payload)
        recorder.copy(real_path, "inputs/real-events.jsonl")
        recorder.copy(twin_path, "inputs/twin-events.jsonl")
        recorder.write_json(
            "inputs/checksums.json",
            {
                "real-events.jsonl": recorder.sha256(real_path),
                "twin-events.jsonl": recorder.sha256(twin_path),
            },
        )
    if args.diagnosis_output:
        export_fidelity_diagnosis(
            report,
            args.diagnosis_output,
            real_events=real_path,
            twin_events=twin_path,
        )
    print(json.dumps(payload, indent=2))


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "fidelity",
        help="compare durable Real and Twin canonical event traces",
    )
    parser.add_argument("real", help="Real run directory or events.jsonl")
    parser.add_argument("twin", help="Twin run directory or events.jsonl")
    parser.add_argument("--output", help="optional directory for fidelity artifacts")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="attribute residual error by metric and aligned component/flow entity",
    )
    parser.add_argument(
        "--diagnosis-output",
        help="optional fresh sealed directory for diagnosis JSON/CSV evidence",
    )
    parser.set_defaults(func=_run)
