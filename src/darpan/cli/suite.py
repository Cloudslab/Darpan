from __future__ import annotations

import json
from pathlib import Path

from darpan.experiment.recorder import ResultRecorder
from darpan.experiment.suite import PaperSuite


def _default_lock_path(source: Path) -> Path:
    return source.with_suffix(".lock.json")


def _lock(args) -> None:
    suite = PaperSuite.load(args.suite)
    payload = suite.lock_payload()
    output = (
        _default_lock_path(suite.source)
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(output), **payload}, indent=2))


def _verify(args) -> None:
    suite = PaperSuite.load(args.suite)
    lock = (
        _default_lock_path(suite.source)
        if args.lock is None
        else Path(args.lock).expanduser().resolve()
    )
    payload = suite.verify_lock(lock)
    rendered = {
        "verified": True,
        "suite": str(suite.source),
        "lock": str(lock),
        "name": suite.name,
        "version": suite.version,
        "fingerprint": payload["fingerprint"],
        "items": len(suite.items),
    }
    print(json.dumps(rendered, indent=2))
    if args.output is not None:
        recorder = ResultRecorder(Path(args.output).expanduser().resolve())
        recorder.write_json("suite-verification.json", rendered)
        recorder.copy(suite.source, "inputs/suite.yaml")
        recorder.copy(lock, "inputs/suite.lock.json")
        recorder.write_json(
            "inputs/checksums.json",
            {
                "suite.yaml": recorder.sha256(suite.source),
                "suite.lock.json": recorder.sha256(lock),
            },
        )


def add_parsers(subparsers) -> None:
    suite = subparsers.add_parser(
        "suite",
        help="freeze and verify named paper workload/baseline/evidence suites",
    )
    children = suite.add_subparsers(dest="suite_command", required=True)

    lock = children.add_parser("lock", help="write a deterministic suite lock")
    lock.add_argument("suite", help="paper suite YAML")
    lock.add_argument("--output", help="lock JSON path (default: beside suite YAML)")
    lock.set_defaults(func=_lock)

    verify = children.add_parser("verify", help="verify suite inputs against its lock")
    verify.add_argument("suite", help="paper suite YAML")
    verify.add_argument("--lock", help="lock JSON path (default: beside suite YAML)")
    verify.add_argument("--output", help="optional durable verification report directory")
    verify.set_defaults(func=_verify)
