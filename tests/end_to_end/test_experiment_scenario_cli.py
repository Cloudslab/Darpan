from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from darpan.experiment.spec import ExperimentSpec


def _write(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_portable_experiment_scenario_is_applied_and_recorded(tmp_path: Path) -> None:
    _write(
        tmp_path / "system.yaml",
        {
            "nodes": [
                {"id": "edge", "resources": {"cpu": 1}},
                {"id": "fog", "resources": {"cpu": 1}},
            ]
        },
    )
    _write(
        tmp_path / "app.yaml",
        {
            "id": "app",
            "components": {
                "task": {"resources": {"cpu": 1}, "work_units": 1},
            },
        },
    )
    _write(
        tmp_path / "scenario.yaml",
        {
            "name": "edge-outage",
            "events": [
                {"at_s": 0, "kind": "node.offline", "node_id": "edge"},
            ],
        },
    )
    _write(
        tmp_path / "experiment.yaml",
        {
            "system": "system.yaml",
            "application": "app.yaml",
            "runtime": "twin",
            "policy": "first-fit",
            "scenario": "scenario.yaml",
            "output": "results",
        },
    )

    completed = subprocess.run(
        [sys.executable, "-m", "darpan.cli.main", "run", str(tmp_path / "experiment.yaml")],
        cwd=tmp_path.parent,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    assert "application_latency_s" in completed.stdout

    events = [
        json.loads(line)
        for line in (tmp_path / "results" / "run-0001" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    outage = next(event for event in events if event["kind"] == "node.offline")
    assert outage["metadata"]["scenario"] == "edge-outage"
    placement = next(event for event in events if event["kind"] == "component.scheduled")
    assert placement["payload"]["node_id"] == "fog"

    manifest = json.loads(
        (tmp_path / "results" / "inputs-manifest.json").read_text(encoding="utf-8")
    )
    assert any(item["source"].endswith("scenario.yaml") for item in manifest)


def test_real_experiment_rejects_scenario_at_preflight(tmp_path: Path) -> None:
    for name in ("system.yaml", "app.yaml", "scenario.yaml"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    _write(
        experiment,
        {
            "system": "system.yaml",
            "application": "app.yaml",
            "runtime": "real",
            "scenario": "scenario.yaml",
        },
    )
    try:
        ExperimentSpec.load(experiment)
    except ValueError as exc:
        assert "real scenario requires physical_control: true" in str(exc)
    else:
        raise AssertionError("runtime: real must reject Twin scenario plans")
