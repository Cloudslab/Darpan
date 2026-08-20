from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from darpan.cli.run import run_experiment_spec
from darpan.experiment.spec import ExperimentSpec


def _write(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_faulted_application_is_explicitly_unsuccessful_not_a_fast_latency_win(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "system.yaml",
        {
            "nodes": [
                {"id": "edge", "resources": {"cpu": 1}},
                {"id": "fog", "resources": {"cpu": 1}},
            ],
            "links": [
                {
                    "id": "edge-fog",
                    "source": "edge",
                    "target": "fog",
                    "bandwidth_mbps": 1,
                }
            ],
        },
    )
    _write(
        tmp_path / "app.yaml",
        {
            "id": "app",
            "components": {
                "a": {"work_units": 1, "resources": {"cpu": 1}},
                "b": {"work_units": 1, "resources": {"cpu": 1}},
            },
            "flows": [
                {"source": "a", "target": "b", "data_size_bytes": 1_000_000}
            ],
        },
    )
    _write(
        tmp_path / "scenario.yaml",
        {
            "events": [
                {"at_s": 1.5, "kind": "link.down", "link_id": "edge-fog"},
            ]
        },
    )
    _write(
        tmp_path / "experiment.yaml",
        {
            "system": "system.yaml",
            "application": "app.yaml",
            "runtime": "twin",
            "policy": "round-robin",
            "scenario": "scenario.yaml",
            "metrics": ["application_latency_s", "application_success_rate"],
        },
    )

    output = asyncio.run(
        run_experiment_spec(
            ExperimentSpec.load(tmp_path / "experiment.yaml"),
            emit_output=False,
        )
    )
    assert output["application_success_rate"] == 0.0
    assert output["_experiment"]["successful"] is False
    assert output["_experiment"]["feasible"] is False
    assert output["_experiment"]["failed_instances"]
    assert output["_experiment"]["failed_components"]
