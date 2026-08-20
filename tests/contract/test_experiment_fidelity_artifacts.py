from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

from darpan.cli.run import run_experiment_spec
from darpan.experiment.spec import ExperimentSpec


def test_real_experiment_can_persist_layered_fidelity_artifacts(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "fidelity-app",
                "components": {
                    "task": {
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resources": {"cpu": 1},
                        "work_units": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: real\n"
        "fidelity_tracking: true\n"
        "output: runs\n",
        encoding="utf-8",
    )

    result = asyncio.run(run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False))
    assert "fidelity" in result
    fidelity = json.loads((tmp_path / "runs" / "run-0001" / "fidelity.json").read_text())
    assert fidelity["execution"]["summary"]["samples"] == 1
    assert fidelity["artifact"]["summary"]["samples"] == 1
    assert fidelity["network"]["summary"] is None
    assert set(fidelity["models"]) >= {"execution", "artifact_size", "network", "queue"}


def test_fidelity_tracking_rejects_twin_self_evaluation() -> None:
    with pytest.raises(ValueError, match="requires runtime: real"):
        ExperimentSpec(
            system="system.yaml",
            application="app.yaml",
            runtime="twin",
            fidelity_tracking=True,
        )
