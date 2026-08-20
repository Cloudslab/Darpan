from __future__ import annotations

import asyncio
import json
import sys

import pytest

from darpan.cli.run import run_experiment_spec
from darpan.experiment.spec import ExperimentSpec


def _write_inputs(tmp_path, *, command: bool = False):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    component = "    resources: {cpu: 1}\n    work_units: 1\n"
    if command:
        component += f"    command: [{sys.executable!r}, -c, 'print(1)']\n"
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task:\n" + component,
        encoding="utf-8",
    )


def _execution_snapshot(mean: float) -> dict:
    return {
        "execution": {
            "calibration_rate": 0.25,
            "estimates": {
                "app|task|n1": {"mean": mean, "variance": 0.0, "count": 10}
            },
        }
    }


def test_twin_experiment_can_warm_start_from_models_json(tmp_path):
    _write_inputs(tmp_path)
    (tmp_path / "models.json").write_text(
        json.dumps(_execution_snapshot(7.5)),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n"
        "model_snapshot: models.json\n"
        "output: runs\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False)
    )
    assert result["application_latency_s"] == 7.5
    manifest = json.loads((tmp_path / "runs" / "inputs-manifest.json").read_text())
    assert any(item["source"].endswith("models.json") for item in manifest)


def test_real_fidelity_can_warm_start_from_prior_fidelity_artifact(tmp_path):
    _write_inputs(tmp_path, command=True)
    (tmp_path / "prior-fidelity.json").write_text(
        json.dumps({"models": _execution_snapshot(4.25)}),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: real\n"
        "fidelity_tracking: true\n"
        "model_snapshot: prior-fidelity.json\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False)
    )
    sample = result["fidelity"]["execution"]["samples"][0]
    assert sample["predicted"] == 4.25
    assert sample["metadata"]["calibrated"] is True
    assert sample["metadata"]["samples"] == 10


def test_real_model_snapshot_requires_fidelity_tracking():
    with pytest.raises(ValueError, match="requires fidelity_tracking"):
        ExperimentSpec(
            system="system.yaml",
            application="app.yaml",
            runtime="real",
            model_snapshot="models.json",
        )
