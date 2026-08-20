from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _event(event_id: str, duration: float) -> str:
    return json.dumps(
        {
            "id": event_id,
            "kind": "component.completed",
            "event_time": duration,
            "source": "test",
            "subject": "app:task",
            "payload": {
                "application_id": "app",
                "component_id": "task",
                "duration_s": duration,
            },
        }
    )


def test_fidelity_cli_can_write_sealed_residual_diagnosis(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"
    real.write_text(_event("r", 4) + "\n", encoding="utf-8")
    twin.write_text(_event("t", 3) + "\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "darpan.cli.main",
            "fidelity",
            str(real),
            str(twin),
            "--diagnose",
            "--diagnosis-output",
            str(tmp_path / "diagnosis"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["diagnosis"]["dominant_metric"] == "execution_duration"
    assert (tmp_path / "diagnosis" / "residuals.csv").is_file()
    assert (tmp_path / "diagnosis" / "artifact-manifest.json").is_file()
