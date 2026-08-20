from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from darpan.core.event import Event, EventKind


def _write_trace(path: Path, duration: float) -> None:
    event = Event(
        kind=EventKind.COMPONENT_COMPLETED,
        event_time=duration,
        source="test",
        payload={
            "application_id": "app",
            "component_id": "work",
            "duration_s": duration,
        },
    )
    path.write_text(json.dumps(event.to_dict()) + "\n")


def test_fidelity_cli_compares_and_persists_run_artifacts(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"
    output = tmp_path / "report"
    _write_trace(real, 2.0)
    _write_trace(twin, 1.5)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "darpan.cli.main",
            "fidelity",
            str(real),
            str(twin),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["execution_duration"]["summary"]["mean_absolute_error"] == 0.5
    assert (output / "fidelity.json").is_file()
    assert (output / "inputs" / "checksums.json").is_file()
