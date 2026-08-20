from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from darpan.experiment.artifact import verify_artifact


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


def test_fidelity_batch_then_refinement_cli_produces_sealed_evidence(tmp_path: Path) -> None:
    for pair_id, real_duration, twin_duration in (
        ("seed-1", 4.0, 3.0),
        ("seed-2", 5.0, 4.0),
    ):
        (tmp_path / f"{pair_id}-real.jsonl").write_text(
            _event(f"{pair_id}-r", real_duration) + "\n",
            encoding="utf-8",
        )
        (tmp_path / f"{pair_id}-twin.jsonl").write_text(
            _event(f"{pair_id}-t", twin_duration) + "\n",
            encoding="utf-8",
        )
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: dev12-smoke\n"
        "refinement_policy: policy.yaml\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: seed-1-real.jsonl\n"
        "    twin: seed-1-twin.jsonl\n"
        "  - id: seed-2\n"
        "    real: seed-2-real.jsonl\n"
        "    twin: seed-2-twin.jsonl\n",
        encoding="utf-8",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "schema: darpan.twin-refinement-policy/v1\n"
        "minimum_total_samples: 2\n"
        "maximum_uncovered_error_contribution: 0.2\n"
        "max_candidates: 1\n"
        "rules:\n"
        "  - metric: execution_duration\n"
        "    target: twin.models.execution\n"
        "    minimum_samples: 2\n"
        "    maximum_acceptable_mae: 0.5\n"
        "    minimum_error_contribution: 0.1\n"
        "    required: true\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    batch_output = tmp_path / "batch"
    batch = subprocess.run(
        [
            sys.executable,
            "-m",
            "darpan.cli.main",
            "fidelity-batch",
            str(manifest),
            "--output",
            str(batch_output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert batch.returncode == 0, batch.stderr
    batch_payload = json.loads(batch.stdout)
    assert batch_payload["pairs"] == 2
    assert batch_payload["dominant_metric"] == "execution_duration"
    assert verify_artifact(batch_output)["verified"] is True

    decision_output = tmp_path / "decision"
    decision = subprocess.run(
        [
            sys.executable,
            "-m",
            "darpan.cli.main",
            "refinement",
            str(batch_output),
            "--output",
            str(decision_output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert decision.returncode == 0, decision.stderr
    decision_payload = json.loads(decision.stdout)
    assert decision_payload["decision"] == "refine"
    assert decision_payload["authorized_targets"] == ["twin.models.execution"]
    assert verify_artifact(decision_output)["verified"] is True
