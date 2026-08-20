from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "darpan.cli.main", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )


def test_campaign_cli_runs_paired_and_all_six_capability_jobs(tmp_path):
    system = tmp_path / "system.yaml"
    system.write_text(
        """
name: local
nodes:
  - id: edge
    resources:
      cpu: 2
links: []
""".lstrip(),
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        """
id: one
components:
  task:
    work_units: 1
    resources:
      cpu: 1
""".lstrip(),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """
system: system.yaml
application: app.yaml
runtime: twin
policy: round-robin
metrics: [application_latency_s]
""".lstrip(),
        encoding="utf-8",
    )

    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "id": "e1",
                        "kind": "action.requested",
                        "event_time": 0.0,
                        "subject": "app:task",
                        "source": "policy",
                        "payload": {
                            "action_id": "a1",
                            "kind": "component.place",
                            "action_payload": {"node_id": "edge"},
                            "metadata": {"reason": "lowest latency"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "id": "e2",
                        "kind": "action.accepted",
                        "event_time": 0.0,
                        "subject": "app:task",
                        "source": "session",
                        "payload": {"action_id": "a1", "kind": "component.place"},
                    }
                ),
                json.dumps(
                    {
                        "id": "e3",
                        "kind": "action.completed",
                        "event_time": 1.0,
                        "subject": "app:task",
                        "source": "session",
                        "payload": {"action_id": "a1", "kind": "component.place"},
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(
        json.dumps(
            {
                "execution": {
                    "samples": [
                        {"predicted": 2.0, "observed": 4.0},
                        {"predicted": 3.0, "observed": 4.0},
                        {"predicted": 3.5, "observed": 4.0},
                        {"predicted": 4.0, "observed": 4.0},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    proactive = tmp_path / "proactive.jsonl"
    proactive.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "predicted_violation": True,
                        "observed_violation": True,
                        "lead_time_s": 5,
                        "uncertainty": 0.2,
                    }
                ),
                json.dumps(
                    {
                        "predicted_violation": False,
                        "observed_violation": False,
                        "uncertainty": 0.1,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        """
name: campaign-smoke
output: results
seed: 11
repeat: 2
bootstrap_resamples: 20
jobs:
  - id: policy
    kind: paired
    baseline: experiment.yaml
    candidate: experiment.yaml
    metric: application_latency_s
    higher_is_better: false
  - id: fidelity
    kind: fidelity
    real: events.jsonl
    twin: events.jsonl
  - id: trustworthy
    kind: trustworthy
    fidelity: fidelity.json
    layer: execution
  - id: adaptive
    kind: adaptive
    fidelity: fidelity.json
    layer: execution
    window_size: 1
  - id: proactive
    kind: proactive
    samples: proactive.jsonl
  - id: explainable
    kind: explainable
    trace: events.jsonl
  - id: robust
    kind: robust
    baseline: 10
    scenarios:
      cpu-loss: 12
      link-loss: 11
    higher_is_better: false
    tolerance: 0.25
  - id: explorable
    kind: explorable
    metric: latency
    baseline: 10
    alternatives:
      edge: 8
      cloud: 14
""".lstrip(),
        encoding="utf-8",
    )

    result = _run_cli("campaign", str(campaign), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "campaign-smoke"
    assert payload["successful_jobs"] == 8
    assert payload["jobs"]["policy"]["result"]["summary"]["ties"] == 2
    assert payload["jobs"]["adaptive"]["result"]["improved"] is True
    assert payload["jobs"]["proactive"]["result"]["f1_score"] == 1.0
    assert payload["jobs"]["explainable"]["result"]["rationale_coverage"] == 1.0
    assert payload["jobs"]["robust"]["result"]["worst_case_scenario"] == "cpu-loss"
    assert payload["jobs"]["explorable"]["result"]["deltas"]["edge"] == -2.0
    assert payload["complete"] is True
    assert payload["failed_jobs"] == 0
    assert (tmp_path / "results" / "campaign.json").is_file()
    assert (tmp_path / "results" / "campaign-state.json").is_file()
    assert (tmp_path / "results" / "provenance.json").is_file()
    assert (tmp_path / "results" / "inputs" / "checksums.json").is_file()
