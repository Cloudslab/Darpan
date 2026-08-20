from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from darpan.cli.campaign import _run


def test_campaign_plan_cli_does_not_create_campaign_run_directory(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        "system: system.yaml\napplication: app.yaml\nmetrics: [application_latency_s]\n",
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: plan\noutput: should-not-exist\nrepeat: 1\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: application_latency_s\n",
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    asyncio.run(
        _run(
            SimpleNamespace(
                campaign=str(campaign),
                output=None,
                plan=True,
                plan_output=str(output),
            )
        )
    )

    payload = json.loads(output.read_text())
    assert payload["total_execution_runs"] == 2
    assert not (tmp_path / "should-not-exist").exists()
