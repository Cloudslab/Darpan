from __future__ import annotations

import asyncio
import json

from darpan.cli.run import run_experiment_spec
from darpan.experiment.campaign import CampaignRunner, CampaignSpec
from darpan.experiment.suite import PaperSuite


def test_campaign_resolves_locked_suite_experiment_references(tmp_path):
    system = tmp_path / "system.yaml"
    system.write_text(
        "nodes:\n"
        "  - id: edge\n"
        "    resources:\n"
        "      cpu: 1\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        "name: app\n"
        "components:\n"
        "  task:\n"
        "    work_units: 1\n"
        "    resources:\n"
        "      cpu: 1\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n"
        "policy: first-fit\n"
        "metrics: [application_latency_s]\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "name: frozen-paper\n"
        "version: '1'\n"
        "items:\n"
        "  - id: baseline\n"
        "    kind: experiment\n"
        "    path: experiment.yaml\n"
        "  - id: candidate\n"
        "    kind: experiment\n"
        "    path: experiment.yaml\n",
        encoding="utf-8",
    )
    lock = tmp_path / "suite.lock.json"
    lock.write_text(
        json.dumps(PaperSuite.load(suite_path).lock_payload()),
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: frozen\n"
        "output: results\n"
        "suite: suite.yaml\n"
        "suite_lock: suite.lock.json\n"
        "repeat: 1\n"
        "bootstrap_resamples: 10\n"
        "jobs:\n"
        "  - id: compare\n"
        "    kind: paired\n"
        "    baseline: suite:baseline\n"
        "    candidate: suite:candidate\n"
        "    metric: application_latency_s\n"
        "    higher_is_better: false\n",
        encoding="utf-8",
    )

    async def execute(spec):
        return await run_experiment_spec(spec, emit_output=False)

    spec = CampaignSpec.load(campaign)
    payload = asyncio.run(CampaignRunner(spec, execute=execute).run())

    assert payload["successful_jobs"] == 1
    assert payload["suite"]["locked"] is True
    assert payload["suite"]["name"] == "frozen-paper"
    assert len(payload["suite"]["fingerprint"]) == 64
    assert (tmp_path / "results" / "inputs" / "suite.yaml").is_file()
    assert (tmp_path / "results" / "inputs" / "suite.lock.json").is_file()
    assert (tmp_path / "results" / "suite-lock-observed.json").is_file()
