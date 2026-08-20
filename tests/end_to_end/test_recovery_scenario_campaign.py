from __future__ import annotations

import asyncio

from darpan.cli.run import run_experiment_spec
from darpan.experiment.campaign import CampaignRunner, CampaignSpec


def test_scenario_campaign_reports_retry_recovery_evidence(tmp_path) -> None:
    (tmp_path / "system.yaml").write_text(
        """
nodes:
  - id: edge
    resources: {cpu: 1}
  - id: fog
    resources: {cpu: 1}
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        """
id: recovery-app
components:
  task:
    work_units: 4
    resources: {cpu: 1}
    retry:
      max_retries: 1
      on: [node_offline]
      backoff_s: 0.5
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "fault.yaml").write_text(
        """
events:
  - at_s: 1
    kind: node.offline
    node_id: edge
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "baseline.yaml").write_text(
        """
system: system.yaml
application: app.yaml
runtime: twin
policy: first-fit
metrics: [application_latency_s]
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "fault-experiment.yaml").write_text(
        """
system: system.yaml
application: app.yaml
runtime: twin
policy: first-fit
scenario: fault.yaml
metrics: [application_latency_s]
""".lstrip(),
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        """
name: recovery-campaign
output: results
seed: 9
repeat: 2
bootstrap_resamples: 20
jobs:
  - id: node-loss
    kind: scenario
    baseline: baseline.yaml
    metric: application_latency_s
    higher_is_better: false
    scenarios:
      node-loss: fault-experiment.yaml
""".lstrip(),
        encoding="utf-8",
    )

    async def execute(spec):
        return await run_experiment_spec(spec, emit_output=False)

    payload = asyncio.run(CampaignRunner(CampaignSpec.load(campaign), execute=execute).run())
    result = payload["jobs"]["node-loss"]["result"]
    baseline = result["recovery"]["baseline"]
    fault = result["recovery"]["scenarios"]["node-loss"]
    assert baseline["total_retry_attempts"] == 0
    assert baseline["component_recovery_rate"] is None
    assert fault["runs"] == 2
    assert fault["total_retry_attempts"] == 2
    assert fault["retried_components"] == 2
    assert fault["recovered_components"] == 2
    assert fault["exhausted_components"] == 0
    assert fault["component_recovery_rate"] == 1.0
    assert fault["mean_retry_downtime_s"] is not None
    assert fault["mean_retry_downtime_s"] >= 0.5
    assert result["success"]["scenario_rates"]["node-loss"] == 1.0
    assert (tmp_path / "results" / "runs" / "node-loss" / "seed-9").is_dir()
