from __future__ import annotations

import json

import pytest

from darpan.experiment.campaign import CampaignPlanner, CampaignSpec
from darpan.experiment.suite import PaperSuite


def _write_experiment(root, *, metric="application_latency_s"):
    (root / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (root / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    experiment = root / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\napplication: app.yaml\nruntime: twin\n"
        f"metrics: [{metric}]\n",
        encoding="utf-8",
    )
    return experiment


def test_campaign_plan_is_stable_and_counts_seed_matched_runs(tmp_path):
    _write_experiment(tmp_path)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: paper\nversion: '1'\nitems:\n"
        "  - id: exp\n    kind: experiment\n    path: experiment.yaml\n",
        encoding="utf-8",
    )
    lock = tmp_path / "suite.lock.json"
    lock.write_text(
        json.dumps(PaperSuite.load(suite).lock_payload()), encoding="utf-8"
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: paper\noutput: results\nsuite: suite.yaml\n"
        "suite_lock: suite.lock.json\nrepeat: 3\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: suite:exp\n"
        "    candidate: suite:exp\n    metric: application_latency_s\n",
        encoding="utf-8",
    )

    first = CampaignPlanner(CampaignSpec.load(campaign)).plan()
    second = CampaignPlanner(CampaignSpec.load(campaign)).plan()

    assert first == second
    assert first["schema"] == "darpan.campaign.plan/v1"
    assert first["total_execution_runs"] == 6
    assert first["jobs"][0]["seeds"] == [0, 1, 2]
    assert first["suite"]["locked"] is True
    assert len(first["fingerprint"]) == 64


def test_campaign_plan_rejects_metric_not_declared_by_experiment(tmp_path):
    _write_experiment(tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: bad\noutput: results\nrepeat: 1\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: missing_metric\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not declared"):
        CampaignPlanner(CampaignSpec.load(campaign)).plan()


def test_campaign_plan_resolves_custom_metric_output_name(tmp_path):
    (tmp_path / "custom_metric.py").write_text(
        "class CustomMetric:\n"
        "    name = 'foreground_latency_s'\n"
        "    def reset(self): pass\n"
        "    def observe(self, event, state): pass\n"
        "    def result(self): return 0.0\n",
        encoding="utf-8",
    )
    _write_experiment(tmp_path, metric="custom_metric.py:CustomMetric")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: custom\noutput: results\nrepeat: 1\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: foreground_latency_s\n",
        encoding="utf-8",
    )

    plan = CampaignPlanner(CampaignSpec.load(campaign)).plan()

    assert plan["jobs"][0]["metric"] == "foreground_latency_s"
    assert plan["jobs"][0]["baseline"]["metrics"] == ["foreground_latency_s"]


def test_campaign_plan_lock_rejects_statistical_config_drift(tmp_path):
    _write_experiment(tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: locked\noutput: results\nrepeat: 2\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: application_latency_s\n",
        encoding="utf-8",
    )
    initial = CampaignPlanner(CampaignSpec.load(campaign)).plan()
    lock = tmp_path / "plan.json"
    lock.write_text(json.dumps(initial), encoding="utf-8")
    campaign.write_text(
        "name: locked\noutput: results\nrepeat: 2\nplan_lock: plan.json\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: application_latency_s\n"
        "    bootstrap_resamples: 123\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="campaign plan lock mismatch"):
        CampaignPlanner(CampaignSpec.load(campaign)).plan()


def test_suite_required_evidence_must_be_covered_by_campaign(tmp_path):
    _write_experiment(tmp_path)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: coverage\nversion: '1'\nrequired_evidence: [trustworthy, robust]\n"
        "items:\n  - id: exp\n    kind: experiment\n    path: experiment.yaml\n",
        encoding="utf-8",
    )
    lock = tmp_path / "suite.lock.json"
    lock.write_text(
        json.dumps(PaperSuite.load(suite).lock_payload()), encoding="utf-8"
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: coverage\noutput: results\nsuite: suite.yaml\n"
        "suite_lock: suite.lock.json\nrepeat: 1\njobs:\n"
        "  - id: paired\n    kind: paired\n    evidence_for: [trustworthy]\n"
        "    baseline: suite:exp\n    candidate: suite:exp\n"
        "    metric: application_latency_s\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="robust"):
        CampaignPlanner(CampaignSpec.load(campaign)).plan()
