from __future__ import annotations

import asyncio

from darpan.cli.run import run_experiment_spec
from darpan.experiment.campaign import CampaignRunner, CampaignSpec


def _experiment(tmp_path, name: str, work_units: float):
    app = tmp_path / f"{name}-app.yaml"
    app.write_text(
        "id: app\ncomponents:\n  task:\n"
        f"    work_units: {work_units}\n"
        "    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / f"{name}.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        f"application: {app.name}\n"
        "runtime: twin\n"
        "policy: first-fit\n",
        encoding="utf-8",
    )
    return experiment


def test_scenario_campaign_runs_matched_seeds_and_builds_robustness(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    baseline = _experiment(tmp_path, "baseline", 1.0)
    mild = _experiment(tmp_path, "mild", 1.5)
    severe = _experiment(tmp_path, "severe", 2.0)
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        "name: scenarios\n"
        "output: results\n"
        "seed: 3\n"
        "repeat: 2\n"
        "bootstrap_resamples: 20\n"
        "jobs:\n"
        "  - id: faults\n"
        "    kind: scenario\n"
        f"    baseline: {baseline.name}\n"
        "    metric: application_latency_s\n"
        "    higher_is_better: false\n"
        "    tolerance: 0.6\n"
        "    scenarios:\n"
        f"      mild: {mild.name}\n"
        f"      severe: {severe.name}\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign_path)

    async def execute(experiment_spec):
        return await run_experiment_spec(experiment_spec, emit_output=False)

    payload = asyncio.run(CampaignRunner(spec, execute=execute).run())
    result = payload["jobs"]["faults"]["result"]
    assert result["seeds"] == [3, 4]
    assert result["paired"]["mild"]["wins"] == 0
    assert result["paired"]["mild"]["losses"] == 2
    assert result["robustness"]["worst_case_scenario"] == "severe"
    assert result["exploration"]["deltas"]["severe"] > 0
    for seed in (3, 4):
        root = tmp_path / "results" / "runs" / "faults" / f"seed-{seed}"
        assert (root / "baseline" / "run-0001" / "events.jsonl").is_file()
        assert (root / "scenarios" / "mild" / "run-0001" / "events.jsonl").is_file()
        assert (root / "scenarios" / "severe" / "run-0001" / "events.jsonl").is_file()
