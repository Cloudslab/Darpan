from __future__ import annotations

import asyncio
import json
from pathlib import Path

from darpan.experiment.campaign import CampaignJob, CampaignRunner, CampaignSpec


def test_campaign_checkpoints_failure_and_can_continue(tmp_path):
    spec = CampaignSpec(
        name="failure-preservation",
        output=str(tmp_path / "results"),
        continue_on_error=True,
        jobs=(
            CampaignJob("bad", "robust", {"baseline": 1.0, "scenarios": {}}),
            CampaignJob(
                "good",
                "explorable",
                {"metric": "latency", "baseline": 2.0, "alternatives": {"x": 1.0}},
            ),
        ),
    )

    async def execute(_):
        raise AssertionError("no experiment execution expected")

    payload = asyncio.run(CampaignRunner(spec, execute=execute).run())
    assert payload["complete"] is True
    assert payload["successful_jobs"] == 1
    assert payload["failed_jobs"] == 1
    assert payload["jobs"]["bad"]["status"] == "failed"
    assert payload["jobs"]["good"]["status"] == "completed"
    state = json.loads((tmp_path / "results" / "campaign-state.json").read_text())
    assert state["jobs"]["bad"]["error"].startswith("ValueError:")
    assert state["complete"] is True


def _write_resume_experiment(tmp_path, name: str, policy: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        "system: system.yaml\napplication: app.yaml\nruntime: twin\n"
        f"policy: {policy}\nmetrics: [score]\n",
        encoding="utf-8",
    )
    return path


def test_campaign_resume_restores_last_sealed_checkpoint_after_interrupt(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    _write_resume_experiment(tmp_path, "good-a", "good-a")
    _write_resume_experiment(tmp_path, "good-b", "good-b")
    _write_resume_experiment(tmp_path, "bad", "bad")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: resume\noutput: results\nrepeat: 1\njobs:\n"
        "  - id: first\n    kind: paired\n    baseline: good-a.yaml\n"
        "    candidate: good-b.yaml\n    metric: score\n"
        "  - id: second\n    kind: paired\n    baseline: good-a.yaml\n"
        "    candidate: bad.yaml\n    metric: score\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign)
    calls = {"good-a": 0, "good-b": 0, "bad": 0}
    interrupt = True

    async def execute(experiment):
        nonlocal interrupt
        calls[experiment.policy] += 1
        if experiment.output is not None:
            output = tmp_path / experiment.output
            output.mkdir(parents=True, exist_ok=True)
            (output / "partial.txt").write_text(experiment.policy, encoding="utf-8")
        if experiment.policy == "bad" and interrupt:
            raise KeyboardInterrupt("simulated process interruption")
        return {"score": 1.0, "_experiment": {"successful": True}}

    import pytest

    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        asyncio.run(CampaignRunner(spec, execute=execute).run())

    state = json.loads((tmp_path / "results" / "campaign-state.json").read_text())
    assert state["jobs"]["first"]["status"] == "completed"
    assert state["jobs"]["second"]["status"] == "pending"
    first_calls = dict(calls)

    interrupt = False
    payload = asyncio.run(CampaignRunner(spec, execute=execute, resume=True).run())

    assert payload["complete"] is True
    assert payload["successful_jobs"] == 2
    assert calls["good-b"] == first_calls["good-b"]  # completed first job was skipped
    assert calls["good-a"] == first_calls["good-a"] + 1
    assert calls["bad"] == first_calls["bad"] + 1
    from darpan.experiment.artifact import verify_artifact

    assert verify_artifact(tmp_path / "results")["verified"] is True


def test_paired_campaign_resume_skips_completed_matched_seeds(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    _write_resume_experiment(tmp_path, "baseline", "baseline")
    _write_resume_experiment(tmp_path, "candidate", "candidate")
    campaign = tmp_path / "paired-resume.yaml"
    campaign.write_text(
        "name: seed-resume\noutput: seed-results\nseed: 7\nrepeat: 3\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: baseline.yaml\n"
        "    candidate: candidate.yaml\n    metric: score\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign)
    calls = {}
    interrupt = True

    async def execute(experiment):
        nonlocal interrupt
        key = (experiment.policy, experiment.seed)
        calls[key] = calls.get(key, 0) + 1
        if experiment.policy == "candidate" and experiment.seed == 8 and interrupt:
            raise KeyboardInterrupt("interrupt second matched seed")
        return {
            "score": float(experiment.seed),
            "_experiment": {"successful": True},
        }

    import pytest

    with pytest.raises(KeyboardInterrupt, match="interrupt second matched seed"):
        asyncio.run(CampaignRunner(spec, execute=execute).run())

    progress = tmp_path / "seed-results" / "progress" / "paired"
    assert (progress / "seed-7.json").is_file()
    assert not (progress / "seed-8.json").exists()
    seed7_calls = {
        key: value for key, value in calls.items() if key[1] == 7
    }

    interrupt = False
    payload = asyncio.run(CampaignRunner(spec, execute=execute, resume=True).run())

    assert payload["complete"] is True
    assert payload["jobs"]["paired"]["result"]["summary"]["samples"] == 3
    assert {key: value for key, value in calls.items() if key[1] == 7} == seed7_calls
    assert calls[("baseline", 8)] == 2  # interrupted seed is restarted as a pair
    assert calls[("candidate", 8)] == 2
    assert calls[("baseline", 9)] == 1
    assert calls[("candidate", 9)] == 1


def test_scenario_campaign_resume_skips_completed_scenario_seeds(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    _write_resume_experiment(tmp_path, "base-scenario", "baseline")
    _write_resume_experiment(tmp_path, "fault-scenario", "fault")
    campaign = tmp_path / "scenario-resume.yaml"
    campaign.write_text(
        "name: scenario-resume\noutput: scenario-results\nseed: 3\nrepeat: 3\njobs:\n"
        "  - id: faults\n    kind: scenario\n    baseline: base-scenario.yaml\n"
        "    scenarios:\n      node-loss: fault-scenario.yaml\n"
        "    metric: score\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign)
    calls = {}
    interrupt = True

    async def execute(experiment):
        nonlocal interrupt
        key = (experiment.policy, experiment.seed)
        calls[key] = calls.get(key, 0) + 1
        if experiment.policy == "fault" and experiment.seed == 4 and interrupt:
            raise KeyboardInterrupt("interrupt second scenario seed")
        return {
            "score": float(experiment.seed),
            "_experiment": {"successful": True},
        }

    import pytest

    with pytest.raises(KeyboardInterrupt, match="interrupt second scenario seed"):
        asyncio.run(CampaignRunner(spec, execute=execute).run())

    progress = tmp_path / "scenario-results" / "progress" / "faults"
    assert (progress / "seed-3.json").is_file()
    seed3_calls = {key: value for key, value in calls.items() if key[1] == 3}

    interrupt = False
    payload = asyncio.run(CampaignRunner(spec, execute=execute, resume=True).run())

    assert payload["complete"] is True
    assert payload["jobs"]["faults"]["result"]["seeds"] == [3, 4, 5]
    assert {key: value for key, value in calls.items() if key[1] == 3} == seed3_calls
    assert calls[("baseline", 4)] == 2
    assert calls[("fault", 4)] == 2
    assert calls[("baseline", 5)] == 1
    assert calls[("fault", 5)] == 1


def test_campaign_failure_rolls_back_inflight_seed_before_sealing(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    _write_resume_experiment(tmp_path, "baseline-failure", "baseline")
    _write_resume_experiment(tmp_path, "candidate-failure", "candidate")
    campaign = tmp_path / "failure-rollback.yaml"
    campaign.write_text(
        "name: rollback\noutput: rollback-results\nseed: 5\nrepeat: 2\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: baseline-failure.yaml\n"
        "    candidate: candidate-failure.yaml\n    metric: score\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign)
    fail = True

    async def execute(experiment):
        nonlocal fail
        output = Path(experiment.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "partial.txt").write_text(
            f"{experiment.policy}:{experiment.seed}", encoding="utf-8"
        )
        if experiment.policy == "candidate" and experiment.seed == 6 and fail:
            raise RuntimeError("candidate failed after partial output")
        return {"score": float(experiment.seed), "_experiment": {"successful": True}}

    import pytest

    from darpan.experiment.artifact import verify_artifact

    with pytest.raises(RuntimeError, match="candidate failed after partial output"):
        asyncio.run(CampaignRunner(spec, execute=execute).run())

    output_root = tmp_path / "rollback-results"
    assert verify_artifact(output_root)["verified"] is True
    assert (output_root / "progress" / "paired" / "seed-5.json").is_file()
    assert not (output_root / "runs" / "paired" / "runs" / "seed-6").exists()

    fail = False
    payload = asyncio.run(CampaignRunner(spec, execute=execute, resume=True).run())
    assert payload["complete"] is True
    assert payload["jobs"]["paired"]["result"]["summary"]["samples"] == 2
