from __future__ import annotations

import asyncio
import json

from darpan.experiment.campaign import CampaignRunner, CampaignSpec


def test_learning_campaign_compares_real_only_and_twin_assisted_by_seed(tmp_path) -> None:
    baseline = {}
    candidate = {}
    for seed in (1, 2):
        base = tmp_path / f"real-{seed}.jsonl"
        twin = tmp_path / f"twin-{seed}.jsonl"
        base.write_text(
            '\n'.join(
                json.dumps(item)
                for item in (
                    {"quality": 0.2, "real_interactions": 10, "wall_time_s": 1},
                    {"quality": 0.8, "real_interactions": 50, "wall_time_s": 5},
                )
            )
            + '\n',
            encoding="utf-8",
        )
        twin.write_text(
            '\n'.join(
                json.dumps(item)
                for item in (
                    {
                        "quality": 0.3,
                        "real_interactions": 10,
                        "virtual_interactions": 20,
                        "wall_time_s": 1.5,
                    },
                    {
                        "quality": 0.85,
                        "real_interactions": 25,
                        "virtual_interactions": 60,
                        "wall_time_s": 3,
                    },
                )
            )
            + '\n',
            encoding="utf-8",
        )
        baseline[str(seed)] = base.name
        candidate[str(seed)] = twin.name

    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: learning\noutput: results\nbootstrap_resamples: 20\njobs:\n"
        "  - id: sample-efficiency\n"
        "    kind: learning\n"
        "    threshold: 0.8\n"
        "    higher_is_better: true\n"
        f"    baseline: {json.dumps(baseline)}\n"
        f"    candidate: {json.dumps(candidate)}\n",
        encoding="utf-8",
    )
    spec = CampaignSpec.load(campaign)

    async def unused_execute(_):
        raise AssertionError("learning evidence must not execute an experiment")

    result = asyncio.run(CampaignRunner(spec, execute=unused_execute).run())
    report = result["jobs"]["sample-efficiency"]["result"]
    assert report["threshold"]["real_interactions"]["candidate_mean"] == 25
    assert report["threshold"]["real_interactions"]["baseline_mean"] == 50
    assert (tmp_path / "results" / "inputs" / "checksums.json").is_file()
