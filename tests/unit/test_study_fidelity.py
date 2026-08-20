from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from darpan.experiment.artifact import seal_artifact
from darpan.experiment.fidelity_batch import FidelityBatchSpec
from darpan.experiment.study_fidelity import build_study_fidelity_manifest


def _write_arm(root: Path, *, runtime: str, seed: int) -> None:
    (root / "run-0001").mkdir(parents=True)
    (root / "experiment-portable.json").write_text(
        json.dumps({"runtime": runtime}) + "\n",
        encoding="utf-8",
    )
    (root / "run-0001" / "metadata.json").write_text(
        json.dumps({"seed": seed}) + "\n",
        encoding="utf-8",
    )
    (root / "run-0001" / "events.jsonl").write_text("", encoding="utf-8")
    seal_artifact(
        root,
        schema="darpan.experiment-artifact/v1",
        identity={"complete": True, "experiment_fingerprint": f"exp-{runtime}-{seed}"},
    )


def _write_study(root: Path, *, runtime: str, seeds: tuple[int, ...]) -> None:
    campaign = root / "campaign"
    for seed in seeds:
        for arm in ("baseline", "candidate"):
            _write_arm(
                campaign
                / "runs"
                / "placement"
                / "runs"
                / f"seed-{seed}"
                / arm,
                runtime=runtime,
                seed=seed,
            )
    campaign_plan = {
        "fingerprint": f"campaign-plan-{runtime}",
        "jobs": [{"id": "placement", "kind": "paired", "seeds": list(seeds)}],
    }
    (campaign / "campaign-plan.json").write_text(
        json.dumps(campaign_plan, indent=2) + "\n",
        encoding="utf-8",
    )
    campaign_result = {
        "complete": True,
        "plan_fingerprint": campaign_plan["fingerprint"],
        "failed_jobs": 0,
        "jobs": {
            "placement": {
                "kind": "paired",
                "status": "completed",
                "result": {
                    "samples": [
                        {"seed": seed, "baseline": 1.0, "candidate": 1.0}
                        for seed in seeds
                    ]
                },
            }
        },
    }
    (campaign / "campaign.json").write_text(
        json.dumps(campaign_result, indent=2) + "\n",
        encoding="utf-8",
    )
    campaign_manifest = seal_artifact(
        campaign,
        schema="darpan.campaign-artifact/v1",
        identity={"complete": True, "plan_fingerprint": campaign_plan["fingerprint"]},
    )

    study_fingerprint = f"study-{runtime}"
    study_plan = {
        "fingerprint": study_fingerprint,
        "physical": False,
        "campaign_plan": campaign_plan,
    }
    (root / "study-plan.json").write_text(
        json.dumps(study_plan, indent=2) + "\n",
        encoding="utf-8",
    )
    study_result = {
        "complete": True,
        "study_fingerprint": study_fingerprint,
        "campaign_plan_fingerprint": campaign_plan["fingerprint"],
        "artifacts": {
            "campaign": {
                "manifest_fingerprint": campaign_manifest["manifest_fingerprint"]
            }
        },
    }
    (root / "study.json").write_text(
        json.dumps(study_result, indent=2) + "\n",
        encoding="utf-8",
    )
    seal_artifact(
        root,
        schema="darpan.study-artifact/v1",
        identity={"study_fingerprint": study_fingerprint, "complete": True},
    )


def _write_mapping(
    root: Path,
    *,
    real_study: Path,
    twin_study: Path,
    seeds: tuple[int, ...] = (),
) -> Path:
    mapping = root / "mapping.yaml"
    payload = {
        "schema": "darpan.study-fidelity-map/v1",
        "name": "physical-vs-twin",
        "real_study": str(real_study),
        "twin_study": str(twin_study),
        "pairs": [
            {
                "id": "firstfit",
                "real": {"job": "placement", "arm": "baseline"},
                "twin": {"job": "placement", "arm": "baseline"},
                **({} if not seeds else {"seeds": list(seeds)}),
            }
        ],
    }
    mapping.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return mapping


def test_build_study_fidelity_manifest_expands_verified_matched_seeds(tmp_path: Path) -> None:
    real = tmp_path / "real-study"
    twin = tmp_path / "twin-study"
    real.mkdir()
    twin.mkdir()
    _write_study(real, runtime="real", seeds=(201, 202))
    _write_study(twin, runtime="twin", seeds=(201, 202))
    mapping = _write_mapping(tmp_path, real_study=real, twin_study=twin)
    output = tmp_path / "batch.yaml"

    result = build_study_fidelity_manifest(mapping, output)

    assert result["pair_count"] == 2
    raw = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert raw["require_sealed_inputs"] is True
    assert [item["id"] for item in raw["pairs"]] == [
        "firstfit-seed-201",
        "firstfit-seed-202",
    ]
    assert raw["pairs"][0]["source"]["mapping_id"] == "firstfit"
    assert raw["pairs"][0]["source"]["seed"] == 201
    assert raw["pairs"][0]["source"]["real_experiment_manifest_fingerprint"]
    spec = FidelityBatchSpec.load(output)
    assert len(spec.pairs) == 2
    for pair in spec.pairs:
        assert spec.resolve_trace(pair.real).sealed is True
        assert spec.resolve_trace(pair.twin).sealed is True


def test_build_study_fidelity_manifest_rejects_unmatched_seed_sets(tmp_path: Path) -> None:
    real = tmp_path / "real-study"
    twin = tmp_path / "twin-study"
    real.mkdir()
    twin.mkdir()
    _write_study(real, runtime="real", seeds=(201, 202))
    _write_study(twin, runtime="twin", seeds=(201,))
    mapping = _write_mapping(tmp_path, real_study=real, twin_study=twin)

    with pytest.raises(ValueError, match="seed sets differ"):
        build_study_fidelity_manifest(mapping, tmp_path / "batch.yaml")


def test_build_study_fidelity_manifest_accepts_frozen_common_seed_subset(tmp_path: Path) -> None:
    real = tmp_path / "real-study"
    twin = tmp_path / "twin-study"
    real.mkdir()
    twin.mkdir()
    _write_study(real, runtime="real", seeds=(201, 202))
    _write_study(twin, runtime="twin", seeds=(201,))
    mapping = _write_mapping(
        tmp_path,
        real_study=real,
        twin_study=twin,
        seeds=(201,),
    )

    result = build_study_fidelity_manifest(mapping, tmp_path / "batch.yaml")

    assert result["pair_count"] == 1
    assert result["pairs"][0]["id"] == "firstfit-seed-201"


def test_build_study_fidelity_manifest_rejects_wrong_runtime_side(tmp_path: Path) -> None:
    real = tmp_path / "real-study"
    twin = tmp_path / "twin-study"
    real.mkdir()
    twin.mkdir()
    _write_study(real, runtime="twin", seeds=(201,))
    _write_study(twin, runtime="twin", seeds=(201,))
    mapping = _write_mapping(tmp_path, real_study=real, twin_study=twin)

    with pytest.raises(ValueError, match="real arm has runtime='twin'"):
        build_study_fidelity_manifest(mapping, tmp_path / "batch.yaml")
