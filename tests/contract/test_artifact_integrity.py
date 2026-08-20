from __future__ import annotations

import asyncio
import json

import pytest

from darpan.cli.run import run_experiment_spec
from darpan.experiment.artifact import verify_artifact
from darpan.experiment.campaign import CampaignRunner, CampaignSpec
from darpan.experiment.spec import ExperimentSpec


def _experiment(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\napplication: app.yaml\nruntime: twin\n"
        "seed: 7\noutput: run-output\nmetrics: [application_latency_s]\n",
        encoding="utf-8",
    )
    return experiment


def test_experiment_artifact_is_sealed_with_stable_run_identity(tmp_path):
    experiment = _experiment(tmp_path)
    asyncio.run(run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False))

    output = tmp_path / "run-output"
    report = verify_artifact(output)
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    metadata = json.loads((output / "run-0001" / "metadata.json").read_text())

    assert report["verified"] is True
    assert manifest["schema"] == "darpan.experiment-artifact/v1"
    assert metadata["schema"] == "darpan.experiment.run/v1"
    assert metadata["experiment_fingerprint"] == manifest["identity"][
        "experiment_fingerprint"
    ]
    assert metadata["run_id"] in manifest["identity"]["run_ids"]


def test_artifact_verification_detects_changed_or_extra_files(tmp_path):
    experiment = _experiment(tmp_path)
    asyncio.run(run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False))
    output = tmp_path / "run-output"

    (output / "run-0001" / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact integrity verification failed"):
        verify_artifact(output)


def test_campaign_artifact_seals_plan_fingerprint(tmp_path):
    _experiment(tmp_path)
    # Campaign owns per-seed output paths; ignore the experiment's default output.
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "name: sealed\noutput: campaign-output\nrepeat: 1\njobs:\n"
        "  - id: paired\n    kind: paired\n    baseline: experiment.yaml\n"
        "    candidate: experiment.yaml\n    metric: application_latency_s\n",
        encoding="utf-8",
    )

    async def execute(spec):
        return await run_experiment_spec(spec, emit_output=False)

    payload = asyncio.run(
        CampaignRunner(CampaignSpec.load(campaign), execute=execute).run()
    )
    output = tmp_path / "campaign-output"
    report = verify_artifact(output)

    assert report["verified"] is True
    assert report["schema"] == "darpan.campaign-artifact/v1"
    assert report["identity"]["plan_fingerprint"] == payload["plan_fingerprint"]


def test_failed_experiment_preserves_and_seals_partial_evidence(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: edge\n    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "name: app\ncomponents:\n  task:\n    work_units: 1\n"
        "    resources:\n      cpu: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "bad_policy.py").write_text(
        "class Policy:\n"
        "    def decide(self, state, trigger):\n"
        "        raise RuntimeError('policy exploded')\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "failed.yaml"
    experiment.write_text(
        "system: system.yaml\napplication: app.yaml\nruntime: twin\n"
        "policy: bad_policy.py:Policy\noutput: failed-output\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="policy exploded"):
        asyncio.run(
            run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False)
        )

    output = tmp_path / "failed-output"
    report = verify_artifact(output)
    failure = json.loads((output / "failure.json").read_text())
    run_failure = json.loads((output / "run-0001" / "failure.json").read_text())
    assert report["identity"]["complete"] is False
    assert failure["error"]["message"] == "policy exploded"
    assert run_failure["type"] == "RuntimeError"
    assert (output / "run-0001" / "events.jsonl").is_file()
    assert (output / "run-0001" / "state.json").is_file()


def test_experiment_refuses_nonempty_output_directory(tmp_path):
    experiment = _experiment(tmp_path)
    output = tmp_path / "run-output"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        asyncio.run(
            run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False)
        )


def test_restore_artifact_checkpoint_discards_only_unsealed_files(tmp_path):
    from darpan.experiment.artifact import restore_artifact_checkpoint, seal_artifact

    sealed = tmp_path / "sealed.json"
    sealed.write_text('{"value": 1}\n', encoding="utf-8")
    seal_artifact(tmp_path, schema="test.checkpoint/v1", identity={"complete": False})
    extra = tmp_path / "inflight" / "partial.bin"
    extra.parent.mkdir()
    extra.write_bytes(b"partial")

    report = restore_artifact_checkpoint(tmp_path)

    assert report["identity"]["complete"] is False
    assert report["discarded_unsealed_files"] == ["inflight/partial.bin"]
    assert sealed.is_file()
    assert not extra.exists()
    assert (tmp_path / "artifact-manifest.json").is_file()


def test_parent_artifact_seals_nested_artifact_manifest_bytes(tmp_path):
    from darpan.experiment.artifact import seal_artifact

    child = tmp_path / "child"
    child.mkdir()
    (child / "payload.txt").write_text("evidence\n", encoding="utf-8")
    seal_artifact(child, schema="test.child/v1", identity={"complete": True})
    parent = seal_artifact(tmp_path, schema="test.parent/v1", identity={"complete": True})

    nested = next(
        item for item in parent["files"] if item["path"] == "child/artifact-manifest.json"
    )
    assert nested["sha256"]
    verify_artifact(tmp_path)

    manifest = child / "artifact-manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact integrity verification failed"):
        verify_artifact(tmp_path)


def test_restore_checkpoint_removes_unsealed_nested_manifest(tmp_path):
    from darpan.experiment.artifact import restore_artifact_checkpoint, seal_artifact

    (tmp_path / "sealed.txt").write_text("stable\n", encoding="utf-8")
    seal_artifact(tmp_path, schema="test.parent/v1", identity={"complete": False})

    inflight = tmp_path / "inflight"
    inflight.mkdir()
    (inflight / "partial.txt").write_text("partial\n", encoding="utf-8")
    (inflight / "artifact-manifest.json").write_text("{}\n", encoding="utf-8")

    report = restore_artifact_checkpoint(tmp_path)

    assert sorted(report["discarded_unsealed_files"]) == [
        "inflight/artifact-manifest.json",
        "inflight/partial.txt",
    ]
    assert not inflight.exists()
    verify_artifact(tmp_path)


def test_legacy_parent_manifest_still_verifies_with_unbound_nested_manifests(tmp_path):
    import hashlib

    from darpan._version import __version__
    from darpan.experiment.artifact import seal_artifact

    child = tmp_path / "child"
    child.mkdir()
    (child / "payload.txt").write_text("legacy evidence\n", encoding="utf-8")
    seal_artifact(child, schema="test.child/v1", identity={"complete": True})

    payload_path = child / "payload.txt"
    files = [
        {
            "path": "child/payload.txt",
            "size_bytes": payload_path.stat().st_size,
            "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        }
    ]

    def canonical(value):
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    identity = {"complete": True}
    manifest = {
        "schema": "test.legacy-parent/v1",
        "darpan_version": __version__,
        "identity": identity,
        "files": files,
        "file_count": len(files),
        "content_fingerprint": canonical(files),
    }
    manifest["manifest_fingerprint"] = canonical(
        {
            "schema": manifest["schema"],
            "darpan_version": manifest["darpan_version"],
            "identity": identity,
            "files": files,
        }
    )
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_artifact(tmp_path)
    assert report["verified"] is True
    assert report["manifest_version"] == 1
