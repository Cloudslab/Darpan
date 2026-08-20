from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from darpan._version import __version__
from darpan.experiment.artifact import seal_artifact
from darpan.experiment.recorder import ResultRecorder
from darpan.experiment.study_binding import bind_study_deployment
from darpan.experiment.study_run import StudyPlanner, StudySpec


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_physical_study(
    root: Path, *, readiness_mode: str = "required"
) -> tuple[Path, Path, Path]:
    cluster = root / "cluster.yaml"
    cluster.write_text(
        "nodes:\n"
        "  - id: edge\n"
        "    host: 127.0.0.1\n"
        "    port: 8765\n",
        encoding="utf-8",
    )
    system = root / "system.yaml"
    system.write_text(
        "nodes:\n"
        "  - id: edge\n"
        "    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    campaign = root / "campaign.yaml"
    campaign.write_text(
        "name: physical-binding\n"
        "output: results\n"
        "jobs:\n"
        "  - id: preflight\n"
        "    kind: cluster_preflight\n"
        "    cluster: cluster.yaml\n"
        "    system: system.yaml\n",
        encoding="utf-8",
    )
    study = root / "study.yaml"
    study.write_text(
        "name: physical-binding\n"
        "version: 1\n"
        "campaign: campaign.yaml\n"
        "output: study-results\n"
        "readiness:\n"
        f"  mode: {readiness_mode}\n"
        "  exercise_data_plane: true\n",
        encoding="utf-8",
    )
    return study, cluster, system


def _write_receipt(root: Path, cluster: Path, system: Path, *, ready: bool = True) -> Path:
    artifact = root / "first-run"
    artifact.mkdir()
    basis = {
        "schema": "darpan.deployment-receipt/v1",
        "darpan_version": __version__,
        "ready_for_study": ready,
        "cluster_sha256": ResultRecorder.sha256(cluster),
        "system_sha256": ResultRecorder.sha256(system),
        "environment_fingerprint": "env-123",
        "source_node_id": "edge",
        "target_node_id": "edge",
        "accepted_capabilities": ["cluster_preflight"],
    }
    receipt = {**basis, "deployment_fingerprint": _canonical_hash(basis)}
    (artifact / "deployment-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal_artifact(
        artifact,
        schema="darpan.cluster-first-run/v1",
        identity={
            "deployment_fingerprint": receipt["deployment_fingerprint"],
            "ready_for_study": ready,
        },
    )
    return artifact


def test_bind_study_deployment_writes_bound_yaml_and_matching_lock(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    study, cluster, system = _write_physical_study(source_root)
    receipt = _write_receipt(tmp_path, cluster, system)
    output_root = tmp_path / "bound"
    output_root.mkdir()
    output = output_root / "physical-study.yaml"

    payload = bind_study_deployment(study, receipt, output)

    assert payload["schema"] == "darpan.study-deployment-binding/v1"
    assert payload["deployment_fingerprint"]
    assert output.is_file()
    lock = output.with_suffix(".lock.json")
    assert lock.is_file()
    raw = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert raw["readiness"]["acceptance_receipt"] == "../first-run"
    assert raw["campaign"] == "../source/campaign.yaml"
    assert raw["lock"] == "physical-study.lock.json"

    spec = StudySpec.load(output)
    locked = StudyPlanner(spec).verify_lock()
    assert locked["fingerprint"] == payload["study_fingerprint"]
    receipt_identity = locked["readiness"]["acceptance_receipt"]
    assert receipt_identity["deployment_fingerprint"] == payload["deployment_fingerprint"]


def test_bind_study_deployment_rejects_mismatched_frozen_inputs(tmp_path: Path) -> None:
    study, cluster, system = _write_physical_study(tmp_path)
    receipt = _write_receipt(tmp_path, cluster, system)
    system.write_text(system.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksums do not match"):
        bind_study_deployment(study, receipt, tmp_path / "bound.yaml")


def test_bind_study_deployment_rejects_skipped_readiness(tmp_path: Path) -> None:
    study, cluster, system = _write_physical_study(tmp_path, readiness_mode="skip")
    receipt = _write_receipt(tmp_path, cluster, system)

    with pytest.raises(ValueError, match="readiness.mode=skip"):
        bind_study_deployment(study, receipt, tmp_path / "bound.yaml")
