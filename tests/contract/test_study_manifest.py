from __future__ import annotations

import json
from pathlib import Path

import pytest

from darpan.experiment.study_run import StudyPlanner, StudySpec


def _study_files(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        """
name: smoke
output: ignored
jobs:
  - id: robust
    kind: robust
    baseline: 10
    scenarios:
      fault: 12
    higher_is_better: false
""".lstrip(),
        encoding="utf-8",
    )
    study = tmp_path / "study.yaml"
    study.write_text(
        """
name: paper-study
version: 1
campaign: campaign.yaml
output: study-results
readiness:
  mode: skip
lock: study.lock.json
""".lstrip(),
        encoding="utf-8",
    )
    return study, campaign


def test_study_plan_has_stable_campaign_identity_and_lock_detects_drift(tmp_path: Path) -> None:
    study, campaign = _study_files(tmp_path)
    spec = StudySpec.load(study)
    planner = StudyPlanner(spec)
    plan = planner.plan()
    assert plan["schema"] == "darpan.study-plan/v1"
    assert plan["physical"] is False
    assert plan["campaign_plan_fingerprint"]
    (tmp_path / "study.lock.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert StudyPlanner(StudySpec.load(study)).verify_lock()["fingerprint"] == plan["fingerprint"]

    campaign.write_text(campaign.read_text(encoding="utf-8").replace("12", "13"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="study lock fingerprint mismatch"):
        StudyPlanner(StudySpec.load(study)).verify_lock()


def test_study_research_questions_must_be_covered_by_campaign(tmp_path: Path) -> None:
    study, _ = _study_files(tmp_path)
    study.write_text(
        study.read_text(encoding="utf-8")
        + "research_questions:\n  RQ-missing: [trustworthy]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="research question evidence is not covered"):
        StudyPlanner(StudySpec.load(study)).plan()


def test_study_plan_freezes_optional_acceptance_receipt_identity(tmp_path: Path) -> None:
    from darpan.experiment.artifact import seal_artifact
    from darpan.experiment.recorder import ResultRecorder
    from darpan.runtime.real.cluster.acceptance import (
        ClusterAcceptanceReport,
        build_deployment_receipt,
    )
    from darpan.runtime.real.cluster.validation import ClusterValidationReport

    study, _ = _study_files(tmp_path)
    cluster = tmp_path / "cluster.yaml"
    system = tmp_path / "system.yaml"
    cluster.write_text("nodes: []\n", encoding="utf-8")
    system.write_text("nodes: []\n", encoding="utf-8")
    report = ClusterAcceptanceReport(
        ready_for_study=True,
        source_node_id="edge",
        target_node_id="fog",
        preflight=ClusterValidationReport(
            ready=True,
            nodes=(),
            links=(),
            errors=(),
            environment_fingerprint="environment",
        ),
        lifecycle=None,
        steps=(),
        errors=(),
    )
    receipt = build_deployment_receipt(report, cluster=cluster, system=system)
    receipt_root = tmp_path / "acceptance"
    recorder = ResultRecorder(receipt_root)
    recorder.write_json("deployment-receipt.json", receipt.to_dict())
    recorder.write_json("cluster-acceptance.json", report.to_dict())
    seal_artifact(
        receipt_root,
        schema="darpan.cluster-acceptance/v1",
        identity={"deployment_fingerprint": receipt.deployment_fingerprint},
    )
    text = study.read_text(encoding="utf-8")
    study.write_text(
        text.replace(
            "readiness:\n  mode: skip\n",
            "readiness:\n  mode: skip\n  acceptance_receipt: acceptance\n",
        ),
        encoding="utf-8",
    )
    plan = StudyPlanner(StudySpec.load(study)).plan()
    frozen = plan["readiness"]["acceptance_receipt"]
    assert frozen["reference"] == "acceptance"
    assert frozen["deployment_fingerprint"] == receipt.deployment_fingerprint
    assert frozen["artifact_manifest_fingerprint"]
