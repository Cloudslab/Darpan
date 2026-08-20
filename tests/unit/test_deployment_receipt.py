from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from darpan.runtime.real.cluster.acceptance import (
    ClusterAcceptanceReport,
    DeploymentReceipt,
    build_deployment_receipt,
    verify_deployment_receipt_payload,
)
from darpan.runtime.real.cluster.validation import ClusterValidationReport


def _report() -> ClusterAcceptanceReport:
    return ClusterAcceptanceReport(
        ready_for_study=True,
        source_node_id="edge",
        target_node_id="fog",
        preflight=ClusterValidationReport(
            ready=True,
            nodes=(),
            links=(),
            errors=(),
            environment_fingerprint="environment-123",
        ),
        lifecycle=None,
        steps=(),
        errors=(),
    )


def test_deployment_receipt_binds_inputs_environment_and_fingerprint(tmp_path: Path) -> None:
    cluster = tmp_path / "cluster.yaml"
    system = tmp_path / "system.yaml"
    cluster.write_text("nodes: []\n", encoding="utf-8")
    system.write_text("nodes: []\n", encoding="utf-8")

    receipt = build_deployment_receipt(_report(), cluster=cluster, system=system)
    verified = verify_deployment_receipt_payload(receipt.to_dict())

    assert isinstance(verified, DeploymentReceipt)
    assert verified.ready_for_study is True
    assert verified.environment_fingerprint == "environment-123"
    assert len(verified.deployment_fingerprint) == 64
    assert "data_plane" in verified.accepted_capabilities

    tampered = receipt.to_dict()
    tampered["system_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint"):
        verify_deployment_receipt_payload(tampered)


def test_deployment_receipt_fingerprint_changes_with_environment(tmp_path: Path) -> None:
    cluster = tmp_path / "cluster.yaml"
    system = tmp_path / "system.yaml"
    cluster.write_text("nodes: []\n", encoding="utf-8")
    system.write_text("nodes: []\n", encoding="utf-8")
    first = build_deployment_receipt(_report(), cluster=cluster, system=system)
    changed_report = replace(
        _report(),
        preflight=replace(_report().preflight, environment_fingerprint="changed"),
    )
    second = build_deployment_receipt(changed_report, cluster=cluster, system=system)
    assert first.deployment_fingerprint != second.deployment_fingerprint
