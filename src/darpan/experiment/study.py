"""Static and live readiness checks for frozen Physical paper studies."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from darpan._version import __version__
from darpan.core.codec import load_system
from darpan.experiment.artifact import (
    require_fresh_artifact_directory,
    seal_artifact,
    verify_artifact,
)
from darpan.experiment.campaign import CampaignPlanner, CampaignSpec
from darpan.experiment.provenance import collect_provenance
from darpan.experiment.recorder import ResultRecorder
from darpan.runtime.real.cluster.acceptance import (
    DEPLOYMENT_RECEIPT_SCHEMA,
    DeploymentReceipt,
    verify_deployment_receipt_payload,
)
from darpan.runtime.real.cluster.inventory import ClusterInventory
from darpan.runtime.real.cluster.validation import validate_cluster


@dataclass(frozen=True, slots=True)
class PhysicalStudyReadiness:
    ready: bool
    campaign: str
    plan_fingerprint: str
    checks: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _walk_physical(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        physical = value.get("physical")
        if isinstance(physical, Mapping):
            yield physical
        gate = value.get("physical_gate")
        if isinstance(gate, Mapping):
            yield gate
        for child in value.values():
            yield from _walk_physical(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_physical(child)


def _descriptor_path(directory: Path, descriptor: Mapping[str, Any]) -> Path:
    raw = Path(str(descriptor["path"]))
    return raw if raw.is_absolute() else (directory / raw).resolve()


def _clock_details(report) -> list[dict[str, Any]]:
    details = []
    for node in report.nodes:
        check = next((item for item in node.checks if item.name == "clock_sync"), None)
        if check is not None:
            details.append({"node_id": node.node_id, **dict(check.details)})
    return details


def _load_deployment_receipt(path: str | Path) -> tuple[DeploymentReceipt, dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    verified = verify_artifact(root)
    if verified.get("schema") not in {
        "darpan.cluster-acceptance/v1",
        "darpan.cluster-first-run/v1",
    }:
        raise ValueError(
            "deployment receipt must come from a sealed cluster acceptance/first-run artifact"
        )
    receipt_path = root / "deployment-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"deployment receipt does not exist: {receipt_path}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("deployment receipt must be a JSON object")
    receipt = verify_deployment_receipt_payload(payload)
    if receipt.darpan_version != __version__:
        raise RuntimeError(
            "deployment receipt Darpan version does not match the current Study release"
        )
    if receipt.schema != DEPLOYMENT_RECEIPT_SCHEMA:
        raise ValueError("unsupported deployment receipt schema")
    identity = verified.get("identity", {})
    if identity.get("deployment_fingerprint") != receipt.deployment_fingerprint:
        raise RuntimeError("acceptance artifact identity does not match deployment receipt")
    return receipt, verified


async def validate_physical_study(
    campaign: str | Path,
    *,
    exercise_data_plane: bool = False,
    max_clock_offset_s: float = 1.0,
    acceptance_receipt: str | Path | None = None,
    output: str | Path | None = None,
) -> PhysicalStudyReadiness:
    """Validate every unique Physical cluster/system pair in one frozen campaign."""

    if max_clock_offset_s < 0:
        raise ValueError("max_clock_offset_s cannot be negative")
    campaign_path = Path(campaign).expanduser().resolve()
    spec = CampaignSpec.load(campaign_path)
    plan = CampaignPlanner(spec).plan()
    directory = campaign_path.parent
    receipt = None
    receipt_artifact = None
    if acceptance_receipt is not None:
        receipt, receipt_artifact = _load_deployment_receipt(acceptance_receipt)
        if not receipt.ready_for_study:
            raise RuntimeError("deployment receipt is not marked ready_for_study")
    physical = list(_walk_physical(plan["jobs"]))
    if not physical:
        raise ValueError("campaign plan contains no Physical experiments or gates")

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for item in physical:
        cluster_descriptor = item.get("cluster")
        if not isinstance(cluster_descriptor, Mapping):
            continue
        system_descriptor = item.get("system")
        cluster_path = _descriptor_path(directory, cluster_descriptor)
        system_path = (
            None
            if not isinstance(system_descriptor, Mapping)
            else _descriptor_path(directory, system_descriptor)
        )
        key = (str(cluster_path), None if system_path is None else str(system_path))
        if key in seen:
            continue
        seen.add(key)
        inventory = ClusterInventory.load(cluster_path)
        system = None if system_path is None else load_system(system_path)
        report = await validate_cluster(
            inventory,
            system=system,
            exercise_data_plane=exercise_data_plane,
            max_clock_offset_s=max_clock_offset_s,
        )
        clock = _clock_details(report)
        local_errors = list(report.errors)
        if receipt is not None:
            observed_cluster_sha = ResultRecorder.sha256(cluster_path)
            observed_system_sha = (
                None if system_path is None else ResultRecorder.sha256(system_path)
            )
            if observed_cluster_sha != receipt.cluster_sha256:
                local_errors.append("deployment receipt cluster checksum does not match campaign")
            if observed_system_sha != receipt.system_sha256:
                local_errors.append("deployment receipt system checksum does not match campaign")
            if report.environment_fingerprint != receipt.environment_fingerprint:
                local_errors.append(
                    "deployment receipt environment fingerprint does not match live cluster"
                )
        requires_control = bool(item.get("physical_control", False))
        network_driver = item.get("network_driver")
        if system is not None and (requires_control or network_driver == "linux-agent"):
            inventory_by_id = {node.id: node for node in inventory.nodes}
            for node in system.nodes:
                configured = inventory_by_id[node.id]
                if not configured.physical_control:
                    local_errors.append(
                        f"node {node.id!r} must set physical_control: true in inventory"
                    )
        if bool(item.get("requires_docker", False)):
            for node_report in report.nodes:
                ping = next(
                    (check for check in node_report.checks if check.name == "ping"),
                    None,
                )
                features = {} if ping is None else dict(ping.details.get("features", {}))
                if ping is None or not ping.ok or not bool(features.get("docker")):
                    local_errors.append(
                        f"node {node_report.node_id!r} lacks Docker required by workload"
                    )
        if network_driver == "linux-agent":
            for node_report in report.nodes:
                control = next(
                    (check for check in node_report.checks if check.name == "physical_control"),
                    None,
                )
                if control is None or not control.ok or not control.details.get("route"):
                    local_errors.append(
                        f"node {node_report.node_id!r} lacks Linux route control capability"
                    )
        ready = report.ready and not local_errors
        checks.append(
            {
                "cluster": str(cluster_path),
                "system": None if system_path is None else str(system_path),
                "ready": ready,
                "environment_fingerprint": report.environment_fingerprint,
                "deployment_receipt": (
                    None
                    if receipt is None
                    else {
                        "deployment_fingerprint": receipt.deployment_fingerprint,
                        "artifact_manifest_fingerprint": receipt_artifact[
                            "manifest_fingerprint"
                        ],
                    }
                ),
                "clock": clock,
                "errors": local_errors,
                "report": report.to_dict(),
            }
        )
        errors.extend(local_errors)

    result = PhysicalStudyReadiness(
        ready=not errors and all(bool(item["ready"]) for item in checks),
        campaign=str(campaign_path),
        plan_fingerprint=str(plan["fingerprint"]),
        checks=tuple(checks),
        errors=tuple(errors),
    )
    if output is not None:
        root = require_fresh_artifact_directory(output)
        recorder = ResultRecorder(root)
        recorder.write_json("study-readiness.json", result.to_dict())
        recorder.write_json("campaign-plan.json", plan)
        recorder.write_json("provenance.json", asdict(collect_provenance()))
        seal_artifact(
            root,
            schema="darpan.study-readiness/v1",
            identity={
                "plan_fingerprint": result.plan_fingerprint,
                "ready": result.ready,
                "deployment_fingerprint": (
                    None if receipt is None else receipt.deployment_fingerprint
                ),
            },
        )
    return result
