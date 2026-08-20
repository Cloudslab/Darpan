"""Bind a frozen Physical Study to one verified deployment receipt."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from darpan._version import __version__
from darpan.experiment.artifact import verify_artifact
from darpan.experiment.recorder import ResultRecorder
from darpan.runtime.real.cluster.acceptance import (
    DeploymentReceipt,
    verify_deployment_receipt_payload,
)

from .study_run import StudyPlanner, StudySpec

_DEPLOYMENT_ARTIFACT_SCHEMAS = frozenset(
    {"darpan.cluster-acceptance/v1", "darpan.cluster-first-run/v1"}
)


def _portable_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _walk_physical(value: Any):
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


def _load_receipt(root: Path) -> tuple[DeploymentReceipt, dict[str, Any]]:
    verification = verify_artifact(root)
    if verification.get("schema") not in _DEPLOYMENT_ARTIFACT_SCHEMAS:
        raise ValueError(
            "deployment binding requires a sealed cluster acceptance/first-run artifact"
        )
    receipt_path = root / "deployment-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"deployment receipt does not exist: {receipt_path}")
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("deployment receipt must contain a JSON object")
    receipt = verify_deployment_receipt_payload(raw)
    if receipt.darpan_version != __version__:
        raise RuntimeError(
            "deployment receipt Darpan version does not match the current Study release"
        )
    if not receipt.ready_for_study:
        raise RuntimeError("deployment receipt is not marked ready_for_study")
    identity = verification.get("identity", {})
    if identity.get("deployment_fingerprint") != receipt.deployment_fingerprint:
        raise RuntimeError("deployment artifact identity does not match deployment receipt")
    return receipt, verification


def _physical_input_pairs(plan: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in _walk_physical(plan.get("jobs", ())):
        cluster = item.get("cluster")
        system = item.get("system")
        if not isinstance(cluster, Mapping) or not isinstance(system, Mapping):
            continue
        cluster_sha = cluster.get("sha256")
        system_sha = system.get("sha256")
        if cluster_sha is None or system_sha is None:
            continue
        pairs.add((str(cluster_sha), str(system_sha)))
    return pairs


def bind_study_deployment(
    study: str | Path,
    deployment_artifact: str | Path,
    output_study: str | Path,
    *,
    lock_output: str | Path | None = None,
) -> dict[str, Any]:
    """Write a new Physical Study YAML bound to one verified deployment receipt.

    The source Study is never modified. The resulting Study preserves the original
    campaign input by rewriting its reference relative to the new Study location,
    records the verified acceptance/first-run artifact in readiness, and writes a
    matching immutable Study lock.
    """

    source = Path(study).expanduser().resolve()
    receipt_root = Path(deployment_artifact).expanduser().resolve()
    destination = Path(output_study).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"study configuration does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"bound Study output already exists: {destination}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("study configuration must be a mapping")
    original = StudySpec.load(source)
    original_plan = StudyPlanner(original).plan()
    if not bool(original_plan.get("physical", False)):
        raise ValueError("deployment binding requires a Physical Study")
    if original.readiness.mode == "skip":
        raise ValueError(
            "deployment binding cannot be used with readiness.mode=skip; use auto or required"
        )

    receipt, verification = _load_receipt(receipt_root)
    physical_pairs = _physical_input_pairs(original_plan.get("campaign_plan", {}))
    if not physical_pairs:
        raise ValueError("Physical Study plan does not expose a cluster/system checksum pair")
    if len(physical_pairs) != 1:
        raise ValueError(
            "one deployment receipt can only bind a Study with one unique Physical "
            f"cluster/system pair; found {len(physical_pairs)}"
        )
    expected_pair = next(iter(physical_pairs))
    observed_pair = (receipt.cluster_sha256, receipt.system_sha256)
    if observed_pair != expected_pair:
        raise RuntimeError(
            "deployment receipt cluster/system checksums do not match the frozen Study plan"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = (
        destination.with_suffix(".lock.json")
        if lock_output is None
        else Path(lock_output).expanduser().resolve()
    )
    if lock_path.exists():
        raise FileExistsError(f"bound Study lock output already exists: {lock_path}")
    if lock_path == destination:
        raise ValueError("bound Study YAML and lock output must be different paths")

    campaign_path = original.resolve(original.campaign)
    bound = dict(raw)
    bound["campaign"] = _portable_path(campaign_path, destination.parent)
    readiness = dict(bound.get("readiness") or {})
    readiness["acceptance_receipt"] = _portable_path(receipt_root, destination.parent)
    bound["readiness"] = readiness
    bound["lock"] = _portable_path(lock_path, destination.parent)

    study_fd, study_tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    lock_fd, lock_tmp_name = tempfile.mkstemp(
        prefix=f".{lock_path.name}.", suffix=".tmp", dir=str(lock_path.parent)
    )
    os.close(study_fd)
    os.close(lock_fd)
    study_tmp = Path(study_tmp_name)
    lock_tmp = Path(lock_tmp_name)
    try:
        study_tmp.write_text(
            yaml.safe_dump(bound, sort_keys=False),
            encoding="utf-8",
        )
        # The temp Study lives beside the final Study, so all rewritten relative
        # references resolve exactly as they will after atomic promotion.
        temp_spec = StudySpec.load(study_tmp)
        lock_payload = StudyPlanner(temp_spec).lock_payload()
        lock_tmp.write_text(
            json.dumps(lock_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lock_tmp.replace(lock_path)
        study_tmp.replace(destination)
    except Exception:
        study_tmp.unlink(missing_ok=True)
        lock_tmp.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    final_spec = StudySpec.load(destination)
    verified_plan = StudyPlanner(final_spec).verify_lock()
    return {
        "schema": "darpan.study-deployment-binding/v1",
        "study": str(destination),
        "study_sha256": ResultRecorder.sha256(destination),
        "lock": str(lock_path),
        "lock_sha256": ResultRecorder.sha256(lock_path),
        "study_fingerprint": verified_plan["fingerprint"],
        "campaign_plan_fingerprint": verified_plan["campaign_plan_fingerprint"],
        "deployment_fingerprint": receipt.deployment_fingerprint,
        "deployment_artifact_manifest_fingerprint": verification[
            "manifest_fingerprint"
        ],
        "cluster_sha256": receipt.cluster_sha256,
        "system_sha256": receipt.system_sha256,
    }
