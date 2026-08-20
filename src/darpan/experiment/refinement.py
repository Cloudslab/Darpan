"""Evidence-gated Twin refinement decisions for Physical residuals."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifact import require_fresh_artifact_directory, seal_artifact, verify_artifact
from .fidelity_batch import FIDELITY_BATCH_SCHEMA
from .fidelity_diagnosis import DIAGNOSIS_SCHEMA, LEGACY_DIAGNOSIS_SCHEMA
from .recorder import ResultRecorder

REFINEMENT_POLICY_SCHEMA = "darpan.twin-refinement-policy/v1"
REFINEMENT_DECISION_SCHEMA = "darpan.twin-refinement-decision/v1"
_SUPPORTED_DIAGNOSIS_ARTIFACT_SCHEMAS = {
    DIAGNOSIS_SCHEMA,
    LEGACY_DIAGNOSIS_SCHEMA,
    FIDELITY_BATCH_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class RefinementRule:
    metric: str
    target: str
    maximum_acceptable_mae: float
    minimum_samples: int = 1
    minimum_error_contribution: float = 0.0
    required: bool = False


@dataclass(frozen=True, slots=True)
class RefinementPolicy:
    path: Path
    source_sha256: str
    minimum_total_samples: int
    maximum_uncovered_error_contribution: float
    max_candidates: int
    rules: tuple[RefinementRule, ...]

    @classmethod
    def load(cls, path: str | Path) -> RefinementPolicy:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"refinement policy does not exist: {source}")
        source_bytes = source.read_bytes()
        raw = yaml.safe_load(source_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("refinement policy must contain a mapping")
        if raw.get("schema") != REFINEMENT_POLICY_SCHEMA:
            raise ValueError(
                f"refinement policy schema must be {REFINEMENT_POLICY_SCHEMA!r}"
            )
        minimum_total_samples = int(raw.get("minimum_total_samples", 1))
        maximum_uncovered = float(raw.get("maximum_uncovered_error_contribution", 0.25))
        max_candidates = int(raw.get("max_candidates", 2))
        if minimum_total_samples <= 0:
            raise ValueError("minimum_total_samples must be positive")
        if not math.isfinite(maximum_uncovered) or not 0.0 <= maximum_uncovered <= 1.0:
            raise ValueError(
                "maximum_uncovered_error_contribution must be finite and in [0, 1]"
            )
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        raw_rules = raw.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("refinement policy requires at least one metric rule")
        rules: list[RefinementRule] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_rules):
            if not isinstance(item, dict):
                raise ValueError(f"refinement rule #{index + 1} must be a mapping")
            metric = str(item.get("metric", "")).strip()
            target = str(item.get("target", "")).strip()
            if not metric or not target:
                raise ValueError(f"refinement rule #{index + 1} requires metric and target")
            if metric in seen:
                raise ValueError(f"duplicate refinement rule for metric: {metric}")
            seen.add(metric)
            if "maximum_acceptable_mae" not in item:
                raise ValueError(
                    f"refinement rule #{index + 1} requires maximum_acceptable_mae"
                )
            maximum_mae = float(item["maximum_acceptable_mae"])
            minimum_samples = int(item.get("minimum_samples", 1))
            minimum_contribution = float(item.get("minimum_error_contribution", 0.0))
            if not math.isfinite(maximum_mae) or maximum_mae < 0:
                raise ValueError(
                    "maximum_acceptable_mae must be finite and non-negative"
                )
            if minimum_samples <= 0:
                raise ValueError("minimum_samples must be positive")
            if not math.isfinite(minimum_contribution) or not 0.0 <= minimum_contribution <= 1.0:
                raise ValueError(
                    "minimum_error_contribution must be finite and in [0, 1]"
                )
            required = item.get("required", False)
            if not isinstance(required, bool):
                raise ValueError("refinement rule required must be a boolean")
            rules.append(
                RefinementRule(
                    metric=metric,
                    target=target,
                    maximum_acceptable_mae=maximum_mae,
                    minimum_samples=minimum_samples,
                    minimum_error_contribution=minimum_contribution,
                    required=required,
                )
            )
        return cls(
            path=source,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            minimum_total_samples=minimum_total_samples,
            maximum_uncovered_error_contribution=maximum_uncovered,
            max_candidates=max_candidates,
            rules=tuple(rules),
        )


def _load_verified_diagnosis(
    directory: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    source = Path(directory).expanduser().resolve()
    verification = verify_artifact(source)
    schema = verification.get("schema")
    if schema not in _SUPPORTED_DIAGNOSIS_ARTIFACT_SCHEMAS:
        raise ValueError(
            "refinement source must be a sealed fidelity diagnosis or fidelity batch "
            f"artifact, got {schema!r}"
        )

    manifest_path = source / "artifact-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    if manifest_payload.get("manifest_fingerprint") != verification.get(
        "manifest_fingerprint"
    ):
        raise RuntimeError("source artifact manifest changed after verification")
    source_files = {
        str(item["path"]): item
        for item in manifest_payload.get("files", [])
        if isinstance(item, dict) and item.get("path") is not None
    }
    expected_diagnosis = source_files.get("fidelity-diagnosis.json")
    if expected_diagnosis is None:
        raise RuntimeError("source artifact manifest does not seal fidelity-diagnosis.json")

    diagnosis_path = source / "fidelity-diagnosis.json"
    if not diagnosis_path.is_file():
        raise FileNotFoundError(
            f"sealed fidelity artifact does not contain fidelity-diagnosis.json: {source}"
        )
    diagnosis_bytes = diagnosis_path.read_bytes()
    diagnosis_sha256 = hashlib.sha256(diagnosis_bytes).hexdigest()
    if diagnosis_sha256 != expected_diagnosis.get("sha256"):
        raise RuntimeError(
            "source fidelity diagnosis changed after artifact verification"
        )
    diagnosis = json.loads(diagnosis_bytes.decode("utf-8"))
    if not isinstance(diagnosis, dict) or diagnosis.get("schema") not in {
        DIAGNOSIS_SCHEMA,
        LEGACY_DIAGNOSIS_SCHEMA,
    }:
        raise ValueError("fidelity-diagnosis.json has an unsupported schema")
    return (
        source,
        diagnosis,
        verification,
        manifest_payload,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def decide_twin_refinement(
    diagnosis: dict[str, Any],
    policy: RefinementPolicy,
) -> dict[str, Any]:
    """Return a deterministic decision without changing any Twin model."""

    total_samples = int(diagnosis.get("samples", 0))
    ranking = diagnosis.get("metric_ranking")
    if not isinstance(ranking, list):
        raise ValueError("fidelity diagnosis metric_ranking is malformed")
    metrics = {
        str(item["metric_key"]): item
        for item in ranking
        if isinstance(item, dict) and item.get("metric_key") is not None
    }
    evaluations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    insufficient_required: list[str] = []
    for rule in policy.rules:
        observed = metrics.get(rule.metric)
        samples = 0 if observed is None else int(observed.get("samples", 0))
        mae = None if observed is None else float(observed.get("mean_absolute_error", 0.0))
        contribution = (
            0.0 if observed is None else float(observed.get("error_contribution", 0.0))
        )
        normalized_error = (
            None
            if observed is None or observed.get("mean_normalized_error") is None
            else float(observed["mean_normalized_error"])
        )
        sufficient = samples >= rule.minimum_samples
        exceeds_mae = bool(sufficient and mae is not None and mae > rule.maximum_acceptable_mae)
        contribution_ok = contribution >= rule.minimum_error_contribution
        eligible = bool(exceeds_mae and contribution_ok)
        evaluation = {
            **asdict(rule),
            "observed_samples": samples,
            "observed_mae": mae,
            "observed_error_contribution": contribution,
            "observed_mean_normalized_error": normalized_error,
            "evidence_sufficient": sufficient,
            "exceeds_acceptable_mae": exceeds_mae,
            "contribution_sufficient": contribution_ok,
            "eligible_for_refinement": eligible,
        }
        evaluations.append(evaluation)
        if rule.required and not sufficient:
            insufficient_required.append(rule.metric)
        if eligible:
            candidates.append(evaluation)

    candidates.sort(
        key=lambda item: (
            -float(item["observed_error_contribution"]),
            -float(item["observed_mean_normalized_error"] or 0.0),
            str(item["metric"]),
        )
    )
    candidates = candidates[: policy.max_candidates]

    covered = {rule.metric for rule in policy.rules}
    uncovered = [
        item
        for item in ranking
        if str(item.get("metric_key")) not in covered
        and float(item.get("error_contribution", 0.0))
        >= policy.maximum_uncovered_error_contribution
    ]

    reasons: list[str] = []
    if total_samples < policy.minimum_total_samples:
        decision = "insufficient_evidence"
        reasons.append(
            f"total fidelity samples {total_samples} < required {policy.minimum_total_samples}"
        )
    elif insufficient_required:
        decision = "insufficient_evidence"
        reasons.append(
            "required metric evidence is insufficient: " + ", ".join(insufficient_required)
        )
    elif uncovered:
        decision = "manual_review"
        reasons.append(
            "material residual contribution is not covered by the frozen refinement policy: "
            + ", ".join(str(item["metric_key"]) for item in uncovered)
        )
    elif candidates:
        decision = "refine"
        reasons.append(
            "one or more pre-declared metric thresholds are exceeded with sufficient evidence"
        )
    else:
        decision = "hold"
        reasons.append("no pre-declared refinement threshold is exceeded")

    authorized_targets = list(
        dict.fromkeys(str(item["target"]) for item in candidates)
    ) if decision == "refine" else []
    return {
        "schema": REFINEMENT_DECISION_SCHEMA,
        "decision": decision,
        "samples": total_samples,
        "dominant_metric": diagnosis.get("dominant_metric"),
        "authorized_targets": authorized_targets,
        "candidate_metrics": [str(item["metric"]) for item in candidates]
        if decision == "refine"
        else [],
        "reasons": reasons,
        "uncovered_material_metrics": [str(item["metric_key"]) for item in uncovered],
        "metric_evaluations": evaluations,
    }


def export_refinement_decision(
    diagnosis_directory: str | Path,
    policy_path: str | Path | None,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify sealed evidence, evaluate a frozen policy, and seal the decision."""

    (
        source,
        diagnosis,
        verification,
        source_manifest_payload,
        source_manifest_sha256,
    ) = _load_verified_diagnosis(diagnosis_directory)
    source_files = {
        str(item["path"]): item
        for item in source_manifest_payload.get("files", [])
        if isinstance(item, dict) and item.get("path") is not None
    }
    bundled_policy = source / "inputs/refinement-policy.yaml"
    policy_predeclared = bundled_policy.is_file()
    if policy_path is None:
        if not policy_predeclared:
            raise ValueError(
                "no refinement policy was predeclared with this evidence; provide --policy "
                "for exploratory use or bind refinement_policy in the fidelity-batch manifest"
            )
        resolved_policy = bundled_policy
    else:
        resolved_policy = Path(policy_path).expanduser().resolve()
    policy = RefinementPolicy.load(resolved_policy)
    if policy_predeclared:
        expected_policy = source_files.get("inputs/refinement-policy.yaml")
        if expected_policy is None:
            raise RuntimeError(
                "source artifact manifest does not seal its predeclared refinement policy"
            )
        if policy.source_sha256 != expected_policy.get("sha256"):
            raise RuntimeError(
                "requested refinement policy differs from the policy sealed before "
                "fidelity evidence was evaluated"
            )
        resolved_policy = bundled_policy
        policy = RefinementPolicy.load(resolved_policy)
        if policy.source_sha256 != expected_policy.get("sha256"):
            raise RuntimeError(
                "predeclared refinement policy changed after artifact verification"
            )
    decision = decide_twin_refinement(diagnosis, policy)

    source_manifest = source / "artifact-manifest.json"
    source_diagnosis = source / "fidelity-diagnosis.json"
    expected_diagnosis = source_files["fidelity-diagnosis.json"]

    target = require_fresh_artifact_directory(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rmdir()
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    )
    try:
        recorder = ResultRecorder(stage)
        copied_policy = recorder.copy(policy.path, "inputs/refinement-policy.yaml")
        if recorder.sha256(copied_policy) != policy.source_sha256:
            raise RuntimeError("refinement policy changed while the decision was evaluated")
        copied_manifest = recorder.copy(
            source_manifest, "inputs/source-artifact-manifest.json"
        )
        if recorder.sha256(copied_manifest) != source_manifest_sha256:
            raise RuntimeError("source artifact manifest changed after verification")
        copied_diagnosis = recorder.copy(
            source_diagnosis, "inputs/source-fidelity-diagnosis.json"
        )
        if recorder.sha256(copied_diagnosis) != expected_diagnosis.get("sha256"):
            raise RuntimeError("source fidelity diagnosis changed after artifact verification")
        payload = {
            **decision,
            "policy": {
                "schema": REFINEMENT_POLICY_SCHEMA,
                "minimum_total_samples": policy.minimum_total_samples,
                "maximum_uncovered_error_contribution": (
                    policy.maximum_uncovered_error_contribution
                ),
                "max_candidates": policy.max_candidates,
                "rules": [asdict(rule) for rule in policy.rules],
                "sha256": policy.source_sha256,
            },
            "policy_predeclared_with_evidence": policy_predeclared,
            "source": {
                "schema": verification["schema"],
                "manifest_fingerprint": verification["manifest_fingerprint"],
                "content_fingerprint": verification["content_fingerprint"],
                "diagnosis_sha256": expected_diagnosis["sha256"],
            },
        }
        recorder.write_json("twin-refinement-decision.json", payload)
        manifest = seal_artifact(
            stage,
            schema=REFINEMENT_DECISION_SCHEMA,
            identity={
                "decision": decision["decision"],
                "samples": decision["samples"],
                "authorized_targets": decision["authorized_targets"],
                "source_manifest_fingerprint": verification["manifest_fingerprint"],
            },
        )
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        **payload,
        "artifact_manifest_fingerprint": manifest["manifest_fingerprint"],
        "output": str(target),
    }
