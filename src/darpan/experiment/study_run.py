"""Frozen study orchestration from readiness through sealed campaign evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from darpan._version import __version__
from darpan.runtime.real.cluster.acceptance import verify_deployment_receipt_payload

from .artifact import (
    ARTIFACT_MANIFEST,
    require_fresh_artifact_directory,
    restore_artifact_checkpoint,
    seal_artifact,
    verify_artifact,
)
from .campaign import CampaignPlanner, CampaignRunner, CampaignSpec
from .recorder import ResultRecorder
from .study import validate_physical_study

STUDY_PLAN_SCHEMA = "darpan.study-plan/v1"
STUDY_RESULT_SCHEMA = "darpan.study-result/v1"
STUDY_ARTIFACT_SCHEMA = "darpan.study-artifact/v1"

ExperimentExecutor = Callable[[Any], Awaitable[dict[str, Any]]]


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _contains_physical(value: Any) -> bool:
    if isinstance(value, Mapping):
        if isinstance(value.get("physical"), Mapping):
            return True
        if isinstance(value.get("physical_gate"), Mapping):
            return True
        return any(_contains_physical(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_physical(item) for item in value)
    return False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class StudyReadinessSpec:
    mode: str = "auto"
    exercise_data_plane: bool = False
    max_clock_offset_s: float = 1.0
    acceptance_receipt: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "required", "skip"}:
            raise ValueError("study readiness mode must be auto, required, or skip")
        if self.max_clock_offset_s < 0:
            raise ValueError("study max_clock_offset_s cannot be negative")


@dataclass(frozen=True, slots=True)
class StudySpec:
    name: str
    version: str
    campaign: str
    output: str
    readiness: StudyReadinessSpec = field(default_factory=StudyReadinessSpec)
    require_all_jobs_success: bool = True
    research_questions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lock: str | None = None
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("study name cannot be empty")
        if not self.version:
            raise ValueError("study version cannot be empty")
        if not self.campaign:
            raise ValueError("study campaign cannot be empty")
        if not self.output:
            raise ValueError("study output cannot be empty")

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def resolve(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.directory / raw).resolve()

    @classmethod
    def load(cls, path: str | Path) -> StudySpec:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("study configuration must be a mapping")
        readiness_raw = data.get("readiness", {}) or {}
        if not isinstance(readiness_raw, dict):
            raise ValueError("study readiness must be a mapping")
        return cls(
            name=str(data.get("name", source.stem)),
            version=str(data.get("version", "1")),
            campaign=str(data["campaign"]),
            output=str(data.get("output", f"{source.stem}-study")),
            readiness=StudyReadinessSpec(
                mode=str(readiness_raw.get("mode", "auto")),
                exercise_data_plane=bool(
                    readiness_raw.get("exercise_data_plane", False)
                ),
                max_clock_offset_s=float(
                    readiness_raw.get("max_clock_offset_s", 1.0)
                ),
                acceptance_receipt=(
                    None
                    if readiness_raw.get("acceptance_receipt") is None
                    else str(readiness_raw["acceptance_receipt"])
                ),
            ),
            require_all_jobs_success=bool(data.get("require_all_jobs_success", True)),
            research_questions=_research_questions(data.get("research_questions", {})),
            lock=None if data.get("lock") is None else str(data["lock"]),
            source=source,
        )


def _research_questions(value: Any) -> dict[str, tuple[str, ...]]:
    if value is None or value == "":
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("study research_questions must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for key, raw in value.items():
        if isinstance(raw, str):
            categories = (raw,)
        elif isinstance(raw, (list, tuple)):
            categories = tuple(str(item) for item in raw)
        else:
            raise ValueError(f"research question {key!r} evidence must be a string or list")
        if not categories or any(not item for item in categories):
            raise ValueError(f"research question {key!r} requires non-empty evidence categories")
        result[str(key)] = categories
    return result


class StudyPlanner:
    def __init__(self, spec: StudySpec) -> None:
        self.spec = spec
        self.campaign_path = spec.resolve(spec.campaign)
        self.campaign_spec = CampaignSpec.load(self.campaign_path)
        self.campaign_plan = CampaignPlanner(self.campaign_spec).plan()

    def _readiness_identity(self) -> dict[str, Any]:
        readiness = asdict(self.spec.readiness)
        receipt_ref = self.spec.readiness.acceptance_receipt
        if receipt_ref is None:
            readiness.pop("acceptance_receipt", None)
            return readiness
        receipt_root = self.spec.resolve(receipt_ref)
        verified = verify_artifact(receipt_root)
        if verified.get("schema") not in {
            "darpan.cluster-acceptance/v1",
            "darpan.cluster-first-run/v1",
        }:
            raise ValueError(
                "deployment receipt must come from a sealed cluster acceptance/first-run artifact"
            )
        receipt = verify_deployment_receipt_payload(
            _read_json(receipt_root / "deployment-receipt.json")
        )
        if receipt.darpan_version != __version__:
            raise RuntimeError(
                "deployment receipt Darpan version does not match the current Study release"
            )
        if verified.get("identity", {}).get(
            "deployment_fingerprint"
        ) != receipt.deployment_fingerprint:
            raise RuntimeError(
                "acceptance artifact identity does not match deployment receipt"
            )
        readiness["acceptance_receipt"] = {
            "reference": receipt_ref,
            "artifact_manifest_fingerprint": verified["manifest_fingerprint"],
            "deployment_fingerprint": receipt.deployment_fingerprint,
        }
        return readiness

    def plan(self) -> dict[str, Any]:
        physical = _contains_physical(self.campaign_plan.get("jobs", ()))
        basis = {
            "schema": STUDY_PLAN_SCHEMA,
            "name": self.spec.name,
            "version": self.spec.version,
            "campaign": self.spec.campaign,
            "campaign_sha256": ResultRecorder.sha256(self.campaign_path),
            "campaign_plan_fingerprint": self.campaign_plan["fingerprint"],
            "physical": physical,
            "readiness": self._readiness_identity(),
            "require_all_jobs_success": self.spec.require_all_jobs_success,
            "research_questions": {
                key: list(value) for key, value in sorted(self.spec.research_questions.items())
            },
        }
        coverage = self.campaign_plan.get("coverage", {})
        categories = coverage.get("categories", {}) if isinstance(coverage, Mapping) else {}
        missing_rq = {
            question: [item for item in required if item not in categories]
            for question, required in self.spec.research_questions.items()
        }
        missing_rq = {key: value for key, value in missing_rq.items() if value}
        if missing_rq:
            details = "; ".join(
                f"{key}: {', '.join(value)}" for key, value in sorted(missing_rq.items())
            )
            raise ValueError(f"study research question evidence is not covered: {details}")
        basis["research_question_jobs"] = {
            question: {
                category: list(categories.get(category, ()))
                for category in required
            }
            for question, required in sorted(self.spec.research_questions.items())
        }
        payload = dict(basis)
        payload["campaign_path"] = str(self.campaign_path)
        payload["campaign_plan"] = self.campaign_plan
        payload["fingerprint"] = _canonical_hash(basis)
        return payload

    def lock_payload(self) -> dict[str, Any]:
        return self.plan()

    def verify_lock(self) -> dict[str, Any]:
        payload = self.plan()
        if self.spec.lock is None:
            return payload
        path = self.spec.resolve(self.spec.lock)
        observed = _read_json(path)
        expected = observed.get("fingerprint")
        if expected != payload["fingerprint"]:
            raise RuntimeError(
                "study lock fingerprint mismatch: "
                f"expected {expected!r}, observed {payload['fingerprint']!r}"
            )
        return payload


class StudyRunner(StudyPlanner):
    def __init__(
        self,
        spec: StudySpec,
        *,
        execute: ExperimentExecutor,
        resume: bool = False,
    ) -> None:
        super().__init__(spec)
        self.execute = execute
        self.resume = resume
        self.plan_payload = self.verify_lock()
        output = spec.resolve(spec.output)
        if resume:
            if not output.is_dir():
                raise FileNotFoundError(f"study resume output does not exist: {output}")
            self._prepare_resume(output)
        else:
            output = require_fresh_artifact_directory(output)
        self.recorder = ResultRecorder(output)

    def _prepare_resume(self, output: Path) -> None:
        study_plan = output / "study-plan.json"
        if not study_plan.is_file():
            raise FileNotFoundError(f"study resume plan does not exist: {study_plan}")
        observed = _read_json(study_plan)
        if observed.get("fingerprint") != self.plan_payload["fingerprint"]:
            raise RuntimeError("study resume fingerprint does not match current frozen plan")
        manifest = output / ARTIFACT_MANIFEST
        if manifest.is_file():
            report = verify_artifact(output)
            if bool(report.get("identity", {}).get("complete", False)):
                raise RuntimeError("completed study artifacts cannot be resumed")
            manifest.unlink()

    def _state(self, stage: str, status: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "schema": "darpan.study-state/v1",
            "study_fingerprint": self.plan_payload["fingerprint"],
            "stage": stage,
            "status": status,
        }
        payload.update(extra)
        self.recorder.write_json("study-state.json", payload)
        return payload

    def _remove(self, name: str) -> None:
        path = self.recorder.directory / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _verified_nested(self, name: str) -> dict[str, Any] | None:
        path = self.recorder.directory / name
        if not path.is_dir():
            return None
        try:
            return verify_artifact(path)
        except (FileNotFoundError, RuntimeError, ValueError):
            return None

    def _restore_campaign_checkpoint(self) -> None:
        campaign = self.recorder.directory / "campaign"
        if not campaign.is_dir():
            return
        try:
            verify_artifact(campaign)
        except RuntimeError:
            restore_artifact_checkpoint(campaign)

    async def _readiness(self) -> dict[str, Any] | None:
        mode = self.spec.readiness.mode
        physical = bool(self.plan_payload["physical"])
        required = mode == "required" or (mode == "auto" and physical)
        if not required:
            return None
        self._remove("readiness")
        report = await validate_physical_study(
            self.campaign_path,
            exercise_data_plane=self.spec.readiness.exercise_data_plane,
            max_clock_offset_s=self.spec.readiness.max_clock_offset_s,
            acceptance_receipt=(
                None
                if self.spec.readiness.acceptance_receipt is None
                else self.spec.resolve(self.spec.readiness.acceptance_receipt)
            ),
            output=self.recorder.directory / "readiness",
        )
        if not report.ready:
            raise RuntimeError("study Physical readiness failed: " + "; ".join(report.errors))
        return report.to_dict()

    async def _campaign(self) -> dict[str, Any]:
        root = self.recorder.directory / "campaign"
        existing = self._verified_nested("campaign")
        if existing is not None and bool(existing.get("identity", {}).get("complete", False)):
            payload = _read_json(root / "campaign.json")
            if bool(payload.get("complete", False)):
                return payload
        campaign_spec = replace(self.campaign_spec, output=str(root))
        payload = await CampaignRunner(
            campaign_spec,
            execute=self.execute,
            resume=root.is_dir() and any(root.iterdir()),
        ).run()
        if self.spec.require_all_jobs_success and int(payload.get("failed_jobs", 0)):
            raise RuntimeError(
                f"study campaign completed with {payload.get('failed_jobs')} failed job(s)"
            )
        return payload

    async def run(self) -> dict[str, Any]:
        if self.spec.source is not None:
            self.recorder.copy(self.spec.source, "inputs/study.yaml")
        self.recorder.copy(self.campaign_path, "inputs/campaign.yaml")
        if self.spec.lock is not None:
            self.recorder.copy(self.spec.resolve(self.spec.lock), "inputs/study.lock.json")
        self.recorder.write_json("study-plan.json", self.plan_payload)
        self._state("planning", "completed")
        artifacts: dict[str, Any] = {}
        try:
            campaign_complete = self._verified_nested("campaign")
            if campaign_complete is None or not bool(
                campaign_complete.get("identity", {}).get("complete", False)
            ):
                self._state("readiness", "running")
                readiness = await self._readiness()
                if readiness is not None:
                    artifacts["readiness"] = verify_artifact(
                        self.recorder.directory / "readiness"
                    )
                self._state("readiness", "completed")
            elif (self.recorder.directory / "readiness").is_dir():
                artifacts["readiness"] = verify_artifact(
                    self.recorder.directory / "readiness"
                )

            self._state("campaign", "running")
            campaign = await self._campaign()
            artifacts["campaign"] = verify_artifact(self.recorder.directory / "campaign")
            self._state("campaign", "completed")

            nested = {
                name: {
                    "schema": report.get("schema"),
                    "manifest_fingerprint": report.get("manifest_fingerprint"),
                    "content_fingerprint": report.get("content_fingerprint"),
                }
                for name, report in artifacts.items()
            }
            payload = {
                "schema": STUDY_RESULT_SCHEMA,
                "name": self.spec.name,
                "version": self.spec.version,
                "complete": True,
                "study_fingerprint": self.plan_payload["fingerprint"],
                "campaign_plan_fingerprint": self.plan_payload[
                    "campaign_plan_fingerprint"
                ],
                "physical": self.plan_payload["physical"],
                "campaign": {
                    "successful_jobs": campaign.get("successful_jobs"),
                    "failed_jobs": campaign.get("failed_jobs"),
                    "total_jobs": campaign.get("total_jobs"),
                },
                "artifacts": nested,
            }
            self.recorder.write_json("study.json", payload)
            self._state("complete", "completed")
            seal_artifact(
                self.recorder.directory,
                schema=STUDY_ARTIFACT_SCHEMA,
                identity={
                    "study_fingerprint": self.plan_payload["fingerprint"],
                    "complete": True,
                },
            )
            return payload
        except BaseException as exc:
            self._restore_campaign_checkpoint()
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self._state("failed", status, error=f"{type(exc).__name__}: {exc}")
            seal_artifact(
                self.recorder.directory,
                schema=STUDY_ARTIFACT_SCHEMA,
                identity={
                    "study_fingerprint": self.plan_payload["fingerprint"],
                    "complete": False,
                },
            )
            raise


def _seed_completeness_errors(
    plan: Mapping[str, Any], campaign: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    planned_jobs = plan.get("campaign_plan", {}).get("jobs", ())
    actual_jobs = campaign.get("jobs", {})
    if not isinstance(planned_jobs, list) or not isinstance(actual_jobs, Mapping):
        return ["study campaign plan/job structure is malformed"]
    for planned in planned_jobs:
        if not isinstance(planned, Mapping) or not isinstance(planned.get("seeds"), list):
            continue
        job_id = str(planned.get("id", ""))
        expected = [int(item) for item in planned["seeds"]]
        wrapper = actual_jobs.get(job_id, {})
        if not isinstance(wrapper, Mapping):
            errors.append(f"study job {job_id!r} is missing")
            continue
        result = wrapper.get("result", {})
        if not isinstance(result, Mapping):
            errors.append(f"study job {job_id!r} has no result payload")
            continue
        kind = str(planned.get("kind", ""))
        if kind == "paired":
            samples = result.get("samples", ())
            observed = [
                int(item["seed"])
                for item in samples
                if isinstance(item, Mapping) and "seed" in item
            ]
        elif kind in {"scenario", "learning"}:
            observed = [int(item) for item in result.get("seeds", ())]
        else:
            continue
        if observed != expected:
            errors.append(
                f"study job {job_id!r} seed mismatch: expected={expected}, observed={observed}"
            )
    return errors


def _physical_restoration_errors(root: Path) -> list[str]:
    errors = []
    for path in sorted((root / "campaign" / "runs").rglob("physical-control.json")):
        payload = _read_json(path)
        if payload.get("restore_verified") is not True:
            errors.append(
                "Physical restoration is not verified: "
                + path.relative_to(root).as_posix()
            )
    return errors


def verify_study(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    top = verify_artifact(root)
    if top.get("schema") != STUDY_ARTIFACT_SCHEMA:
        raise ValueError("study verify requires a Darpan study artifact")
    result = _read_json(root / "study.json")
    if not bool(result.get("complete", False)):
        raise RuntimeError("study result is incomplete")
    plan = _read_json(root / "study-plan.json")
    fingerprint = str(result.get("study_fingerprint", ""))
    if plan.get("fingerprint") != fingerprint:
        raise RuntimeError("study result and plan fingerprint do not match")
    if top.get("identity", {}).get("study_fingerprint") != fingerprint:
        raise RuntimeError("study manifest and result fingerprint do not match")

    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("study result artifact map is malformed")
    verified: dict[str, Any] = {}
    names = {
        "readiness": "readiness",
        "campaign": "campaign",
    }
    for key, descriptor in artifacts.items():
        if key not in names or not isinstance(descriptor, Mapping):
            continue
        report = verify_artifact(root / names[key])
        expected = descriptor.get("manifest_fingerprint")
        if report.get("manifest_fingerprint") != expected:
            raise RuntimeError(f"study nested artifact fingerprint mismatch: {key}")
        verified[key] = report

    campaign = _read_json(root / "campaign" / "campaign.json")
    if int(campaign.get("failed_jobs", 0)):
        raise RuntimeError("study contains failed campaign jobs")
    completeness_errors = _seed_completeness_errors(plan, campaign)
    completeness_errors.extend(_physical_restoration_errors(root))
    if completeness_errors:
        raise RuntimeError(
            "study completeness verification failed: " + "; ".join(completeness_errors)
        )
    if bool(plan.get("physical", False)) and "readiness" not in verified:
        raise RuntimeError("Physical study does not contain readiness evidence")
    return {
        "verified": True,
        "schema": STUDY_ARTIFACT_SCHEMA,
        "study_fingerprint": fingerprint,
        "manifest_fingerprint": top.get("manifest_fingerprint"),
        "campaign_plan_fingerprint": result.get("campaign_plan_fingerprint"),
        "seed_completeness_verified": True,
        "physical_restoration_verified": True,
        "nested_artifacts": {
            key: value.get("manifest_fingerprint") for key, value in verified.items()
        },
    }
