"""Paper-campaign orchestration over durable Darpan experiment evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from darpan.core.codec import load_application, load_system, load_workload
from darpan.core.loading import materialize_plugin
from darpan.runtime.real.cluster.acceptance import (
    accept_cluster,
    build_deployment_receipt,
)
from darpan.runtime.real.cluster.exercise import exercise_cluster_runtime
from darpan.runtime.real.cluster.inventory import ClusterInventory
from darpan.runtime.real.cluster.session import validate_inventory_system_mapping
from darpan.runtime.real.cluster.validation import validate_cluster

from .artifact import (
    require_fresh_artifact_directory,
    restore_artifact_checkpoint,
    seal_artifact,
    verify_artifact,
)
from .benchmark import (
    FidelitySample,
    PairedSample,
    summarize_calibration_progress,
    summarize_fidelity,
    summarize_paired,
)
from .benchmark_spec import PairedExperimentBenchmarkSpec
from .capabilities import (
    ProactiveSample,
    assess_explainability,
    assess_proactive_benchmark,
    assess_robustness,
)
from .learning import (
    compare_learning_runs,
    load_learning_trace,
    summarize_learning_run,
    validate_learning_budget,
)
from .paired_runner import (
    ensure_successful,
    experiment_success,
    metric_value,
    run_paired_benchmark,
)
from .provenance import collect_provenance
from .recorder import ResultRecorder
from .reproducibility import resolved_inputs
from .spec import ExperimentSpec
from .suite import PaperSuite
from .trace_fidelity import compare_event_traces, load_event_trace, resolve_event_trace

SUPPORTED_CAMPAIGN_KINDS = frozenset(
    {
        "paired",
        "fidelity",
        "trustworthy",
        "adaptive",
        "proactive",
        "explainable",
        "robust",
        "explorable",
        "scenario",
        "learning",
        "cluster_preflight",
        "cluster_exercise",
        "cluster_acceptance",
    }
)


def _declared_metric_names(spec: ExperimentSpec) -> tuple[str, ...]:
    names: list[str] = []
    for reference in spec.metrics:
        if ":" not in reference:
            names.append(reference)
            continue
        metric = materialize_plugin(reference, search_path=spec.directory)
        name = getattr(metric, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"custom metric {reference!r} must declare a non-empty string name"
            )
        names.append(name)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class CampaignJob:
    id: str
    kind: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("campaign job id cannot be empty")
        if self.kind not in SUPPORTED_CAMPAIGN_KINDS:
            allowed = ", ".join(sorted(SUPPORTED_CAMPAIGN_KINDS))
            raise ValueError(f"unsupported campaign job kind {self.kind!r}; choose {allowed}")


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    name: str
    jobs: tuple[CampaignJob, ...]
    output: str
    seed: int = 0
    repeat: int = 10
    bootstrap_resamples: int = 2000
    continue_on_error: bool = False
    suite: str | None = None
    suite_lock: str | None = None
    plan_lock: str | None = None
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign name cannot be empty")
        if not self.jobs:
            raise ValueError("campaign requires at least one job")
        ids = tuple(job.id for job in self.jobs)
        if len(set(ids)) != len(ids):
            raise ValueError("campaign job ids must be unique")
        if not self.output:
            raise ValueError("campaign output cannot be empty")
        if self.repeat <= 0:
            raise ValueError("campaign repeat must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("campaign bootstrap_resamples must be positive")
        if self.suite_lock is not None and self.suite is None:
            raise ValueError("campaign suite_lock requires suite")

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def resolve(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.directory / raw).resolve()

    @classmethod
    def load(cls, path: str | Path) -> CampaignSpec:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("campaign configuration must be a mapping")
        jobs_raw = data.get("jobs", ())
        if not isinstance(jobs_raw, list):
            raise ValueError("campaign jobs must be a list")
        jobs = []
        for raw in jobs_raw:
            if not isinstance(raw, dict):
                raise ValueError("each campaign job must be a mapping")
            config = dict(raw)
            job_id = str(config.pop("id"))
            kind = str(config.pop("kind"))
            jobs.append(CampaignJob(job_id, kind, config))
        return cls(
            name=str(data.get("name", source.stem)),
            jobs=tuple(jobs),
            output=str(data.get("output", f"{source.stem}-results")),
            seed=int(data.get("seed", 0)),
            repeat=int(data.get("repeat", 10)),
            bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
            continue_on_error=bool(data.get("continue_on_error", False)),
            suite=None if data.get("suite") is None else str(data["suite"]),
            suite_lock=(
                None if data.get("suite_lock") is None else str(data["suite_lock"])
            ),
            plan_lock=(
                None if data.get("plan_lock") is None else str(data["plan_lock"])
            ),
            source=source,
        )


ExperimentExecutor = Callable[[ExperimentSpec], Awaitable[dict[str, Any]]]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _portable_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _required(config: Mapping[str, Any], key: str, job: CampaignJob) -> Any:
    if key not in config:
        raise ValueError(f"campaign job {job.id!r} requires {key!r}")
    return config[key]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                records.append(raw)
        return records
    raw = _read_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
        raw = raw["samples"]
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{path} must contain a list of JSON objects")
    return list(raw)


def _fidelity_layer(path: Path, layer: str) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"fidelity artifact must be a JSON object: {path}")
    payload = raw.get(layer)
    if not isinstance(payload, dict):
        raise KeyError(f"fidelity artifact does not contain layer {layer!r}: {path}")
    return payload


def _recovery_evidence(output: Mapping[str, Any]) -> dict[str, Any]:
    status = output.get("_experiment", {})
    if not isinstance(status, Mapping):
        status = {}
    retried = tuple(str(item) for item in status.get("retried_components", ()))
    recovered = tuple(str(item) for item in status.get("recovered_components", ()))
    exhausted = tuple(
        str(item) for item in status.get("retry_exhausted_components", ())
    )
    return {
        "retry_attempts": int(status.get("retry_attempts", 0)),
        "retried_components": len(retried),
        "recovered_components": len(recovered),
        "exhausted_components": len(exhausted),
        "recovery_rate": status.get("retry_recovery_rate"),
        "retry_downtime_s": status.get("retry_downtime_s"),
    }


def _summarize_recovery(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "runs": 0,
            "total_retry_attempts": 0,
            "mean_retry_attempts": 0.0,
            "retried_components": 0,
            "recovered_components": 0,
            "exhausted_components": 0,
            "component_recovery_rate": None,
            "mean_retry_downtime_s": None,
        }
    attempts = sum(int(item.get("retry_attempts", 0)) for item in items)
    retried = sum(int(item.get("retried_components", 0)) for item in items)
    recovered = sum(int(item.get("recovered_components", 0)) for item in items)
    exhausted = sum(int(item.get("exhausted_components", 0)) for item in items)
    downtimes = [
        float(item["retry_downtime_s"])
        for item in items
        if item.get("retry_downtime_s") is not None
    ]
    return {
        "runs": len(items),
        "total_retry_attempts": attempts,
        "mean_retry_attempts": attempts / len(items),
        "retried_components": retried,
        "recovered_components": recovered,
        "exhausted_components": exhausted,
        "component_recovery_rate": None if not retried else recovered / retried,
        "mean_retry_downtime_s": (
            None if not downtimes else sum(downtimes) / len(downtimes)
        ),
    }


def _fidelity_samples(payload: Mapping[str, Any]) -> tuple[FidelitySample, ...]:
    samples = payload.get("samples", ())
    if not isinstance(samples, list):
        raise ValueError("fidelity layer samples must be a list")
    return tuple(
        FidelitySample(
            predicted=float(item["predicted"]),
            observed=float(item["observed"]),
            lower=None if item.get("lower") is None else float(item["lower"]),
            upper=None if item.get("upper") is None else float(item["upper"]),
            uncertainty=(
                None if item.get("uncertainty") is None else float(item["uncertainty"])
            ),
            metadata=dict(item.get("metadata", {})),
        )
        for item in samples
    )


class CampaignPlanner:
    """Resolve and fingerprint a campaign without executing any workload."""

    def __init__(self, spec: CampaignSpec) -> None:
        self.spec = spec
        self.suite = (
            None if spec.suite is None else PaperSuite.load(spec.resolve(spec.suite))
        )
        self.suite_lock_payload: dict[str, Any] | None = None
        if self.suite is not None:
            if spec.suite_lock is None:
                self.suite_lock_payload = self.suite.lock_payload()
            else:
                lock_path = spec.resolve(spec.suite_lock)
                if not lock_path.is_file():
                    raise FileNotFoundError(
                        f"campaign suite lock does not exist: {lock_path}"
                    )
                self.suite_lock_payload = self.suite.verify_lock(lock_path)

    def _resolve_resource(
        self,
        value: str,
        *,
        kinds: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    ) -> Path:
        if value.startswith("@") or value.startswith("suite:"):
            if self.suite is None:
                raise ValueError(
                    f"campaign resource {value!r} requires a top-level suite"
                )
            return self.suite.resolve_reference(value, kinds=kinds)
        return self.spec.resolve(value)

    def _file_descriptor(self, reference: str, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"campaign input does not exist: {path}")
        return {
            "reference": reference,
            "path": _portable_path(path, self.spec.directory),
            "sha256": ResultRecorder.sha256(path),
        }

    def _experiment_descriptor(self, reference: str) -> dict[str, Any]:
        path = self._resolve_resource(reference, kinds={"experiment"})
        spec = ExperimentSpec.load(path)
        physical = None
        if spec.runtime == "real" and spec.cluster is not None:
            system_path = spec.resolve(spec.system)
            cluster_path = spec.resolve(spec.cluster)
            system = load_system(system_path)
            inventory = ClusterInventory.load(cluster_path)
            validate_inventory_system_mapping(inventory, system)
            if spec.application is not None:
                application = load_application(spec.resolve(spec.application))
                requires_docker = any(
                    component.image is not None for component in application.components
                )
            else:
                workload = load_workload(spec.resolve(spec.workload))
                requires_docker = any(
                    component.image is not None
                    for application in workload.applications
                    for component in application.components
                )
            physical = {
                "cluster": self._file_descriptor(
                    _portable_path(cluster_path, path.parent), cluster_path
                ),
                "system": self._file_descriptor(
                    _portable_path(system_path, path.parent), system_path
                ),
                "nodes": [node.id for node in system.nodes],
                "physical_control": spec.physical_control,
                "network_driver": spec.network_driver,
                "requires_docker": requires_docker,
            }
        descriptor = {
            "reference": reference,
            "runtime": spec.runtime,
            "policy": spec.policy,
            "metrics": list(_declared_metric_names(spec)),
            "inputs": [
                self._file_descriptor(
                    _portable_path(item, path.parent),
                    item,
                )
                for item in resolved_inputs(spec)
            ],
        }
        if physical is not None:
            descriptor["physical"] = physical
        return descriptor

    def _seeds(self, config: Mapping[str, Any]) -> tuple[int, ...]:
        seeds = tuple(int(item) for item in config.get("seeds", ()))
        if seeds:
            if len(set(seeds)) != len(seeds):
                raise ValueError("campaign plan seeds must be unique")
            return seeds
        repeat = int(config.get("repeat", self.spec.repeat))
        seed = int(config.get("seed", self.spec.seed))
        if repeat <= 0:
            raise ValueError("campaign plan repeat must be positive")
        return tuple(seed + index for index in range(repeat))

    def _cluster_descriptor(
        self, cluster_ref: str, system_ref: str | None
    ) -> dict[str, Any]:
        cluster_path = self._resolve_resource(cluster_ref, kinds={"cluster"})
        inventory = ClusterInventory.load(cluster_path)
        descriptor: dict[str, Any] = {
            "cluster": self._file_descriptor(cluster_ref, cluster_path),
            "nodes": [node.id for node in inventory.nodes],
        }
        if system_ref is not None:
            system_path = self._resolve_resource(system_ref, kinds={"system"})
            system = load_system(system_path)
            validate_inventory_system_mapping(inventory, system)
            descriptor["system"] = self._file_descriptor(system_ref, system_path)
            descriptor["system_nodes"] = [node.id for node in system.nodes]
            descriptor["links"] = [link.id for link in system.links]
        return descriptor

    def _plan_job(self, job: CampaignJob) -> dict[str, Any]:
        config = job.config
        result: dict[str, Any] = {
            "id": job.id,
            "kind": job.kind,
            "config": dict(config),
        }
        if job.kind == "paired":
            metric = str(_required(config, "metric", job))
            baseline_ref = str(_required(config, "baseline", job))
            candidate_ref = str(_required(config, "candidate", job))
            baseline = self._experiment_descriptor(baseline_ref)
            candidate = self._experiment_descriptor(candidate_ref)
            for label, experiment in (("baseline", baseline), ("candidate", candidate)):
                if metric not in experiment["metrics"]:
                    raise ValueError(
                        f"campaign job {job.id!r} metric {metric!r} is not declared "
                        f"by {label} experiment"
                    )
            seeds = self._seeds(config)
            result.update(
                metric=metric,
                seeds=list(seeds),
                execution_runs=2 * len(seeds),
                baseline=baseline,
                candidate=candidate,
            )
            return result
        if job.kind == "scenario":
            metric = str(_required(config, "metric", job))
            baseline_ref = str(_required(config, "baseline", job))
            scenarios_raw = _required(config, "scenarios", job)
            if not isinstance(scenarios_raw, Mapping) or not scenarios_raw:
                raise ValueError(f"campaign job {job.id!r} scenarios must be a mapping")
            baseline = self._experiment_descriptor(baseline_ref)
            if metric not in baseline["metrics"]:
                raise ValueError(
                    f"campaign job {job.id!r} metric {metric!r} is not declared "
                    "by baseline experiment"
                )
            scenarios = {}
            for name, reference in scenarios_raw.items():
                descriptor = self._experiment_descriptor(str(reference))
                if metric not in descriptor["metrics"]:
                    raise ValueError(
                        f"campaign job {job.id!r} metric {metric!r} is not declared "
                        f"by scenario {name!r}"
                    )
                scenarios[str(name)] = descriptor
            seeds = self._seeds(config)
            result.update(
                metric=metric,
                seeds=list(seeds),
                execution_runs=(1 + len(scenarios)) * len(seeds),
                baseline=baseline,
                scenarios=scenarios,
            )
            return result
        if job.kind == "learning":
            baseline_raw = _required(config, "baseline", job)
            candidate_raw = _required(config, "candidate", job)
            if not isinstance(baseline_raw, Mapping) or not isinstance(candidate_raw, Mapping):
                raise ValueError(
                    f"campaign job {job.id!r} baseline/candidate must map seed to trace"
                )
            baseline_refs = {
                int(seed): str(reference) for seed, reference in baseline_raw.items()
            }
            candidate_refs = {
                int(seed): str(reference) for seed, reference in candidate_raw.items()
            }
            baseline_paths = {
                seed: self._resolve_resource(reference, kinds={"evidence"})
                for seed, reference in baseline_refs.items()
            }
            candidate_paths = {
                seed: self._resolve_resource(reference, kinds={"evidence"})
                for seed, reference in candidate_refs.items()
            }
            if set(baseline_paths) != set(candidate_paths):
                raise ValueError(
                    f"campaign job {job.id!r} requires identical baseline/candidate seeds"
                )
            for path in (*baseline_paths.values(), *candidate_paths.values()):
                load_learning_trace(path)
            baseline = {
                seed: self._file_descriptor(baseline_refs[seed], path)
                for seed, path in baseline_paths.items()
            }
            candidate = {
                seed: self._file_descriptor(candidate_refs[seed], path)
                for seed, path in candidate_paths.items()
            }
            result.update(
                seeds=sorted(baseline),
                execution_runs=0,
                baseline={str(key): value for key, value in baseline.items()},
                candidate={str(key): value for key, value in candidate.items()},
            )
            return result
        if job.kind == "cluster_preflight":
            cluster_ref = str(_required(config, "cluster", job))
            system_ref = None if config.get("system") is None else str(config["system"])
            result.update(
                execution_runs=0,
                physical_gate=self._cluster_descriptor(cluster_ref, system_ref),
                exercise_data_plane=bool(config.get("exercise_data_plane", False)),
            )
            return result
        if job.kind == "cluster_exercise":
            cluster_ref = str(_required(config, "cluster", job))
            system_ref = str(_required(config, "system", job))
            descriptor = self._cluster_descriptor(cluster_ref, system_ref)
            source = str(_required(config, "source", job))
            target = str(_required(config, "target", job))
            for node_id in (source, target):
                if node_id not in descriptor["system_nodes"]:
                    raise ValueError(
                        f"campaign job {job.id!r} exercise node is absent from system: "
                        f"{node_id}"
                    )
            result.update(
                execution_runs=0,
                source=source,
                target=target,
                physical_gate=descriptor,
            )
            return result
        if job.kind == "cluster_acceptance":
            cluster_ref = str(_required(config, "cluster", job))
            system_ref = str(_required(config, "system", job))
            descriptor = self._cluster_descriptor(cluster_ref, system_ref)
            source = str(_required(config, "source", job))
            target = str(_required(config, "target", job))
            for node_id in (source, target):
                if node_id not in descriptor["system_nodes"]:
                    raise ValueError(
                        f"campaign job {job.id!r} acceptance node is absent from system: "
                        f"{node_id}"
                    )
            result.update(
                execution_runs=0,
                source=source,
                target=target,
                physical_gate=descriptor,
                exercise_physical_control=bool(
                    config.get("exercise_physical_control", False)
                ),
                exercise_link_control=bool(config.get("exercise_link_control", False)),
            )
            return result

        path_fields = {
            "fidelity": (("real", "twin"), {"evidence"}),
            "trustworthy": (("fidelity",), {"evidence"}),
            "adaptive": (("fidelity",), {"evidence"}),
            "proactive": (("samples",), {"evidence"}),
            "explainable": (("trace",), {"evidence"}),
        }
        if job.kind in path_fields:
            fields, kinds = path_fields[job.kind]
            inputs = []
            for field_name in fields:
                reference = str(_required(config, field_name, job))
                path = self._resolve_resource(reference, kinds=kinds)
                inputs.append(self._file_descriptor(reference, path))
            result.update(execution_runs=0, inputs=inputs)
            return result
        if job.kind in {"robust", "explorable"}:
            result.update(execution_runs=0, config=dict(config))
            return result
        raise AssertionError(f"unhandled campaign job kind: {job.kind}")

    def _coverage(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        categories: dict[str, list[str]] = {}
        for job in jobs:
            raw = job.get("config", {}).get("evidence_for", ())
            if isinstance(raw, str):
                values = (raw,)
            elif isinstance(raw, (list, tuple)):
                values = tuple(str(item) for item in raw)
            else:
                raise ValueError(
                    f"campaign job {job['id']!r} evidence_for must be a string or list"
                )
            for category in values:
                if not category:
                    raise ValueError(
                        f"campaign job {job['id']!r} evidence_for cannot contain empty values"
                    )
                categories.setdefault(category, []).append(str(job["id"]))
        required = () if self.suite is None else self.suite.required_evidence
        missing = tuple(item for item in required if item not in categories)
        if missing:
            raise ValueError(
                "campaign does not cover required suite evidence: "
                + ", ".join(missing)
            )
        return {
            "required": list(required),
            "categories": {key: value for key, value in sorted(categories.items())},
            "missing": list(missing),
            "complete": not missing,
        }

    def plan(self) -> dict[str, Any]:
        jobs = [self._plan_job(job) for job in self.spec.jobs]
        coverage = self._coverage(jobs)
        payload: dict[str, Any] = {
            "schema": "darpan.campaign.plan/v1",
            "name": self.spec.name,
            "defaults": {
                "seed": self.spec.seed,
                "repeat": self.spec.repeat,
                "bootstrap_resamples": self.spec.bootstrap_resamples,
                "continue_on_error": self.spec.continue_on_error,
            },
            "suite": (
                None
                if self.suite is None or self.suite_lock_payload is None
                else {
                    "name": self.suite.name,
                    "version": self.suite.version,
                    "fingerprint": self.suite_lock_payload["fingerprint"],
                    "locked": self.spec.suite_lock is not None,
                }
            ),
            "jobs": jobs,
            "coverage": coverage,
            "total_jobs": len(jobs),
            "total_execution_runs": sum(int(job["execution_runs"]) for job in jobs),
        }
        payload["fingerprint"] = _canonical_hash(payload)
        if self.spec.plan_lock is not None:
            lock_path = self.spec.resolve(self.spec.plan_lock)
            if not lock_path.is_file():
                raise FileNotFoundError(f"campaign plan lock does not exist: {lock_path}")
            with lock_path.open("r", encoding="utf-8") as handle:
                expected = json.load(handle)
            expected_fingerprint = (
                expected.get("fingerprint") if isinstance(expected, dict) else None
            )
            if expected_fingerprint != payload["fingerprint"]:
                raise RuntimeError(
                    "campaign plan lock mismatch: "
                    f"expected {expected_fingerprint!r}, "
                    f"observed {payload['fingerprint']!r}"
                )
        return payload


class CampaignRunner(CampaignPlanner):
    def __init__(
        self,
        spec: CampaignSpec,
        *,
        execute: ExperimentExecutor,
        resume: bool = False,
    ) -> None:
        super().__init__(spec)
        self.execute = execute
        self.plan_payload = self.plan()
        self.resume = resume
        output = spec.resolve(spec.output)
        self._resume_results: dict[str, Any] | None = None
        if resume:
            self._resume_results = self._prepare_resume(output)
        else:
            output = require_fresh_artifact_directory(output)
        self.recorder = ResultRecorder(output)
        self._inputs: dict[str, str] = self._load_existing_inputs() if resume else {}

    def _load_existing_inputs(self) -> dict[str, str]:
        path = self.recorder.directory / "inputs" / "checksums.json"
        if not path.is_file():
            return {}
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"campaign input checksum file is malformed: {path}")
        return {str(key): str(value) for key, value in raw.items()}

    def _prepare_resume(self, output: Path) -> dict[str, Any]:
        if not output.is_dir():
            raise FileNotFoundError(f"campaign resume output does not exist: {output}")
        try:
            report = verify_artifact(output)
        except RuntimeError:
            report = restore_artifact_checkpoint(output)
        identity = report.get("identity", {})
        if not isinstance(identity, Mapping):
            identity = {}
        if bool(identity.get("complete", False)):
            raise RuntimeError("completed campaign artifacts cannot be resumed")
        expected_plan = self.plan_payload["fingerprint"]
        if identity.get("plan_fingerprint") != expected_plan:
            raise RuntimeError(
                "campaign resume plan mismatch: "
                f"artifact={identity.get('plan_fingerprint')!r}, current={expected_plan!r}"
            )
        state_path = output / "campaign-state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"campaign resume state does not exist: {state_path}")
        state = _read_json(state_path)
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
            raise ValueError(f"campaign resume state is malformed: {state_path}")
        if state.get("plan_fingerprint") != expected_plan:
            raise RuntimeError("campaign resume state plan fingerprint does not match")
        (output / "artifact-manifest.json").unlink()
        return {str(key): dict(value) for key, value in state["jobs"].items()}

    def _reset_job_artifacts(self, job_id: str) -> None:
        job_result = self.recorder.directory / "jobs" / f"{job_id}.json"
        if job_result.exists():
            job_result.unlink()
        progress_root = self.recorder.directory / "progress" / job_id
        run_root = self.recorder.directory / "runs" / job_id
        if run_root.exists() and not progress_root.is_dir():
            shutil.rmtree(run_root)

    def _seal_progress(self) -> None:
        self.recorder.write_json("inputs/checksums.json", self._inputs)
        seal_artifact(
            self.recorder.directory,
            schema="darpan.campaign-artifact/v1",
            identity={
                "plan_fingerprint": self.plan_payload["fingerprint"],
                "complete": False,
            },
        )

    def _rollback_inflight(self) -> None:
        try:
            verify_artifact(self.recorder.directory)
        except RuntimeError:
            restore_artifact_checkpoint(self.recorder.directory)
        manifest = self.recorder.directory / "artifact-manifest.json"
        if manifest.exists():
            manifest.unlink()
        self._inputs = self._load_existing_inputs()

    def _paired_progress(self, job_id: str) -> tuple[PairedSample, ...]:
        root = self.recorder.directory / "progress" / job_id
        if not root.is_dir():
            return ()
        samples = []
        for path in sorted(root.glob("seed-*.json")):
            raw = _read_json(path)
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            samples.append(
                PairedSample(
                    seed=int(raw["seed"]),
                    baseline=float(raw["baseline"]),
                    candidate=float(raw["candidate"]),
                    metadata=dict(metadata),
                )
            )
        return tuple(samples)

    def _scenario_progress(self, job_id: str) -> dict[int, dict[str, Any]]:
        root = self.recorder.directory / "progress" / job_id
        if not root.is_dir():
            return {}
        payloads: dict[int, dict[str, Any]] = {}
        for path in sorted(root.glob("seed-*.json")):
            raw = _read_json(path)
            seed = int(raw["seed"])
            payloads[seed] = raw
        return payloads

    def _capture(self, source: Path, name: str) -> None:
        destination = f"inputs/{name}"
        self.recorder.copy(source, destination)
        self._inputs[destination] = self.recorder.sha256(source)

    def _capture_experiment(self, path: Path, prefix: str) -> ExperimentSpec:
        spec = ExperimentSpec.load(path)
        for index, source in enumerate(resolved_inputs(spec), start=1):
            self._capture(source, f"{prefix}/{index:02d}-{source.name}")
        return spec

    def _campaign_payload(
        self,
        results: Mapping[str, Any],
        *,
        complete: bool,
    ) -> dict[str, Any]:
        successful = sum(item.get("status") == "completed" for item in results.values())
        failed = sum(item.get("status") == "failed" for item in results.values())
        return {
            "schema": "darpan.campaign.result/v1",
            "name": self.spec.name,
            "complete": complete,
            "plan_fingerprint": self.plan_payload["fingerprint"],
            "suite": (
                None
                if self.suite is None or self.suite_lock_payload is None
                else {
                    "name": self.suite.name,
                    "version": self.suite.version,
                    "fingerprint": self.suite_lock_payload["fingerprint"],
                    "locked": self.spec.suite_lock is not None,
                }
            ),
            "jobs": dict(results),
            "successful_jobs": successful,
            "failed_jobs": failed,
            "total_jobs": len(self.spec.jobs),
        }

    def _checkpoint(self, results: Mapping[str, Any], *, complete: bool) -> None:
        self.recorder.write_json(
            "campaign-state.json",
            self._campaign_payload(results, complete=complete),
        )
        self.recorder.write_json("inputs/checksums.json", self._inputs)
        seal_artifact(
            self.recorder.directory,
            schema="darpan.campaign-artifact/v1",
            identity={
                "plan_fingerprint": self.plan_payload["fingerprint"],
                "complete": complete,
            },
        )

    async def run(self) -> dict[str, Any]:
        if self.spec.source is not None:
            self._capture(self.spec.source, "campaign.yaml")
        if self.suite is not None and self.suite.source is not None:
            self._capture(self.suite.source, "suite.yaml")
        if self.spec.suite_lock is not None:
            self._capture(self.spec.resolve(self.spec.suite_lock), "suite.lock.json")
        if self.spec.plan_lock is not None:
            self._capture(self.spec.resolve(self.spec.plan_lock), "campaign-plan.lock.json")
        if self.suite_lock_payload is not None:
            self.recorder.write_json("suite-lock-observed.json", self.suite_lock_payload)
        self.recorder.write_json("campaign-plan.json", self.plan_payload)
        self.recorder.write_json(
            "provenance.json",
            asdict(collect_provenance(project_root=self.spec.directory)),
        )
        results: dict[str, Any] = {
            job.id: {"kind": job.kind, "status": "pending"} for job in self.spec.jobs
        }
        if self._resume_results is not None:
            for job in self.spec.jobs:
                previous = self._resume_results.get(job.id)
                if previous is None:
                    continue
                if previous.get("kind") != job.kind:
                    raise RuntimeError(
                        f"campaign resume job kind mismatch for {job.id!r}: "
                        f"artifact={previous.get('kind')!r}, current={job.kind!r}"
                    )
                if previous.get("status") == "completed":
                    results[job.id] = previous
        self._checkpoint(results, complete=False)
        for job in self.spec.jobs:
            if results[job.id].get("status") == "completed":
                continue
            self._reset_job_artifacts(job.id)
            try:
                result = await self._run_job(job)
            except Exception as exc:
                self._rollback_inflight()
                results[job.id] = {
                    "kind": job.kind,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self.recorder.write_json(f"jobs/{job.id}.json", results[job.id])
                self._checkpoint(results, complete=False)
                if not self.spec.continue_on_error:
                    self.recorder.write_json(
                        "campaign.json",
                        self._campaign_payload(results, complete=False),
                    )
                    seal_artifact(
                        self.recorder.directory,
                        schema="darpan.campaign-artifact/v1",
                        identity={
                            "plan_fingerprint": self.plan_payload["fingerprint"],
                            "complete": False,
                        },
                    )
                    raise
                continue
            results[job.id] = {
                "kind": job.kind,
                "status": "completed",
                "result": result,
            }
            self.recorder.write_json(f"jobs/{job.id}.json", results[job.id])
            self._checkpoint(results, complete=False)
        payload = self._campaign_payload(results, complete=True)
        self.recorder.write_json("campaign.json", payload)
        self._checkpoint(results, complete=True)
        seal_artifact(
            self.recorder.directory,
            schema="darpan.campaign-artifact/v1",
            identity={
                "plan_fingerprint": self.plan_payload["fingerprint"],
                "complete": True,
            },
        )
        return payload

    async def _run_job(self, job: CampaignJob) -> dict[str, Any]:
        handlers = {
            "paired": self._paired,
            "fidelity": self._fidelity,
            "trustworthy": self._trustworthy,
            "adaptive": self._adaptive,
            "proactive": self._proactive,
            "explainable": self._explainable,
            "robust": self._robust,
            "explorable": self._explorable,
            "scenario": self._scenario,
            "learning": self._learning,
            "cluster_preflight": self._cluster_preflight,
            "cluster_exercise": self._cluster_exercise,
            "cluster_acceptance": self._cluster_acceptance,
        }
        return await handlers[job.kind](job)

    async def _cluster_preflight(self, job: CampaignJob) -> dict[str, Any]:
        cluster_path = self._resolve_resource(
            str(_required(job.config, "cluster", job)), kinds={"cluster"}
        )
        system_value = job.config.get("system")
        system_path = (
            None
            if system_value is None
            else self._resolve_resource(str(system_value), kinds={"system"})
        )
        if not cluster_path.is_file():
            raise FileNotFoundError(f"cluster inventory does not exist: {cluster_path}")
        if system_path is not None and not system_path.is_file():
            raise FileNotFoundError(f"cluster system does not exist: {system_path}")
        self._capture(cluster_path, f"{job.id}-cluster.yaml")
        if system_path is not None:
            self._capture(system_path, f"{job.id}-system.yaml")
        report = await validate_cluster(
            ClusterInventory.load(cluster_path),
            system=None if system_path is None else load_system(system_path),
            exercise_data_plane=bool(job.config.get("exercise_data_plane", False)),
            payload_bytes=int(job.config.get("payload_bytes", 4096)),
            timeout_s=float(job.config.get("timeout_s", 5.0)),
        )
        payload = report.to_dict()
        payload["cluster"] = str(cluster_path)
        payload["system"] = None if system_path is None else str(system_path)
        if bool(job.config.get("require_ready", True)) and not report.ready:
            raise RuntimeError(
                f"physical cluster preflight failed for job {job.id!r}: "
                + "; ".join(report.errors)
            )
        return payload

    async def _cluster_exercise(self, job: CampaignJob) -> dict[str, Any]:
        cluster_path = self._resolve_resource(
            str(_required(job.config, "cluster", job)), kinds={"cluster"}
        )
        system_path = self._resolve_resource(
            str(_required(job.config, "system", job)), kinds={"system"}
        )
        self._capture(cluster_path, f"{job.id}-cluster.yaml")
        self._capture(system_path, f"{job.id}-system.yaml")
        python_command = str(job.config.get("python_command", sys.executable))
        report = await exercise_cluster_runtime(
            ClusterInventory.load(cluster_path),
            load_system(system_path),
            source_node_id=str(_required(job.config, "source", job)),
            target_node_id=str(_required(job.config, "target", job)),
            command=(python_command, "-c", "import time; time.sleep(3600)"),
            cpu_request=(
                None if job.config.get("cpu") is None else float(job.config["cpu"])
            ),
            timeout_s=float(job.config.get("timeout_s", 10.0)),
        )
        payload = report.to_dict()
        payload["cluster"] = str(cluster_path)
        payload["system"] = str(system_path)
        if bool(job.config.get("require_ready", True)) and not report.ready:
            raise RuntimeError(
                f"physical cluster exercise failed for job {job.id!r}: "
                + "; ".join(report.errors)
            )
        return payload

    async def _cluster_acceptance(self, job: CampaignJob) -> dict[str, Any]:
        cluster_path = self._resolve_resource(
            str(_required(job.config, "cluster", job)), kinds={"cluster"}
        )
        system_path = self._resolve_resource(
            str(_required(job.config, "system", job)), kinds={"system"}
        )
        self._capture(cluster_path, f"{job.id}-cluster.yaml")
        self._capture(system_path, f"{job.id}-system.yaml")
        python_command = str(job.config.get("python_command", sys.executable))
        report = await accept_cluster(
            ClusterInventory.load(cluster_path),
            load_system(system_path),
            source_node_id=str(_required(job.config, "source", job)),
            target_node_id=str(_required(job.config, "target", job)),
            command=(python_command, "-c", "import time; time.sleep(3600)"),
            python_command=python_command,
            cpu_request=(
                None if job.config.get("cpu") is None else float(job.config["cpu"])
            ),
            timeout_s=float(job.config.get("timeout_s", 10.0)),
            exercise_physical_control=bool(
                job.config.get("exercise_physical_control", False)
            ),
            exercise_link_control=bool(job.config.get("exercise_link_control", False)),
        )
        payload = report.to_dict()
        payload["cluster"] = str(cluster_path)
        payload["system"] = str(system_path)
        payload["deployment_receipt"] = build_deployment_receipt(
            report,
            cluster=cluster_path,
            system=system_path,
        ).to_dict()
        if bool(job.config.get("require_ready", True)) and not report.ready_for_study:
            raise RuntimeError(
                f"physical cluster acceptance failed for job {job.id!r}: "
                + "; ".join(report.errors)
            )
        return payload

    async def _paired(self, job: CampaignJob) -> dict[str, Any]:
        config = job.config
        baseline = str(_required(config, "baseline", job))
        candidate = str(_required(config, "candidate", job))
        metric = str(_required(config, "metric", job))
        spec = PairedExperimentBenchmarkSpec(
            baseline=baseline,
            candidate=candidate,
            metric=metric,
            higher_is_better=bool(config.get("higher_is_better", True)),
            require_success=bool(config.get("require_success", True)),
            seed=int(config.get("seed", self.spec.seed)),
            repeat=int(config.get("repeat", self.spec.repeat)),
            seeds=tuple(int(item) for item in config.get("seeds", ())),
            bootstrap_resamples=int(
                config.get("bootstrap_resamples", self.spec.bootstrap_resamples)
            ),
            output=str(self.recorder.directory / "runs" / job.id),
            source=self.spec.source,
        )
        baseline_path = self._resolve_resource(baseline, kinds={"experiment"})
        candidate_path = self._resolve_resource(candidate, kinds={"experiment"})
        spec = replace(
            spec,
            baseline=str(baseline_path),
            candidate=str(candidate_path),
        )
        for label, path in (("baseline", baseline_path), ("candidate", candidate_path)):
            if not path.is_file():
                raise FileNotFoundError(f"campaign {job.id} {label} does not exist: {path}")
            self._capture_experiment(path, f"{job.id}-{label}")
        async def checkpoint_sample(sample: PairedSample) -> None:
            self.recorder.write_json(
                f"progress/{job.id}/seed-{sample.seed}.json",
                asdict(sample),
            )
            self._seal_progress()

        payload, _ = await run_paired_benchmark(
            spec,
            execute=self.execute,
            initial_samples=self._paired_progress(job.id),
            on_sample=checkpoint_sample,
        )
        return payload

    async def _fidelity(self, job: CampaignJob) -> dict[str, Any]:
        real = self._resolve_resource(
            str(_required(job.config, "real", job)), kinds={"evidence"}
        )
        twin = self._resolve_resource(
            str(_required(job.config, "twin", job)), kinds={"evidence"}
        )
        real_trace = resolve_event_trace(real)
        twin_trace = resolve_event_trace(twin)
        self._capture(real_trace, f"{job.id}-real-events.jsonl")
        self._capture(twin_trace, f"{job.id}-twin-events.jsonl")
        return compare_event_traces(
            load_event_trace(real_trace),
            load_event_trace(twin_trace),
        ).to_dict()

    async def _trustworthy(self, job: CampaignJob) -> dict[str, Any]:
        path = self._resolve_resource(
            str(_required(job.config, "fidelity", job)), kinds={"evidence"}
        )
        layer = str(job.config.get("layer", "execution"))
        self._capture(path, f"{job.id}-fidelity.json")
        samples = _fidelity_samples(_fidelity_layer(path, layer))
        return asdict(summarize_fidelity(samples))

    async def _adaptive(self, job: CampaignJob) -> dict[str, Any]:
        path = self._resolve_resource(
            str(_required(job.config, "fidelity", job)), kinds={"evidence"}
        )
        layer = str(job.config.get("layer", "execution"))
        self._capture(path, f"{job.id}-fidelity.json")
        samples = _fidelity_samples(_fidelity_layer(path, layer))
        window = job.config.get("window_size")
        summary = summarize_calibration_progress(
            samples,
            window_size=None if window is None else int(window),
        )
        payload = asdict(summary)
        payload["improved"] = summary.improved
        return payload

    async def _proactive(self, job: CampaignJob) -> dict[str, Any]:
        path = self._resolve_resource(
            str(_required(job.config, "samples", job)), kinds={"evidence"}
        )
        self._capture(path, f"{job.id}-samples{path.suffix or '.json'}")
        samples = tuple(
            ProactiveSample(
                predicted_violation=bool(item["predicted_violation"]),
                observed_violation=bool(item["observed_violation"]),
                lead_time_s=(
                    None if item.get("lead_time_s") is None else float(item["lead_time_s"])
                ),
                uncertainty=(
                    None if item.get("uncertainty") is None else float(item["uncertainty"])
                ),
            )
            for item in _read_records(path)
        )
        return asdict(assess_proactive_benchmark(samples))

    async def _explainable(self, job: CampaignJob) -> dict[str, Any]:
        path = resolve_event_trace(
            self._resolve_resource(
                str(_required(job.config, "trace", job)), kinds={"evidence"}
            )
        )
        self._capture(path, f"{job.id}-events.jsonl")
        return asdict(assess_explainability(load_event_trace(path)))

    async def _robust(self, job: CampaignJob) -> dict[str, Any]:
        scenarios = _required(job.config, "scenarios", job)
        if not isinstance(scenarios, Mapping):
            raise ValueError(f"campaign job {job.id!r} scenarios must be a mapping")
        return asdict(
            assess_robustness(
                float(_required(job.config, "baseline", job)),
                {str(key): float(value) for key, value in scenarios.items()},
                higher_is_better=bool(job.config.get("higher_is_better", True)),
                tolerance=float(job.config.get("tolerance", 0.1)),
            )
        )

    async def _explorable(self, job: CampaignJob) -> dict[str, Any]:
        alternatives = _required(job.config, "alternatives", job)
        if not isinstance(alternatives, Mapping) or not alternatives:
            raise ValueError(f"campaign job {job.id!r} alternatives must be a mapping")
        baseline = float(_required(job.config, "baseline", job))
        values = {str(key): float(value) for key, value in alternatives.items()}
        return {
            "metric": str(_required(job.config, "metric", job)),
            "baseline": baseline,
            "alternatives": values,
            "deltas": {key: value - baseline for key, value in values.items()},
        }
    async def _learning(self, job: CampaignJob) -> dict[str, Any]:
        baseline_raw = _required(job.config, "baseline", job)
        candidate_raw = _required(job.config, "candidate", job)
        if not isinstance(baseline_raw, Mapping) or not isinstance(candidate_raw, Mapping):
            raise ValueError(
                f"campaign job {job.id!r} baseline/candidate must map seed to trace"
            )
        baseline_paths = {
            int(seed): self._resolve_resource(str(path), kinds={"evidence"})
            for seed, path in baseline_raw.items()
        }
        candidate_paths = {
            int(seed): self._resolve_resource(str(path), kinds={"evidence"})
            for seed, path in candidate_raw.items()
        }
        if set(baseline_paths) != set(candidate_paths):
            raise ValueError(
                f"campaign job {job.id!r} requires identical baseline/candidate seeds"
            )
        threshold = job.config.get("threshold")
        threshold_value = None if threshold is None else float(threshold)
        higher_is_better = bool(job.config.get("higher_is_better", True))
        baseline = {}
        candidate = {}
        budget_raw = job.config.get("budget", {}) or {}
        if not isinstance(budget_raw, Mapping):
            raise ValueError(f"campaign job {job.id!r} learning budget must be a mapping")
        baseline_budget = budget_raw.get("baseline", budget_raw)
        candidate_budget = budget_raw.get("candidate", budget_raw)
        if not isinstance(baseline_budget, Mapping) or not isinstance(candidate_budget, Mapping):
            raise ValueError(
                f"campaign job {job.id!r} baseline/candidate learning budgets must be mappings"
            )
        budget_evidence: dict[str, dict[str, Any]] = {"baseline": {}, "candidate": {}}
        for seed in sorted(baseline_paths):
            for label, path in (
                ("baseline", baseline_paths[seed]),
                ("candidate", candidate_paths[seed]),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"learning trace does not exist: {path}")
                self._capture(
                    path,
                    f"{job.id}-{label}-seed-{seed}{path.suffix or '.jsonl'}",
                )
            baseline[seed] = summarize_learning_run(
                load_learning_trace(baseline_paths[seed]),
                threshold=threshold_value,
                higher_is_better=higher_is_better,
            )
            candidate[seed] = summarize_learning_run(
                load_learning_trace(candidate_paths[seed]),
                threshold=threshold_value,
                higher_is_better=higher_is_better,
            )
            budget_evidence["baseline"][str(seed)] = validate_learning_budget(
                baseline[seed], dict(baseline_budget), label=f"baseline seed {seed}"
            )
            budget_evidence["candidate"][str(seed)] = validate_learning_budget(
                candidate[seed], dict(candidate_budget), label=f"candidate seed {seed}"
            )
        payload = compare_learning_runs(
            baseline,
            candidate,
            higher_is_better=higher_is_better,
            bootstrap_seed=int(job.config.get("seed", self.spec.seed)),
            bootstrap_resamples=int(
                job.config.get("bootstrap_resamples", self.spec.bootstrap_resamples)
            ),
        )
        payload["budget"] = {
            "baseline": dict(baseline_budget),
            "candidate": dict(candidate_budget),
            "evidence": budget_evidence,
        }
        return payload

    async def _scenario(self, job: CampaignJob) -> dict[str, Any]:
        config = job.config
        baseline_value = str(_required(config, "baseline", job))
        scenarios_raw = _required(config, "scenarios", job)
        metric = str(_required(config, "metric", job))
        if not isinstance(scenarios_raw, Mapping) or not scenarios_raw:
            raise ValueError(f"campaign job {job.id!r} scenarios must be a mapping")
        baseline_path = self._resolve_resource(baseline_value, kinds={"experiment"})
        if not baseline_path.is_file():
            raise FileNotFoundError(f"scenario baseline does not exist: {baseline_path}")
        scenario_paths = {
            str(name): self._resolve_resource(str(value), kinds={"experiment"})
            for name, value in scenarios_raw.items()
        }
        missing = [str(path) for path in scenario_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("scenario experiments do not exist: " + ", ".join(missing))
        baseline_template = self._capture_experiment(
            baseline_path,
            f"{job.id}-baseline",
        )
        scenario_templates = {
            name: self._capture_experiment(path, f"{job.id}-scenario-{name}")
            for name, path in scenario_paths.items()
        }
        seeds = tuple(int(item) for item in config.get("seeds", ()))
        if not seeds:
            repeat = int(config.get("repeat", self.spec.repeat))
            seed = int(config.get("seed", self.spec.seed))
            seeds = tuple(seed + index for index in range(repeat))
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"campaign job {job.id!r} seeds must be unique")

        baseline_samples: dict[int, float] = {}
        baseline_success: dict[int, bool] = {}
        baseline_recovery: dict[int, dict[str, Any]] = {}
        scenario_samples: dict[str, list[PairedSample]] = {
            name: [] for name in scenario_templates
        }
        scenario_success: dict[str, list[bool]] = {
            name: [] for name in scenario_templates
        }
        scenario_recovery: dict[str, list[dict[str, Any]]] = {
            name: [] for name in scenario_templates
        }
        progress = self._scenario_progress(job.id)
        unexpected_progress = set(progress) - set(seeds)
        if unexpected_progress:
            raise RuntimeError(
                f"scenario progress contains unexpected seeds: {sorted(unexpected_progress)}"
            )
        require_success = bool(config.get("require_success", True))
        for seed in seeds:
            seed_payload = progress.get(seed)
            if seed_payload is None:
                seed_root = self.recorder.directory / "runs" / job.id / f"seed-{seed}"
                baseline_output = await self.execute(
                    replace(
                        baseline_template,
                        repeat=1,
                        seed=seed,
                        output=str(seed_root / "baseline"),
                    )
                )
                if require_success:
                    ensure_successful(
                        baseline_output, label=f"scenario baseline seed {seed}"
                    )
                baseline_metric = metric_value(baseline_output, metric)
                baseline_ok = experiment_success(baseline_output)
                scenarios_payload: dict[str, Any] = {}
                for name, template in scenario_templates.items():
                    output = await self.execute(
                        replace(
                            template,
                            repeat=1,
                            seed=seed,
                            output=str(seed_root / "scenarios" / name),
                        )
                    )
                    if require_success:
                        ensure_successful(output, label=f"scenario {name} seed {seed}")
                    scenarios_payload[name] = {
                        "metric": metric_value(output, metric),
                        "successful": experiment_success(output),
                        "recovery": _recovery_evidence(output),
                    }
                seed_payload = {
                    "seed": seed,
                    "baseline": {
                        "metric": baseline_metric,
                        "successful": baseline_ok,
                        "recovery": _recovery_evidence(baseline_output),
                    },
                    "scenarios": scenarios_payload,
                }
                self.recorder.write_json(
                    f"progress/{job.id}/seed-{seed}.json",
                    seed_payload,
                )
                self._seal_progress()

            baseline_raw = seed_payload.get("baseline", {})
            scenarios_payload = seed_payload.get("scenarios", {})
            if not isinstance(baseline_raw, Mapping) or not isinstance(
                scenarios_payload, Mapping
            ):
                raise ValueError(f"scenario progress seed {seed} is malformed")
            baseline_metric = float(baseline_raw["metric"])
            baseline_ok = bool(baseline_raw.get("successful", True))
            baseline_samples[seed] = baseline_metric
            baseline_success[seed] = baseline_ok
            baseline_recovery[seed] = dict(baseline_raw.get("recovery", {}))
            for name in scenario_templates:
                raw = scenarios_payload.get(name)
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        f"scenario progress seed {seed} is missing scenario {name!r}"
                    )
                successful = bool(raw.get("successful", True))
                scenario_success[name].append(successful)
                scenario_recovery[name].append(dict(raw.get("recovery", {})))
                scenario_samples[name].append(
                    PairedSample(
                        seed=seed,
                        baseline=baseline_metric,
                        candidate=float(raw["metric"]),
                        metadata={
                            "baseline_successful": baseline_ok,
                            "candidate_successful": successful,
                        },
                    )
                )

        higher_is_better = bool(config.get("higher_is_better", True))
        bootstrap_resamples = int(
            config.get("bootstrap_resamples", self.spec.bootstrap_resamples)
        )
        summaries = {
            name: summarize_paired(
                samples,
                higher_is_better=higher_is_better,
                bootstrap_seed=int(config.get("seed", self.spec.seed)),
                bootstrap_resamples=bootstrap_resamples,
            )
            for name, samples in scenario_samples.items()
        }
        baseline_mean = sum(baseline_samples.values()) / len(baseline_samples)
        scenario_means = {name: summary.candidate_mean for name, summary in summaries.items()}
        robustness = assess_robustness(
            baseline_mean,
            scenario_means,
            higher_is_better=higher_is_better,
            tolerance=float(config.get("tolerance", 0.1)),
        )
        return {
            "metric": metric,
            "seeds": list(seeds),
            "baseline_mean": baseline_mean,
            "scenario_means": scenario_means,
            "success": {
                "baseline_rate": sum(baseline_success.values()) / len(baseline_success),
                "scenario_rates": {
                    name: sum(values) / len(values)
                    for name, values in scenario_success.items()
                },
            },
            "paired": {name: asdict(summary) for name, summary in summaries.items()},
            "recovery": {
                "baseline": _summarize_recovery(list(baseline_recovery.values())),
                "scenarios": {
                    name: _summarize_recovery(items)
                    for name, items in scenario_recovery.items()
                },
                "per_seed": {
                    "baseline": {str(seed): item for seed, item in baseline_recovery.items()},
                    "scenarios": {
                        name: {str(seed): item for seed, item in zip(seeds, items, strict=True)}
                        for name, items in scenario_recovery.items()
                    },
                },
            },
            "robustness": asdict(robustness),
            "exploration": {
                "metric": metric,
                "baseline": baseline_mean,
                "alternatives": scenario_means,
                "deltas": {
                    name: value - baseline_mean for name, value in scenario_means.items()
                },
            },
        }
