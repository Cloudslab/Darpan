"""Derive a paper-grade fidelity-batch manifest from two verified Study artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifact import verify_artifact
from .recorder import ResultRecorder
from .study_run import verify_study

STUDY_FIDELITY_MAP_SCHEMA = "darpan.study-fidelity-map/v1"
_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class StudyArmRef:
    job: str
    arm: str


@dataclass(frozen=True, slots=True)
class StudyFidelityPair:
    id: str
    real: StudyArmRef
    twin: StudyArmRef
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class StudyFidelitySpec:
    path: Path
    source_sha256: str
    name: str
    real_study: str
    twin_study: str
    pairs: tuple[StudyFidelityPair, ...]
    refinement_policy: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> StudyFidelitySpec:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Study fidelity map does not exist: {source}")
        source_bytes = source.read_bytes()
        raw = yaml.safe_load(source_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Study fidelity map must contain a mapping")
        if raw.get("schema") != STUDY_FIDELITY_MAP_SCHEMA:
            raise ValueError(
                f"Study fidelity map schema must be {STUDY_FIDELITY_MAP_SCHEMA!r}"
            )
        real_study = str(raw.get("real_study", "")).strip()
        twin_study = str(raw.get("twin_study", "")).strip()
        if not real_study or not twin_study:
            raise ValueError("Study fidelity map requires real_study and twin_study")
        raw_pairs = raw.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("Study fidelity map requires at least one pair mapping")
        pairs: list[StudyFidelityPair] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_pairs, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Study fidelity pair #{index} must be a mapping")
            pair_id = str(item.get("id", "")).strip()
            if not _PAIR_ID.fullmatch(pair_id):
                raise ValueError(f"invalid Study fidelity pair id: {pair_id!r}")
            if pair_id in seen:
                raise ValueError(f"duplicate Study fidelity pair id: {pair_id}")
            seen.add(pair_id)
            real = _arm_ref(item.get("real"), index=index, side="real")
            twin = _arm_ref(item.get("twin"), index=index, side="twin")
            raw_seeds = item.get("seeds", ())
            if raw_seeds is None:
                raw_seeds = ()
            if not isinstance(raw_seeds, (list, tuple)):
                raise ValueError(f"Study fidelity pair {pair_id!r} seeds must be a list")
            seeds = tuple(int(seed) for seed in raw_seeds)
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Study fidelity pair {pair_id!r} seeds must be unique")
            pairs.append(StudyFidelityPair(pair_id, real, twin, seeds))
        policy = raw.get("refinement_policy")
        if policy is not None and not str(policy).strip():
            raise ValueError("refinement_policy must be a non-empty path when provided")
        return cls(
            path=source,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            name=str(raw.get("name") or source.stem),
            real_study=real_study,
            twin_study=twin_study,
            pairs=tuple(pairs),
            refinement_policy=None if policy is None else str(policy).strip(),
        )

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()


def _arm_ref(value: Any, *, index: int, side: str) -> StudyArmRef:
    if not isinstance(value, Mapping):
        raise ValueError(f"Study fidelity pair #{index} {side} must be a mapping")
    job = str(value.get("job", "")).strip()
    arm = str(value.get("arm", "")).strip()
    if not job or not arm:
        raise ValueError(f"Study fidelity pair #{index} {side} requires job and arm")
    return StudyArmRef(job, arm)


def _portable_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _planned_job(study_root: Path, job_id: str) -> dict[str, Any]:
    plan = _read_json(study_root / "study-plan.json")
    jobs = plan.get("campaign_plan", {}).get("jobs", ())
    if not isinstance(jobs, list):
        raise ValueError("Study campaign plan jobs are malformed")
    for item in jobs:
        if isinstance(item, dict) and str(item.get("id")) == job_id:
            return item
    raise KeyError(f"Study campaign does not contain job {job_id!r}")


def _job_seeds(job: Mapping[str, Any]) -> tuple[int, ...]:
    raw = job.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Study fidelity job {job.get('id')!r} does not expose matched run seeds"
        )
    return tuple(int(seed) for seed in raw)


def _arm_directory(study_root: Path, job: Mapping[str, Any], arm: str, seed: int) -> Path:
    job_id = str(job["id"])
    kind = str(job.get("kind", ""))
    if kind == "paired":
        if arm not in {"baseline", "candidate"}:
            raise ValueError(
                f"paired Study job {job_id!r} arm must be baseline or candidate"
            )
        return (
            study_root
            / "campaign"
            / "runs"
            / job_id
            / "runs"
            / f"seed-{seed}"
            / arm
        )
    if kind == "scenario":
        if arm == "baseline":
            return (
                study_root / "campaign" / "runs" / job_id / f"seed-{seed}" / "baseline"
            )
        prefix = "scenario:"
        if not arm.startswith(prefix) or not arm[len(prefix) :]:
            raise ValueError(
                f"scenario Study job {job_id!r} arm must be baseline or scenario:<name>"
            )
        scenario = arm[len(prefix) :]
        return (
            study_root
            / "campaign"
            / "runs"
            / job_id
            / f"seed-{seed}"
            / "scenarios"
            / scenario
        )
    raise ValueError(
        f"Study fidelity currently supports paired/scenario experiment jobs, got {kind!r}"
    )


def _verify_experiment_arm(path: Path, *, seed: int, runtime: str) -> dict[str, Any]:
    verification = verify_artifact(path)
    if verification.get("schema") != "darpan.experiment-artifact/v1":
        raise ValueError(f"Study fidelity arm is not an experiment artifact: {path}")
    if verification.get("identity", {}).get("complete") is not True:
        raise RuntimeError(f"Study fidelity arm is incomplete: {path}")
    portable = _read_json(path / "experiment-portable.json")
    if str(portable.get("runtime")) != runtime:
        raise ValueError(
            f"Study fidelity {runtime} arm has runtime={portable.get('runtime')!r}: {path}"
        )
    metadata = sorted(path.glob("run-*/metadata.json"))
    if len(metadata) != 1:
        raise ValueError(
            "Study fidelity requires exactly one run in each campaign arm; "
            f"found {len(metadata)} under {path}"
        )
    run_metadata = _read_json(metadata[0])
    if int(run_metadata.get("seed", -1)) != seed:
        raise RuntimeError(
            f"Study fidelity arm seed mismatch: expected {seed}, "
            f"observed {run_metadata.get('seed')} under {path}"
        )
    return verification


def build_study_fidelity_manifest(
    mapping: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    """Expand two complete verified Studies into an explicit fidelity-batch manifest."""

    spec = StudyFidelitySpec.load(mapping)
    output = Path(output_manifest).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Study fidelity manifest output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    real_root = spec.resolve(spec.real_study)
    twin_root = spec.resolve(spec.twin_study)
    real_verification = verify_study(real_root)
    twin_verification = verify_study(twin_root)

    expanded: list[dict[str, Any]] = []
    for pair in spec.pairs:
        real_job = _planned_job(real_root, pair.real.job)
        twin_job = _planned_job(twin_root, pair.twin.job)
        real_seeds = _job_seeds(real_job)
        twin_seeds = _job_seeds(twin_job)
        if pair.seeds:
            seeds = pair.seeds
            missing_real = sorted(set(seeds) - set(real_seeds))
            missing_twin = sorted(set(seeds) - set(twin_seeds))
            if missing_real or missing_twin:
                raise ValueError(
                    f"Study fidelity pair {pair.id!r} requested unavailable seeds; "
                    f"missing_real={missing_real}, missing_twin={missing_twin}"
                )
        else:
            if real_seeds != twin_seeds:
                raise ValueError(
                    f"Study fidelity pair {pair.id!r} seed sets differ; explicitly freeze "
                    "a common seed subset or rerun matched Studies"
                )
            seeds = real_seeds
        for seed in seeds:
            real_arm = _arm_directory(real_root, real_job, pair.real.arm, seed)
            twin_arm = _arm_directory(twin_root, twin_job, pair.twin.arm, seed)
            real_arm_verification = _verify_experiment_arm(
                real_arm, seed=seed, runtime="real"
            )
            twin_arm_verification = _verify_experiment_arm(
                twin_arm, seed=seed, runtime="twin"
            )
            expanded.append(
                {
                    "id": f"{pair.id}-seed-{seed}",
                    "real": _portable_path(real_arm, output.parent),
                    "twin": _portable_path(twin_arm, output.parent),
                    "source": {
                        "mapping_id": pair.id,
                        "seed": seed,
                        "real_job": pair.real.job,
                        "real_arm": pair.real.arm,
                        "twin_job": pair.twin.job,
                        "twin_arm": pair.twin.arm,
                        "real_experiment_manifest_fingerprint": real_arm_verification[
                            "manifest_fingerprint"
                        ],
                        "twin_experiment_manifest_fingerprint": twin_arm_verification[
                            "manifest_fingerprint"
                        ],
                    },
                }
            )

    batch: dict[str, Any] = {
        "schema": "darpan.fidelity-batch/v1",
        "name": spec.name,
        "require_sealed_inputs": True,
        "source_studies": {
            "real": {
                "path": _portable_path(real_root, output.parent),
                "study_fingerprint": real_verification["study_fingerprint"],
                "manifest_fingerprint": real_verification["manifest_fingerprint"],
            },
            "twin": {
                "path": _portable_path(twin_root, output.parent),
                "study_fingerprint": twin_verification["study_fingerprint"],
                "manifest_fingerprint": twin_verification["manifest_fingerprint"],
            },
        },
        "source_mapping": {
            "path": _portable_path(spec.path, output.parent),
            "sha256": spec.source_sha256,
        },
        "pairs": [
            {
                "id": item["id"],
                "real": item["real"],
                "twin": item["twin"],
                "source": item["source"],
            }
            for item in expanded
        ],
    }
    if spec.refinement_policy is not None:
        policy = spec.resolve(spec.refinement_policy)
        if not policy.is_file():
            raise FileNotFoundError(f"refinement policy does not exist: {policy}")
        batch["refinement_policy"] = _portable_path(policy, output.parent)

    output.write_text(yaml.safe_dump(batch, sort_keys=False), encoding="utf-8")
    return {
        "schema": STUDY_FIDELITY_MAP_SCHEMA,
        "name": spec.name,
        "output": str(output),
        "output_sha256": ResultRecorder.sha256(output),
        "pair_count": len(expanded),
        "real_study_fingerprint": real_verification["study_fingerprint"],
        "twin_study_fingerprint": twin_verification["study_fingerprint"],
        "pairs": expanded,
    }
