"""Deterministic seeding and durable experiment-run artifact capture."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from darpan.core.loading import resolve_local_plugin_file
from darpan.core.serialization import to_primitive
from darpan.runtime.session import Session

from .artifact import require_fresh_artifact_directory, seal_artifact
from .provenance import collect_provenance
from .recorder import ResultRecorder
from .spec import ExperimentSpec


def seed_everything(seed: int) -> None:
    """Seed Darpan's supported process-local stochastic sources."""

    random.seed(seed)
    np.random.seed(seed % (2**32))


def resolved_inputs(spec: ExperimentSpec) -> list[Path]:
    inputs = [spec.resolve(spec.system)]
    if spec.application is not None:
        inputs.append(spec.resolve(spec.application))
    if spec.workload is not None:
        inputs.append(spec.resolve(spec.workload))
    if spec.cluster is not None:
        inputs.append(spec.resolve(spec.cluster))
    if spec.model_snapshot is not None:
        inputs.append(spec.resolve(spec.model_snapshot))
    if spec.scenario is not None:
        inputs.append(spec.resolve(spec.scenario))
    plugin_references = (
        spec.policy,
        *spec.metrics,
        *spec.models,
        *((spec.network_driver,) if spec.network_driver is not None else ()),
    )
    for reference in plugin_references:
        if ":" not in reference:
            continue
        plugin = resolve_local_plugin_file(reference, search_path=spec.directory)
        if plugin is not None:
            inputs.append(plugin)
    if spec.source is not None:
        inputs.append(spec.source)
    unique: dict[Path, None] = {}
    for path in inputs:
        unique[path.resolve()] = None
    return list(unique)


def experiment_fingerprint(spec: ExperimentSpec) -> str:
    """Fingerprint portable experiment semantics plus every resolved input file."""

    portable = spec.to_dict(resolved=False)
    portable.pop("output", None)
    inputs = []
    for path in sorted(resolved_inputs(spec), key=lambda item: str(item)):
        rel = Path(os.path.relpath(path.resolve(), spec.directory.resolve())).as_posix()
        inputs.append(
            {
                "path": rel,
                "sha256": ResultRecorder.sha256(path),
            }
        )
    payload = {
        "schema": "darpan.experiment.plan/v1",
        "experiment": portable,
        "inputs": inputs,
    }
    rendered = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def experiment_run_id(fingerprint: str, *, index: int, seed: int) -> str:
    rendered = f"{fingerprint}:{index}:{seed}".encode()
    return hashlib.sha256(rendered).hexdigest()[:20]


class ExperimentArtifactRecorder:
    """Write one self-describing directory for a CLI experiment."""

    def __init__(self, directory: str | Path, spec: ExperimentSpec) -> None:
        self.directory = require_fresh_artifact_directory(directory)
        self.spec = spec
        self.root = ResultRecorder(self.directory)
        self.fingerprint = experiment_fingerprint(spec)
        self.run_ids: list[str] = []
        self._write_inputs()
        provenance = collect_provenance(project_root=Path.cwd())
        self.root.write_json("provenance.json", asdict(provenance))
        self.root.write_json("experiment.json", self.spec.to_dict(resolved=True))
        self.root.write_json("experiment-portable.json", self.spec.to_dict(resolved=False))
        self.root.write_json(
            "experiment-identity.json",
            {
                "schema": "darpan.experiment.plan/v1",
                "fingerprint": self.fingerprint,
            },
        )

    def _write_inputs(self) -> None:
        manifest = []
        recorder = ResultRecorder(self.directory / "inputs")
        for index, source in enumerate(resolved_inputs(self.spec), start=1):
            name = f"{index:02d}-{source.name}"
            destination = recorder.copy(source, name)
            manifest.append(
                {
                    "source": str(source),
                    "file": str(destination.relative_to(self.directory)),
                    "sha256": recorder.sha256(destination),
                }
            )
        self.root.write_json("inputs-manifest.json", manifest)

    def record_run(
        self,
        *,
        index: int,
        seed: int,
        session: Session,
        result: dict[str, Any],
        extra_json: dict[str, Any] | None = None,
    ) -> None:
        recorder = ResultRecorder(self.directory / f"run-{index:04d}")
        run_id = experiment_run_id(self.fingerprint, index=index, seed=seed)
        self.run_ids.append(run_id)
        recorder.write_json("result.json", result)
        recorder.write_json(
            "metadata.json",
            {
                "schema": "darpan.experiment.run/v1",
                "run_id": run_id,
                "experiment_fingerprint": self.fingerprint,
                "index": index,
                "seed": seed,
            },
        )
        recorder.write_json("state.json", to_primitive(session.state))
        recorder.write_jsonl(
            "events.jsonl",
            (event.to_dict() for event in session.event_log),
        )
        models = getattr(session.backend, "models", None)
        if models is not None and hasattr(models, "snapshot"):
            recorder.write_json("models.json", models.snapshot())
        for name, payload in (extra_json or {}).items():
            recorder.write_json(name, payload)

    def record_failed_run(
        self,
        *,
        index: int,
        seed: int,
        session: Session,
        error: BaseException,
        extra_json: dict[str, Any] | None = None,
    ) -> None:
        recorder = ResultRecorder(self.directory / f"run-{index:04d}")
        run_id = experiment_run_id(self.fingerprint, index=index, seed=seed)
        self.run_ids.append(run_id)
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
        }
        recorder.write_json(
            "metadata.json",
            {
                "schema": "darpan.experiment.run/v1",
                "run_id": run_id,
                "experiment_fingerprint": self.fingerprint,
                "index": index,
                "seed": seed,
                "status": "failed",
            },
        )
        recorder.write_json("failure.json", error_payload)
        recorder.write_json("state.json", to_primitive(session.state))
        recorder.write_jsonl(
            "events.jsonl",
            (event.to_dict() for event in session.event_log),
        )
        models = getattr(session.backend, "models", None)
        if models is not None and hasattr(models, "snapshot"):
            recorder.write_json("models.json", models.snapshot())
        for name, payload in (extra_json or {}).items():
            recorder.write_json(name, payload)
        self.root.write_json(
            "failure.json",
            {
                "complete": False,
                "failed_run": index,
                "run_id": run_id,
                "error": error_payload,
            },
        )
        seal_artifact(
            self.directory,
            schema="darpan.experiment-artifact/v1",
            identity={
                "experiment_fingerprint": self.fingerprint,
                "run_ids": list(self.run_ids),
                "complete": False,
            },
        )

    def record_summary(self, payload: dict[str, Any]) -> None:
        self.root.write_json("summary.json", payload)
        seal_artifact(
            self.directory,
            schema="darpan.experiment-artifact/v1",
            identity={
                "experiment_fingerprint": self.fingerprint,
                "run_ids": list(self.run_ids),
                "complete": True,
            },
        )
