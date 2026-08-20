"""Matched-run aggregation for durable Real-vs-Twin fidelity evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .artifact import require_fresh_artifact_directory, seal_artifact, verify_artifact
from .benchmark import FidelitySample, summarize_fidelity
from .fidelity_diagnosis import diagnose_trace_fidelity
from .recorder import ResultRecorder
from .trace_fidelity import (
    TraceFidelityReport,
    TraceMetricReport,
    compare_event_traces,
    load_event_trace,
    resolve_event_trace,
)

FIDELITY_BATCH_SCHEMA = "darpan.fidelity-batch/v1"
_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_METRIC_FIELDS = (
    "application_latency",
    "execution_duration",
    "artifact_size",
    "queue_delay",
    "transfer_duration",
    "migration_downtime",
    "restart_downtime",
    "retry_downtime",
    "scale_convergence",
    "route_change",
)


@dataclass(frozen=True, slots=True)
class FidelityPairSpec:
    id: str
    real: str
    twin: str


@dataclass(frozen=True, slots=True)
class FidelityBatchSpec:
    path: Path
    source_sha256: str
    name: str
    pairs: tuple[FidelityPairSpec, ...]
    refinement_policy: str | None = None
    require_sealed_inputs: bool = False

    @classmethod
    def load(cls, path: str | Path) -> FidelityBatchSpec:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"fidelity batch manifest does not exist: {source}")
        source_bytes = source.read_bytes()
        raw = yaml.safe_load(source_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("fidelity batch manifest must contain a mapping")
        schema = raw.get("schema")
        if schema != FIDELITY_BATCH_SCHEMA:
            raise ValueError(
                f"fidelity batch schema must be {FIDELITY_BATCH_SCHEMA!r}, got {schema!r}"
            )
        raw_pairs = raw.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("fidelity batch manifest requires at least one pair")
        pairs: list[FidelityPairSpec] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_pairs):
            if not isinstance(item, dict):
                raise ValueError(f"fidelity pair #{index + 1} must be a mapping")
            pair_id = str(item.get("id", "")).strip()
            if not _PAIR_ID.fullmatch(pair_id):
                raise ValueError(
                    "fidelity pair id must use only letters, digits, '.', '_' or '-': "
                    f"{pair_id!r}"
                )
            if pair_id in seen:
                raise ValueError(f"duplicate fidelity pair id: {pair_id}")
            seen.add(pair_id)
            real = str(item.get("real", "")).strip()
            twin = str(item.get("twin", "")).strip()
            if not real or not twin:
                raise ValueError(f"fidelity pair {pair_id!r} requires real and twin paths")
            pairs.append(FidelityPairSpec(id=pair_id, real=real, twin=twin))
        refinement_policy = raw.get("refinement_policy")
        if refinement_policy is not None and not str(refinement_policy).strip():
            raise ValueError("refinement_policy must be a non-empty path when provided")
        require_sealed_inputs = raw.get("require_sealed_inputs", False)
        if not isinstance(require_sealed_inputs, bool):
            raise ValueError("require_sealed_inputs must be a boolean")
        return cls(
            path=source,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            name=str(raw.get("name") or source.stem),
            pairs=tuple(pairs),
            refinement_policy=(
                None if refinement_policy is None else str(refinement_policy).strip()
            ),
            require_sealed_inputs=require_sealed_inputs,
        )

    def resolve_file(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"fidelity batch input does not exist: {candidate}")
        return candidate

    def resolve_trace(self, value: str) -> ResolvedTrace:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.path.parent / candidate
        return _resolve_trace_evidence(
            candidate.resolve(),
            require_sealed=self.require_sealed_inputs,
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrace:
    path: Path
    artifact_root: Path | None = None
    artifact_manifest: Path | None = None
    artifact_schema: str | None = None
    artifact_manifest_fingerprint: str | None = None
    sealed_trace_sha256: str | None = None

    @property
    def sealed(self) -> bool:
        return self.artifact_manifest is not None


def _manifest_trace_path(root: Path, manifest: dict[str, Any]) -> Path:
    traces = [
        str(item["path"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and str(item["path"]).endswith("events.jsonl")
    ]
    if len(traces) != 1:
        raise ValueError(
            "sealed fidelity input directory must contain exactly one events.jsonl; "
            f"found {len(traces)} under {root}"
        )
    path = (root / traces[0]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"sealed event trace does not exist: {path}")
    return path


def _nearest_sealed_artifact(trace: Path) -> ResolvedTrace | None:
    for root in (trace.parent, *trace.parents):
        manifest_path = root / "artifact-manifest.json"
        if not manifest_path.is_file():
            continue
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("files"), list):
            continue
        try:
            relative = trace.relative_to(root).as_posix()
        except ValueError:
            continue
        entries = {
            str(item["path"]): item
            for item in raw["files"]
            if isinstance(item, dict) and item.get("path") is not None
        }
        entry = entries.get(relative)
        if entry is None:
            continue
        verification = verify_artifact(root)
        return ResolvedTrace(
            path=trace,
            artifact_root=root,
            artifact_manifest=manifest_path,
            artifact_schema=str(verification.get("schema")),
            artifact_manifest_fingerprint=str(verification["manifest_fingerprint"]),
            sealed_trace_sha256=str(entry["sha256"]),
        )
    return None


def _resolve_trace_evidence(candidate: Path, *, require_sealed: bool) -> ResolvedTrace:
    if candidate.is_dir() and (candidate / "artifact-manifest.json").is_file():
        verification = verify_artifact(candidate)
        manifest_path = candidate / "artifact-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trace = _manifest_trace_path(candidate, manifest)
        relative = trace.relative_to(candidate).as_posix()
        entries = {str(item["path"]): item for item in manifest["files"]}
        resolved = ResolvedTrace(
            path=trace,
            artifact_root=candidate,
            artifact_manifest=manifest_path,
            artifact_schema=str(verification.get("schema")),
            artifact_manifest_fingerprint=str(verification["manifest_fingerprint"]),
            sealed_trace_sha256=str(entries[relative]["sha256"]),
        )
    else:
        trace = resolve_event_trace(candidate)
        resolved = _nearest_sealed_artifact(trace) or ResolvedTrace(path=trace)
    if require_sealed and not resolved.sealed:
        raise ValueError(
            "fidelity batch requires sealed inputs, but no verified artifact manifest "
            f"covers event trace: {resolved.path}"
        )
    return resolved


def _pair_sample(sample: FidelitySample, pair_id: str) -> FidelitySample:
    metadata = dict(sample.metadata)
    metadata["pair_id"] = pair_id
    return FidelitySample(
        predicted=sample.predicted,
        observed=sample.observed,
        lower=sample.lower,
        upper=sample.upper,
        uncertainty=sample.uncertainty,
        metadata=metadata,
    )


def _aggregate_metric(
    pairs: tuple[tuple[str, TraceFidelityReport], ...],
    field: str,
) -> TraceMetricReport:
    reports = tuple(getattr(report, field) for _, report in pairs)
    metric_names = {report.metric for report in reports}
    if len(metric_names) != 1:
        raise ValueError(f"cannot aggregate inconsistent fidelity metric names: {metric_names}")
    samples = tuple(
        _pair_sample(sample, pair_id)
        for pair_id, report in pairs
        for sample in getattr(report, field).samples
    )
    return TraceMetricReport(
        metric=reports[0].metric,
        matched=sum(report.matched for report in reports),
        real_only=sum(report.real_only for report in reports),
        twin_only=sum(report.twin_only for report in reports),
        summary=None if not samples else summarize_fidelity(samples),
        samples=samples,
    )


def aggregate_trace_fidelity_reports(
    reports: tuple[tuple[str, TraceFidelityReport], ...],
) -> TraceFidelityReport:
    """Aggregate matched Real/Twin run reports without losing run identity."""

    if not reports:
        raise ValueError("fidelity aggregation requires at least one report")
    return TraceFidelityReport(
        **{field: _aggregate_metric(reports, field) for field in _METRIC_FIELDS}
    )


def _copy_trace_evidence(
    recorder: ResultRecorder,
    trace: ResolvedTrace,
    destination: str,
    manifest_destination: str,
) -> tuple[Path, dict[str, Any] | None]:
    copied = recorder.copy(trace.path, destination)
    copied_sha = recorder.sha256(copied)
    if trace.sealed_trace_sha256 is not None and copied_sha != trace.sealed_trace_sha256:
        raise RuntimeError("sealed fidelity input trace changed while it was being staged")
    source_artifact = None
    if trace.artifact_manifest is not None:
        copied_manifest = recorder.copy(trace.artifact_manifest, manifest_destination)
        manifest = json.loads(copied_manifest.read_text(encoding="utf-8"))
        if manifest.get("manifest_fingerprint") != trace.artifact_manifest_fingerprint:
            raise RuntimeError(
                "sealed fidelity input artifact manifest changed while it was being staged"
            )
        if trace.artifact_root is None:
            raise RuntimeError("sealed fidelity trace is missing its artifact root")
        relative = trace.path.relative_to(trace.artifact_root).as_posix()
        entries = {
            str(item["path"]): item
            for item in manifest.get("files", [])
            if isinstance(item, dict) and item.get("path") is not None
        }
        entry = entries.get(relative)
        if entry is None or entry.get("sha256") != copied_sha:
            raise RuntimeError(
                "copied source artifact manifest does not seal the staged event trace"
            )
        source_artifact = {
            "schema": trace.artifact_schema,
            "manifest_fingerprint": trace.artifact_manifest_fingerprint,
        }
    return copied, source_artifact


def _write_residual_csv(output: Path, diagnosis: dict[str, Any]) -> None:
    fields = (
        "pair_id",
        "metric_key",
        "metric",
        "entity",
        "alignment_key",
        "predicted",
        "observed",
        "residual",
        "absolute_error",
        "normalized_error",
        "relative_error",
    )
    with (output / "residuals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in diagnosis["residuals"]:
            item = {field: row.get(field) for field in fields}
            item["alignment_key"] = json.dumps(
                item["alignment_key"], separators=(",", ":")
            )
            writer.writerow(item)


def run_fidelity_batch(
    manifest: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Compare and aggregate a frozen set of matched Real/Twin event traces.

    Inputs are first copied into a private staging directory. All statistics are
    computed from those copied bytes, then the sealed staging directory is
    atomically promoted to the requested output. The durable artifact therefore
    always contains exactly the evidence that produced its statistics.
    """

    spec = FidelityBatchSpec.load(manifest)
    target = require_fresh_artifact_directory(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rmdir()
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    )
    try:
        recorder = ResultRecorder(stage)
        copied_manifest = recorder.copy(spec.path, "inputs/fidelity-batch.yaml")
        if recorder.sha256(copied_manifest) != spec.source_sha256:
            raise RuntimeError("fidelity batch manifest changed while it was being staged")

        refinement_policy = None
        if spec.refinement_policy is not None:
            policy_path = spec.resolve_file(spec.refinement_policy)
            copied_policy = recorder.copy(policy_path, "inputs/refinement-policy.yaml")
            # Parse the staged bytes before any residual evidence is evaluated.
            from .refinement import RefinementPolicy

            RefinementPolicy.load(copied_policy)
            refinement_policy = {
                "reference": spec.refinement_policy,
                "sha256": recorder.sha256(copied_policy),
            }

        pair_payloads: list[dict[str, Any]] = []
        paired_reports: list[tuple[str, TraceFidelityReport]] = []
        for pair in spec.pairs:
            source_real = spec.resolve_trace(pair.real)
            source_twin = spec.resolve_trace(pair.twin)
            real, real_source_artifact = _copy_trace_evidence(
                recorder,
                source_real,
                f"inputs/{pair.id}/real-events.jsonl",
                f"inputs/{pair.id}/real-source-artifact-manifest.json",
            )
            twin, twin_source_artifact = _copy_trace_evidence(
                recorder,
                source_twin,
                f"inputs/{pair.id}/twin-events.jsonl",
                f"inputs/{pair.id}/twin-source-artifact-manifest.json",
            )
            report = compare_event_traces(load_event_trace(real), load_event_trace(twin))
            paired_reports.append((pair.id, report))
            pair_payloads.append(
                {
                    "id": pair.id,
                    "real": pair.real,
                    "twin": pair.twin,
                    "real_sha256": recorder.sha256(real),
                    "twin_sha256": recorder.sha256(twin),
                    "real_source_artifact": real_source_artifact,
                    "twin_source_artifact": twin_source_artifact,
                    "report": report.to_dict(),
                }
            )

        aggregate = aggregate_trace_fidelity_reports(tuple(paired_reports))
        diagnosis = diagnose_trace_fidelity(aggregate)
        payload = {
            "schema": FIDELITY_BATCH_SCHEMA,
            "name": spec.name,
            "pairs": pair_payloads,
            "pair_count": len(pair_payloads),
            "require_sealed_inputs": spec.require_sealed_inputs,
            "refinement_policy": refinement_policy,
            "aggregate": aggregate.to_dict(),
            "diagnosis": diagnosis,
        }
        recorder.write_json("fidelity-batch.json", payload)
        recorder.write_json("fidelity-diagnosis.json", diagnosis)
        _write_residual_csv(stage, diagnosis)
        manifest_payload = seal_artifact(
            stage,
            schema=FIDELITY_BATCH_SCHEMA,
            identity={
                "name": spec.name,
                "pairs": len(pair_payloads),
                "samples": diagnosis["samples"],
                "dominant_metric": diagnosis["dominant_metric"],
            },
        )
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        **payload,
        "artifact_manifest_fingerprint": manifest_payload["manifest_fingerprint"],
        "output": str(target),
    }
