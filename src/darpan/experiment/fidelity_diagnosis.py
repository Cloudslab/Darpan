"""Residual attribution for Real-vs-Twin trace fidelity reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifact import require_fresh_artifact_directory, seal_artifact
from .recorder import ResultRecorder
from .trace_fidelity import TraceFidelityReport

DIAGNOSIS_SCHEMA = "darpan.fidelity-diagnosis/v2"
LEGACY_DIAGNOSIS_SCHEMA = "darpan.fidelity-diagnosis/v1"


def _entity(metric_key: str, alignment: list[Any]) -> str:
    if metric_key == "transfer_duration" and len(alignment) >= 3:
        return f"{alignment[0]}->{alignment[1]}:{alignment[2]}"
    if metric_key == "route_change" and len(alignment) >= 3:
        return f"{alignment[1]}->{alignment[2]}"
    if len(alignment) >= 2:
        return str(alignment[1])
    return "unknown"


def diagnose_trace_fidelity(report: TraceFidelityReport) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_reports = {
        "application_latency": report.application_latency,
        "execution_duration": report.execution_duration,
        "artifact_size": report.artifact_size,
        "queue_delay": report.queue_delay,
        "transfer_duration": report.transfer_duration,
        "migration_downtime": report.migration_downtime,
        "restart_downtime": report.restart_downtime,
        "retry_downtime": report.retry_downtime,
        "scale_convergence": report.scale_convergence,
        "route_change": report.route_change,
    }
    for metric_key, item in metric_reports.items():
        for sample in item.samples:
            alignment = list(sample.metadata.get("alignment_key", ()))
            residual = float(sample.observed) - float(sample.predicted)
            scale = max(abs(float(sample.observed)), abs(float(sample.predicted)), 1e-12)
            normalized_error = abs(residual) / scale
            rows.append(
                {
                    "pair_id": sample.metadata.get("pair_id"),
                    "metric_key": metric_key,
                    "metric": item.metric,
                    "entity": _entity(metric_key, alignment),
                    "alignment_key": alignment,
                    "predicted": float(sample.predicted),
                    "observed": float(sample.observed),
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "normalized_error": normalized_error,
                    "relative_error": sample.relative_error,
                }
            )

    total_normalized_error = sum(float(row["normalized_error"]) for row in rows)
    by_metric: dict[str, dict[str, Any]] = {}
    by_entity: dict[tuple[str, str], dict[str, Any]] = {}
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        metric = str(row["metric_key"])
        bucket = by_metric.setdefault(
            metric,
            {
                "metric_key": metric,
                "metric": row["metric"],
                "samples": 0,
                "absolute_error_sum": 0.0,
                "normalized_error_sum": 0.0,
                "max_absolute_error": 0.0,
            },
        )
        error = float(row["absolute_error"])
        normalized_error = float(row["normalized_error"])
        bucket["samples"] += 1
        bucket["absolute_error_sum"] += error
        bucket["normalized_error_sum"] += normalized_error
        bucket["max_absolute_error"] = max(bucket["max_absolute_error"], error)
        entity_key = (metric, str(row["entity"]))
        entity = by_entity.setdefault(
            entity_key,
            {
                "metric_key": metric,
                "metric": row["metric"],
                "entity": row["entity"],
                "samples": 0,
                "absolute_error_sum": 0.0,
                "normalized_error_sum": 0.0,
            },
        )
        entity["samples"] += 1
        entity["absolute_error_sum"] += error
        entity["normalized_error_sum"] += normalized_error
        pair_id = row.get("pair_id")
        if pair_id is not None:
            pair = by_pair.setdefault(
                str(pair_id),
                {
                    "pair_id": str(pair_id),
                    "samples": 0,
                    "normalized_error_sum": 0.0,
                },
            )
            pair["samples"] += 1
            pair["normalized_error_sum"] += normalized_error

    metric_ranking = []
    for item in by_metric.values():
        item["mean_absolute_error"] = item["absolute_error_sum"] / item["samples"]
        item["mean_normalized_error"] = item["normalized_error_sum"] / item["samples"]
        item["error_contribution"] = (
            0.0
            if total_normalized_error <= 1e-12
            else item["normalized_error_sum"] / total_normalized_error
        )
        metric_ranking.append(item)
    metric_ranking.sort(
        key=lambda item: (-item["normalized_error_sum"], item["metric_key"])
    )

    entity_ranking = []
    for item in by_entity.values():
        item["mean_absolute_error"] = item["absolute_error_sum"] / item["samples"]
        item["mean_normalized_error"] = item["normalized_error_sum"] / item["samples"]
        item["error_contribution"] = (
            0.0
            if total_normalized_error <= 1e-12
            else item["normalized_error_sum"] / total_normalized_error
        )
        entity_ranking.append(item)
    entity_ranking.sort(
        key=lambda item: (
            -item["normalized_error_sum"],
            item["metric_key"],
            item["entity"],
        )
    )
    pair_ranking = []
    for item in by_pair.values():
        item["mean_normalized_error"] = item["normalized_error_sum"] / item["samples"]
        item["error_contribution"] = (
            0.0
            if total_normalized_error <= 1e-12
            else item["normalized_error_sum"] / total_normalized_error
        )
        pair_ranking.append(item)
    pair_ranking.sort(
        key=lambda item: (
            -item["mean_normalized_error"],
            -item["normalized_error_sum"],
            item["pair_id"],
        )
    )
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "samples": len(rows),
        "total_normalized_error": total_normalized_error,
        "dominant_metric": None if not metric_ranking else metric_ranking[0]["metric_key"],
        "metric_ranking": metric_ranking,
        "entity_ranking": entity_ranking,
        "pair_ranking": pair_ranking,
        "residuals": rows,
    }


def export_fidelity_diagnosis(
    report: TraceFidelityReport,
    output_directory: str | Path,
    *,
    real_events: str | Path | None = None,
    twin_events: str | Path | None = None,
) -> dict[str, Any]:
    output = require_fresh_artifact_directory(output_directory)
    recorder = ResultRecorder(output)
    diagnosis = diagnose_trace_fidelity(report)
    recorder.write_json("fidelity-diagnosis.json", diagnosis)
    residuals = diagnosis["residuals"]
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
        for row in residuals:
            item = dict(row)
            item["alignment_key"] = json.dumps(item["alignment_key"], separators=(",", ":"))
            writer.writerow(item)
    recorder.write_json("fidelity-report.json", asdict(report))
    if real_events is not None:
        recorder.copy(real_events, "inputs/real-events.jsonl")
    if twin_events is not None:
        recorder.copy(twin_events, "inputs/twin-events.jsonl")
    seal_artifact(
        output,
        schema=DIAGNOSIS_SCHEMA,
        identity={
            "samples": diagnosis["samples"],
            "dominant_metric": diagnosis["dominant_metric"],
        },
    )
    return diagnosis
