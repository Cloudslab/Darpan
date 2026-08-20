from __future__ import annotations

import json
from pathlib import Path

import pytest

from darpan.core.event import Event, EventKind
from darpan.experiment.artifact import seal_artifact, verify_artifact
from darpan.experiment.fidelity_batch import (
    aggregate_trace_fidelity_reports,
    run_fidelity_batch,
)
from darpan.experiment.fidelity_diagnosis import diagnose_trace_fidelity
from darpan.experiment.trace_fidelity import compare_event_traces


def _completed(event_id: str, component: str, duration: float) -> Event:
    return Event(
        id=event_id,
        kind=EventKind.COMPONENT_COMPLETED,
        event_time=duration,
        source="test",
        subject=component,
        payload={
            "application_id": "app",
            "component_id": component,
            "duration_s": duration,
        },
    )


def _event(event_id: str, duration: float) -> str:
    return json.dumps(
        {
            "id": event_id,
            "kind": "component.completed",
            "event_time": duration,
            "source": "test",
            "subject": "app:task",
            "payload": {
                "application_id": "app",
                "component_id": "task",
                "duration_s": duration,
            },
        }
    )


def test_aggregate_trace_fidelity_preserves_pair_identity() -> None:
    first = compare_event_traces(
        (_completed("r1", "task", 4),),
        (_completed("t1", "task", 3),),
    )
    second = compare_event_traces(
        (_completed("r2", "task", 6),),
        (_completed("t2", "task", 4),),
    )
    aggregate = aggregate_trace_fidelity_reports((("seed-1", first), ("seed-2", second)))
    assert aggregate.execution_duration.matched == 2
    assert aggregate.execution_duration.summary is not None
    assert aggregate.execution_duration.summary.mean_absolute_error == 1.5
    assert [
        sample.metadata["pair_id"] for sample in aggregate.execution_duration.samples
    ] == ["seed-1", "seed-2"]
    diagnosis = diagnose_trace_fidelity(aggregate)
    assert diagnosis["pair_ranking"][0]["pair_id"] == "seed-2"
    assert diagnosis["pair_ranking"][0]["normalized_error_sum"] == 1 / 3


def test_run_fidelity_batch_copies_and_seals_all_input_traces(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    for pair_id, real_duration, twin_duration in (
        ("seed-201", 4.0, 3.0),
        ("seed-202", 5.0, 4.5),
    ):
        (traces / f"{pair_id}-real.jsonl").write_text(
            _event(f"{pair_id}-real", real_duration) + "\n",
            encoding="utf-8",
        )
        (traces / f"{pair_id}-twin.jsonl").write_text(
            _event(f"{pair_id}-twin", twin_duration) + "\n",
            encoding="utf-8",
        )
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: physical-reference\n"
        "pairs:\n"
        "  - id: seed-201\n"
        "    real: traces/seed-201-real.jsonl\n"
        "    twin: traces/seed-201-twin.jsonl\n"
        "  - id: seed-202\n"
        "    real: traces/seed-202-real.jsonl\n"
        "    twin: traces/seed-202-twin.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch"
    payload = run_fidelity_batch(manifest, output)
    assert payload["pair_count"] == 2
    assert payload["diagnosis"]["samples"] == 2
    assert payload["diagnosis"]["dominant_metric"] == "execution_duration"
    assert (output / "inputs/seed-201/real-events.jsonl").is_file()
    assert (output / "inputs/seed-202/twin-events.jsonl").is_file()
    assert verify_artifact(output)["verified"] is True


def test_fidelity_batch_can_require_and_preserve_sealed_source_evidence(
    tmp_path: Path,
) -> None:
    for name, duration in (("real", 4.0), ("twin", 3.0)):
        artifact = tmp_path / name
        run = artifact / "run-0001"
        run.mkdir(parents=True)
        (run / "events.jsonl").write_text(
            _event(name, duration) + "\n",
            encoding="utf-8",
        )
        seal_artifact(artifact, schema="darpan.experiment-artifact/v1")

    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: sealed-inputs\n"
        "require_sealed_inputs: true\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real\n"
        "    twin: twin\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch"
    payload = run_fidelity_batch(manifest, output)
    pair = payload["pairs"][0]
    assert payload["require_sealed_inputs"] is True
    assert pair["real_source_artifact"]["schema"] == "darpan.experiment-artifact/v1"
    assert pair["twin_source_artifact"]["manifest_fingerprint"]
    assert (output / "inputs/seed-1/real-source-artifact-manifest.json").is_file()
    assert (output / "inputs/seed-1/twin-source-artifact-manifest.json").is_file()
    assert verify_artifact(output)["verified"] is True


def test_fidelity_batch_rejects_tampered_sealed_source_trace(tmp_path: Path) -> None:
    for name, duration in (("real", 4.0), ("twin", 3.0)):
        artifact = tmp_path / name
        artifact.mkdir()
        (artifact / "events.jsonl").write_text(
            _event(name, duration) + "\n",
            encoding="utf-8",
        )
        seal_artifact(artifact, schema="darpan.experiment-artifact/v1")
    (tmp_path / "real/events.jsonl").write_text(
        _event("tampered", 40.0) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: tampered-source\n"
        "require_sealed_inputs: true\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real\n"
        "    twin: twin\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="artifact integrity verification failed"):
        run_fidelity_batch(manifest, output)
    assert not output.exists()


def test_fidelity_batch_rejects_unsealed_trace_when_required(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"
    real.write_text(_event("real", 1.0) + "\n", encoding="utf-8")
    twin.write_text(_event("twin", 1.0) + "\n", encoding="utf-8")
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: require-sealed\n"
        "require_sealed_inputs: true\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real.jsonl\n"
        "    twin: twin.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="requires sealed inputs"):
        run_fidelity_batch(manifest, output)
    assert not output.exists()


def test_fidelity_batch_does_not_publish_partial_artifact_on_invalid_trace(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"
    real.write_text(_event("real", 1.0) + "\n", encoding="utf-8")
    twin.write_text("{not-json}\n", encoding="utf-8")
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: invalid-trace\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real.jsonl\n"
        "    twin: twin.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="invalid event trace"):
        run_fidelity_batch(manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".batch.staging-*"))
