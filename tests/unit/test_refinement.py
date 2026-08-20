from __future__ import annotations

import json
from pathlib import Path

import pytest

from darpan.core.event import Event, EventKind
from darpan.experiment.artifact import verify_artifact
from darpan.experiment.fidelity_diagnosis import export_fidelity_diagnosis
from darpan.experiment.refinement import (
    RefinementPolicy,
    decide_twin_refinement,
    export_refinement_decision,
)
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


def _policy(path: Path, *, minimum_total_samples: int = 2) -> RefinementPolicy:
    path.write_text(
        "schema: darpan.twin-refinement-policy/v1\n"
        f"minimum_total_samples: {minimum_total_samples}\n"
        "maximum_uncovered_error_contribution: 0.20\n"
        "max_candidates: 2\n"
        "rules:\n"
        "  - metric: execution_duration\n"
        "    target: twin.models.execution\n"
        "    minimum_samples: 2\n"
        "    maximum_acceptable_mae: 0.5\n"
        "    minimum_error_contribution: 0.1\n"
        "    required: true\n",
        encoding="utf-8",
    )
    return RefinementPolicy.load(path)


def test_refinement_policy_authorizes_only_predeclared_threshold_violation(
    tmp_path: Path,
) -> None:
    policy = _policy(tmp_path / "policy.yaml")
    diagnosis = {
        "samples": 3,
        "dominant_metric": "execution_duration",
        "metric_ranking": [
            {
                "metric_key": "execution_duration",
                "samples": 3,
                "mean_absolute_error": 1.25,
                "error_contribution": 0.81,
            },
            {
                "metric_key": "queue_delay",
                "samples": 3,
                "mean_absolute_error": 0.1,
                "error_contribution": 0.19,
            },
        ],
    }
    decision = decide_twin_refinement(diagnosis, policy)
    assert decision["decision"] == "refine"
    assert decision["authorized_targets"] == ["twin.models.execution"]
    assert decision["candidate_metrics"] == ["execution_duration"]


def test_refinement_gate_requires_enough_evidence_and_flags_uncovered_residual(
    tmp_path: Path,
) -> None:
    policy = _policy(tmp_path / "policy.yaml", minimum_total_samples=4)
    too_small = {
        "samples": 2,
        "dominant_metric": "execution_duration",
        "metric_ranking": [
            {
                "metric_key": "execution_duration",
                "samples": 2,
                "mean_absolute_error": 2.0,
                "error_contribution": 1.0,
            }
        ],
    }
    assert decide_twin_refinement(too_small, policy)["decision"] == "insufficient_evidence"

    policy = _policy(tmp_path / "policy-2.yaml", minimum_total_samples=2)
    uncovered = {
        "samples": 4,
        "dominant_metric": "route_change",
        "metric_ranking": [
            {
                "metric_key": "route_change",
                "samples": 2,
                "mean_absolute_error": 1.0,
                "error_contribution": 0.7,
            },
            {
                "metric_key": "execution_duration",
                "samples": 2,
                "mean_absolute_error": 0.1,
                "error_contribution": 0.3,
            },
        ],
    }
    assert decide_twin_refinement(uncovered, policy)["decision"] == "manual_review"


def test_export_refinement_decision_verifies_source_artifact_before_use(
    tmp_path: Path,
) -> None:
    report = compare_event_traces(
        (
            _completed("r1", "a", 4.0),
            _completed("r2", "b", 5.0),
        ),
        (
            _completed("t1", "a", 3.0),
            _completed("t2", "b", 4.0),
        ),
    )
    diagnosis_dir = tmp_path / "diagnosis"
    export_fidelity_diagnosis(report, diagnosis_dir)
    policy_path = tmp_path / "policy.yaml"
    _policy(policy_path)
    output = tmp_path / "decision"
    payload = export_refinement_decision(diagnosis_dir, policy_path, output)
    assert payload["decision"] == "refine"
    assert payload["authorized_targets"] == ["twin.models.execution"]
    assert verify_artifact(output)["verified"] is True

    diagnosis_json = diagnosis_dir / "fidelity-diagnosis.json"
    raw = json.loads(diagnosis_json.read_text(encoding="utf-8"))
    raw["samples"] = 999
    diagnosis_json.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact integrity verification failed"):
        export_refinement_decision(
            diagnosis_dir,
            policy_path,
            tmp_path / "tampered-decision",
        )
    assert not (tmp_path / "tampered-decision").exists()


def test_predeclared_batch_policy_rejects_post_hoc_policy_change(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"
    real.write_text(
        json.dumps(
            {
                "id": "r",
                "kind": "component.completed",
                "event_time": 4.0,
                "source": "test",
                "subject": "app:task",
                "payload": {
                    "application_id": "app",
                    "component_id": "task",
                    "duration_s": 4.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    twin.write_text(
        json.dumps(
            {
                "id": "t",
                "kind": "component.completed",
                "event_time": 3.0,
                "source": "test",
                "subject": "app:task",
                "payload": {
                    "application_id": "app",
                    "component_id": "task",
                    "duration_s": 3.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.yaml"
    _policy(policy_path, minimum_total_samples=1)
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: frozen-policy\n"
        "refinement_policy: policy.yaml\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real.jsonl\n"
        "    twin: twin.jsonl\n",
        encoding="utf-8",
    )
    from darpan.experiment.fidelity_batch import run_fidelity_batch

    batch = tmp_path / "batch"
    run_fidelity_batch(manifest, batch)
    changed = tmp_path / "changed-policy.yaml"
    changed.write_text(
        policy_path.read_text(encoding="utf-8").replace(
            "maximum_acceptable_mae: 0.5",
            "maximum_acceptable_mae: 5.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="differs from the policy sealed before"):
        export_refinement_decision(batch, changed, tmp_path / "decision")

    payload = export_refinement_decision(batch, None, tmp_path / "frozen-decision")
    assert payload["policy_predeclared_with_evidence"] is True


def test_artifact_size_residual_can_authorize_only_artifact_model(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    twin = tmp_path / "twin.jsonl"

    def event(event_id: str, output_bytes: int) -> str:
        return json.dumps(
            {
                "id": event_id,
                "kind": "component.completed",
                "event_time": 1.0,
                "source": "test",
                "subject": "app:task",
                "payload": {
                    "application_id": "app",
                    "component_id": "task",
                    "duration_s": 1.0,
                    "output_bytes": output_bytes,
                },
            }
        )

    real.write_text(event("real", 2_000_000) + "\n", encoding="utf-8")
    twin.write_text(event("twin", 1_000_000) + "\n", encoding="utf-8")
    policy = tmp_path / "artifact-policy.yaml"
    policy.write_text(
        "schema: darpan.twin-refinement-policy/v1\n"
        "minimum_total_samples: 1\n"
        "maximum_uncovered_error_contribution: 0.2\n"
        "max_candidates: 1\n"
        "rules:\n"
        "  - metric: artifact_size\n"
        "    target: twin.models.artifact_size\n"
        "    minimum_samples: 1\n"
        "    maximum_acceptable_mae: 100000\n"
        "    minimum_error_contribution: 0.5\n"
        "    required: true\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        "schema: darpan.fidelity-batch/v1\n"
        "name: artifact-size-gate\n"
        "refinement_policy: artifact-policy.yaml\n"
        "pairs:\n"
        "  - id: seed-1\n"
        "    real: real.jsonl\n"
        "    twin: twin.jsonl\n",
        encoding="utf-8",
    )
    from darpan.experiment.fidelity_batch import run_fidelity_batch

    batch = tmp_path / "batch"
    batch_payload = run_fidelity_batch(manifest, batch)
    assert batch_payload["diagnosis"]["dominant_metric"] == "artifact_size"
    decision = export_refinement_decision(batch, None, tmp_path / "decision")
    assert decision["decision"] == "refine"
    assert decision["authorized_targets"] == ["twin.models.artifact_size"]


def test_refinement_hold_requires_sufficient_evidence_below_threshold(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "hold-policy.yaml")
    diagnosis = {
        "samples": 3,
        "dominant_metric": "execution_duration",
        "metric_ranking": [
            {
                "metric_key": "execution_duration",
                "samples": 3,
                "mean_absolute_error": 0.25,
                "error_contribution": 1.0,
            }
        ],
    }
    decision = decide_twin_refinement(diagnosis, policy)
    assert decision["decision"] == "hold"
    assert decision["authorized_targets"] == []
