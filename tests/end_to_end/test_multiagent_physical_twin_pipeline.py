from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import yaml

from darpan.cli.run import run_experiment_spec
from darpan.experiment.artifact import verify_artifact
from darpan.experiment.fidelity_batch import run_fidelity_batch
from darpan.experiment.refinement import export_refinement_decision
from darpan.experiment.study_fidelity import build_study_fidelity_manifest
from darpan.experiment.study_run import StudyRunner, StudySpec, verify_study
from darpan.runtime.real.agent import AgentServer


def _write(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def test_multiagent_physical_study_to_twin_refinement_pipeline(tmp_path: Path) -> None:
    async def run() -> None:
        agents = {
            node: AgentServer(
                node,
                host="127.0.0.1",
                port=0,
                workspace_root=tmp_path / f"agent-{node}",
                allow_network_probe=True,
                allow_artifact_forward=True,
            )
            for node in ("edge", "fog", "cloud")
        }
        for agent in agents.values():
            await agent.start()
        try:
            _write(
                tmp_path / "cluster.yaml",
                f"""
                nodes:
                  - id: edge
                    host: 127.0.0.1
                    port: {agents['edge'].port}
                    network_probe: true
                    artifact_forward: true
                  - id: fog
                    host: 127.0.0.1
                    port: {agents['fog'].port}
                    network_probe: true
                    artifact_forward: true
                  - id: cloud
                    host: 127.0.0.1
                    port: {agents['cloud'].port}
                    network_probe: true
                    artifact_forward: true
                """,
            )
            _write(
                tmp_path / "system.yaml",
                """
                nodes:
                  - id: edge
                    resources: {cpu: 1, memory: 1}
                  - id: fog
                    resources: {cpu: 1, memory: 1}
                  - id: cloud
                    resources: {cpu: 1, memory: 1}
                links:
                  - id: edge-fog
                    source: edge
                    target: fog
                    latency_ms: 1
                    bandwidth_mbps: 100
                  - id: fog-cloud
                    source: fog
                    target: cloud
                    latency_ms: 1
                    bandwidth_mbps: 100
                """,
            )
            _write(
                tmp_path / "app.yaml",
                """
                id: multi-agent-artifact
                components:
                  source:
                    command:
                      - python3
                      - -c
                      - >-
                        from pathlib import Path;
                        Path('payload.bin').write_bytes(b'x' * 65536)
                    resources: {cpu: 0.1}
                    work_units: 0.02
                  sink:
                    command:
                      - python3
                      - -c
                      - >-
                        from pathlib import Path;
                        assert len(Path('input.bin').read_bytes()) == 65536
                    resources: {cpu: 0.1}
                    work_units: 0.02
                flows:
                  - source: source
                    target: sink
                    data_size_bytes: 65536
                    artifact: payload.bin
                    target_path: input.bin
                """,
            )
            _write(
                tmp_path / "policies.py",
                """
                from darpan.core.action import Action

                class _Pinned:
                    sink_node = "fog"

                    def decide(self, state, trigger):
                        for item in sorted(state.ready_components(), key=lambda value: value.id):
                            target = "edge" if item.component_id == "source" else self.sink_node
                            return Action.place(item.id, target, source="multi-agent-e2e")
                        return None

                class EdgeFog(_Pinned):
                    sink_node = "fog"

                class EdgeCloud(_Pinned):
                    sink_node = "cloud"
                """,
            )
            for runtime in ("real", "twin"):
                for label, policy in (
                    ("edge-fog", "EdgeFog"),
                    ("edge-cloud", "EdgeCloud"),
                ):
                    cluster = "cluster: cluster.yaml\n" if runtime == "real" else ""
                    experiment = (
                        "system: system.yaml\n"
                        "application: app.yaml\n"
                        f"runtime: {runtime}\n"
                        f"{cluster}"
                        f"policy: policies.py:{policy}\n"
                        "repeat: 1\n"
                        "seed: 301\n"
                        "timeout: 20\n"
                        "metrics: [application_latency_s, application_success_rate]\n"
                    )
                    _write(tmp_path / f"{runtime}-{label}.yaml", experiment)
            _write(
                tmp_path / "real-campaign.yaml",
                """
                name: multi-agent-real
                output: ignored-real
                seed: 301
                repeat: 1
                jobs:
                  - id: acceptance
                    kind: cluster_acceptance
                    cluster: cluster.yaml
                    system: system.yaml
                    source: edge
                    target: fog
                    python_command: python3
                    cpu: 0.1
                    timeout_s: 5
                  - id: placement
                    kind: paired
                    baseline: real-edge-fog.yaml
                    candidate: real-edge-cloud.yaml
                    metric: application_latency_s
                    higher_is_better: false
                    require_success: true
                """,
            )
            _write(
                tmp_path / "twin-campaign.yaml",
                """
                name: multi-agent-twin
                output: ignored-twin
                seed: 301
                repeat: 1
                jobs:
                  - id: placement
                    kind: paired
                    baseline: twin-edge-fog.yaml
                    candidate: twin-edge-cloud.yaml
                    metric: application_latency_s
                    higher_is_better: false
                    require_success: true
                """,
            )
            _write(
                tmp_path / "real-study.yaml",
                """
                name: multi-agent-real-study
                version: 1
                campaign: real-campaign.yaml
                output: real-study-results
                readiness:
                  mode: required
                  exercise_data_plane: true
                  max_clock_offset_s: 1.0
                require_all_jobs_success: true
                """,
            )
            _write(
                tmp_path / "twin-study.yaml",
                """
                name: multi-agent-twin-study
                version: 1
                campaign: twin-campaign.yaml
                output: twin-study-results
                readiness: {mode: skip}
                require_all_jobs_success: true
                """,
            )

            async def execute(spec):
                return await run_experiment_spec(spec, emit_output=False)

            real = await StudyRunner(
                StudySpec.load(tmp_path / "real-study.yaml"), execute=execute
            ).run()
            twin = await StudyRunner(
                StudySpec.load(tmp_path / "twin-study.yaml"), execute=execute
            ).run()
            assert real["complete"] is True
            assert real["campaign"]["failed_jobs"] == 0
            assert twin["complete"] is True
            assert verify_study(tmp_path / "real-study-results")["verified"] is True
            assert verify_study(tmp_path / "twin-study-results")["verified"] is True

            _write(
                tmp_path / "refinement-policy.yaml",
                """
                schema: darpan.twin-refinement-policy/v1
                minimum_total_samples: 1
                maximum_uncovered_error_contribution: 1.0
                max_candidates: 1
                rules:
                  - metric: queue_delay
                    target: twin.models.queue
                    maximum_acceptable_mae: 1000000
                    minimum_samples: 1
                """,
            )
            mapping = {
                "schema": "darpan.study-fidelity-map/v1",
                "name": "multi-agent-physical-vs-twin",
                "real_study": "real-study-results",
                "twin_study": "twin-study-results",
                "refinement_policy": "refinement-policy.yaml",
                "pairs": [
                    {
                        "id": "edge-fog",
                        "real": {"job": "placement", "arm": "baseline"},
                        "twin": {"job": "placement", "arm": "baseline"},
                    },
                    {
                        "id": "edge-cloud",
                        "real": {"job": "placement", "arm": "candidate"},
                        "twin": {"job": "placement", "arm": "candidate"},
                    },
                ],
            }
            (tmp_path / "fidelity-map.yaml").write_text(
                yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8"
            )
            planned = build_study_fidelity_manifest(
                tmp_path / "fidelity-map.yaml",
                tmp_path / "fidelity-batch.yaml",
            )
            assert planned["pair_count"] == 2
            batch_yaml = yaml.safe_load(
                (tmp_path / "fidelity-batch.yaml").read_text(encoding="utf-8")
            )
            assert batch_yaml["require_sealed_inputs"] is True
            assert batch_yaml["pairs"][0]["source"]["seed"] == 301

            batch = run_fidelity_batch(
                tmp_path / "fidelity-batch.yaml",
                tmp_path / "fidelity-results",
            )
            assert batch["pair_count"] == 2
            assert batch["diagnosis"]["samples"] > 0
            assert verify_artifact(tmp_path / "fidelity-results")["verified"] is True

            decision = export_refinement_decision(
                tmp_path / "fidelity-results",
                None,
                tmp_path / "refinement-results",
            )
            assert decision["decision"] == "hold"
            assert decision["policy_predeclared_with_evidence"] is True
            assert verify_artifact(tmp_path / "refinement-results")["verified"] is True
        finally:
            for agent in reversed(tuple(agents.values())):
                await agent.close()

    asyncio.run(run())
