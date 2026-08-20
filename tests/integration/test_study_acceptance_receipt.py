from __future__ import annotations

import asyncio
from pathlib import Path

import darpan.runtime.real.agent as agent_module
from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.artifact import seal_artifact, verify_artifact
from darpan.experiment.recorder import ResultRecorder
from darpan.experiment.study import validate_physical_study
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.acceptance import accept_cluster, build_deployment_receipt
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


def _write_system(path: Path) -> None:
    path.write_text(
        "nodes:\n"
        "  - id: edge\n"
        "    resources:\n"
        "      - name: cpu\n"
        "        capacity: 1\n"
        "  - id: fog\n"
        "    resources:\n"
        "      - name: cpu\n"
        "        capacity: 1\n"
        "links:\n"
        "  - id: edge-fog\n"
        "    source: edge\n"
        "    target: fog\n",
        encoding="utf-8",
    )


def _write_acceptance_artifact(
    root: Path,
    *,
    report,
    cluster: Path,
    system: Path,
) -> None:
    recorder = ResultRecorder(root)
    receipt = build_deployment_receipt(report, cluster=cluster, system=system)
    recorder.write_json("cluster-acceptance.json", report.to_dict())
    recorder.write_json("deployment-receipt.json", receipt.to_dict())
    recorder.copy(cluster, "inputs/cluster.yaml")
    recorder.copy(system, "inputs/system.yaml")
    seal_artifact(
        root,
        schema="darpan.cluster-acceptance/v1",
        identity={
            "ready_for_study": report.ready_for_study,
            "deployment_fingerprint": receipt.deployment_fingerprint,
        },
    )


def test_physical_study_requires_matching_live_deployment_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    async def run() -> None:
        source = AgentServer(
            "edge",
            host="127.0.0.1",
            port=0,
            allow_network_probe=True,
            allow_artifact_forward=True,
        )
        target = AgentServer("fog", host="127.0.0.1", port=0)
        await source.start()
        await target.start()
        try:
            cluster = tmp_path / "cluster.yaml"
            cluster.write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    host: 127.0.0.1\n"
                f"    port: {source.port}\n"
                "    network_probe: true\n"
                "    artifact_forward: true\n"
                "  - id: fog\n"
                "    host: 127.0.0.1\n"
                f"    port: {target.port}\n",
                encoding="utf-8",
            )
            system_path = tmp_path / "system.yaml"
            _write_system(system_path)
            system = SystemSpec(
                nodes=(
                    NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
                    NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
                ),
                links=(LinkSpec("edge-fog", "edge", "fog"),),
            )
            inventory = ClusterInventory(
                (
                    ClusterNode(
                        "edge",
                        "127.0.0.1",
                        source.port,
                        network_probe=True,
                        artifact_forward=True,
                    ),
                    ClusterNode("fog", "127.0.0.1", target.port),
                )
            )
            accepted = await accept_cluster(
                inventory,
                system,
                source_node_id="edge",
                target_node_id="fog",
                command=("python3", "-c", "import time; time.sleep(3600)"),
                cpu_request=0.1,
                timeout_s=5,
            )
            assert accepted.ready_for_study is True
            receipt_root = tmp_path / "acceptance"
            _write_acceptance_artifact(
                receipt_root,
                report=accepted,
                cluster=cluster,
                system=system_path,
            )
            assert verify_artifact(receipt_root)["verified"] is True

            campaign = tmp_path / "campaign.yaml"
            campaign.write_text(
                "name: receipt-ready\n"
                "output: campaign-results\n"
                "jobs:\n"
                "  - id: preflight\n"
                "    kind: cluster_preflight\n"
                "    cluster: cluster.yaml\n"
                "    system: system.yaml\n",
                encoding="utf-8",
            )
            readiness = await validate_physical_study(
                campaign,
                acceptance_receipt=receipt_root,
            )
            assert readiness.ready is True
            receipt_evidence = readiness.checks[0]["deployment_receipt"]
            assert receipt_evidence["deployment_fingerprint"]

            original_snapshot = agent_module.environment_snapshot

            def changed_environment():
                payload = dict(original_snapshot())
                payload["hostname"] = "changed-after-acceptance"
                return payload

            monkeypatch.setattr(agent_module, "environment_snapshot", changed_environment)
            drifted = await validate_physical_study(
                campaign,
                acceptance_receipt=receipt_root,
            )
            assert drifted.ready is False
            assert any("environment fingerprint" in error for error in drifted.errors)
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())
