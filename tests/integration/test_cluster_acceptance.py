from __future__ import annotations

import asyncio

from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.acceptance import accept_cluster
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


def test_cluster_acceptance_runs_data_lifecycle_and_scale_on_real_agents():
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
            system = SystemSpec(
                nodes=(
                    NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
                    NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
                ),
                links=(LinkSpec("edge-fog", "edge", "fog"),),
            )
            report = await accept_cluster(
                inventory,
                system,
                source_node_id="edge",
                target_node_id="fog",
                command=("python3", "-c", "import time; time.sleep(3600)"),
                cpu_request=0.1,
                timeout_s=5,
            )
            assert report.ready_for_study is True
            assert report.preflight.ready is True
            assert report.lifecycle is not None and report.lifecycle.ready is True
            scale = next(step for step in report.steps if step.name == "scale_out_in")
            assert scale.ok is True
            assert scale.details["source_active_after_scale_out"] == 1
            assert scale.details["target_active_after_scale_out"] == 1
            assert scale.details["target_active_after_scale_in"] == 0
            assert scale.details["application_success"] is True
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())
