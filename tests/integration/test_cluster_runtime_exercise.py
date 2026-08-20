from __future__ import annotations

import asyncio
import sys

from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.exercise import exercise_cluster_runtime
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


def test_cluster_runtime_exercise_runs_real_place_migrate_stop_across_agents():
    async def run() -> None:
        source = AgentServer("edge", host="127.0.0.1", port=0)
        target = AgentServer("fog", host="127.0.0.1", port=0)
        await source.start()
        await target.start()
        try:
            inventory = ClusterInventory(
                (
                    ClusterNode("edge", "127.0.0.1", source.port),
                    ClusterNode("fog", "127.0.0.1", target.port),
                )
            )
            system = SystemSpec(
                nodes=(
                    NodeSpec("edge", resources=(ResourceSpec("cpu", 0.25),)),
                    NodeSpec("fog", resources=(ResourceSpec("cpu", 0.25),)),
                ),
                links=(LinkSpec("edge-fog", "edge", "fog"),),
            )
            report = await exercise_cluster_runtime(
                inventory,
                system,
                source_node_id="edge",
                target_node_id="fog",
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
                cpu_request=0.1,
                timeout_s=5.0,
            )
            assert report.ready is True
            assert report.errors == ()
            assert report.preflight.ready is True
            assert tuple(step.name for step in report.steps) == (
                "place_source",
                "restart_source",
                "migrate_target",
                "stop_target",
            )
            assert all(step.ok for step in report.steps)
            restarted = report.steps[1].details
            assert restarted["attempt"] == 1
            assert restarted["restart_mode"] == "process"
            assert restarted["active_executions"] == 1
            migrated = report.steps[2].details
            assert migrated["attempt"] == 2
            assert migrated["source_active_executions"] == 0
            assert migrated["target_active_executions"] == 1
            assert report.steps[3].details["active_executions"] == 0
            assert (await inventory.client(inventory.nodes[0]).telemetry())[\
                "active_executions"
            ] == 0
            assert (await inventory.client(inventory.nodes[1]).telemetry())[\
                "active_executions"
            ] == 0
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())


def test_cluster_runtime_exercise_refuses_failed_preflight():
    async def run() -> None:
        server = AgentServer("actual", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (
                    ClusterNode("declared", "127.0.0.1", server.port),
                    ClusterNode("other", "127.0.0.1", server.port),
                )
            )
            system = SystemSpec(
                nodes=(NodeSpec("declared"), NodeSpec("other")),
            )
            report = await exercise_cluster_runtime(
                inventory,
                system,
                source_node_id="declared",
                target_node_id="other",
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
            )
            assert report.ready is False
            assert report.steps == ()
            assert report.errors
        finally:
            await server.close()

    asyncio.run(run())
