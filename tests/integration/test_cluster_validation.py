from __future__ import annotations

import asyncio

from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode
from darpan.runtime.real.cluster.validation import validate_cluster


def test_cluster_validation_exercises_agent_to_agent_probe_and_direct_artifact():
    async def run():
        source = AgentServer(
            "edge",
            host="127.0.0.1",
            port=0,
            token="edge-secret",
            allow_network_probe=True,
            allow_artifact_forward=True,
        )
        target = AgentServer(
            "cloud",
            host="127.0.0.1",
            port=0,
            token="cloud-secret",
        )
        await source.start()
        await target.start()
        try:
            import os

            os.environ["DARPAN_EDGE_TOKEN"] = "edge-secret"
            os.environ["DARPAN_CLOUD_TOKEN"] = "cloud-secret"
            inventory = ClusterInventory(
                (
                    ClusterNode(
                        "edge",
                        "127.0.0.1",
                        source.port,
                        token_env="DARPAN_EDGE_TOKEN",
                        network_probe=True,
                        artifact_forward=True,
                    ),
                    ClusterNode(
                        "cloud",
                        "127.0.0.1",
                        target.port,
                        token_env="DARPAN_CLOUD_TOKEN",
                    ),
                )
            )
            system = SystemSpec(
                nodes=(NodeSpec("edge"), NodeSpec("cloud")),
                links=(LinkSpec("uplink", "edge", "cloud"),),
            )
            report = await validate_cluster(
                inventory,
                system=system,
                exercise_data_plane=True,
                payload_bytes=2048,
            )
            assert report.ready is True
            assert report.errors == ()
            assert all(node.reachable for node in report.nodes)
            assert report.environment_fingerprint is not None
            assert len(report.environment_fingerprint) == 64
            ping = next(
                check
                for check in report.nodes[0].checks
                if check.name == "ping"
            )
            assert ping.details["environment"]["python_version"]
            assert ping.details["environment"]["machine"]
            assert len(report.links) == 1
            checks = {check.name: check for check in report.links[0].checks}
            assert checks["network_probe"].ok is True
            assert checks["network_probe"].details["bandwidth_mbps"] > 0
            assert checks["direct_artifact"].ok is True
            assert checks["direct_artifact"].details["size_bytes"] == 2048
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())


def test_cluster_validation_reports_inventory_agent_identity_mismatch():
    async def run():
        server = AgentServer("actual", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (ClusterNode("declared", "127.0.0.1", server.port),)
            )
            report = await validate_cluster(inventory)
            assert report.ready is False
            assert report.nodes[0].reachable is False
            assert "does not match agent id" in report.errors[0]
        finally:
            await server.close()

    asyncio.run(run())


def test_cluster_validation_detects_inventory_feature_mismatch():
    async def run():
        server = AgentServer("edge", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (
                    ClusterNode(
                        "edge",
                        "127.0.0.1",
                        server.port,
                        network_probe=True,
                    ),
                )
            )
            report = await validate_cluster(inventory)
            assert report.ready is False
            assert "network_probe" in report.errors[0]
        finally:
            await server.close()

    asyncio.run(run())


def test_cluster_validation_rejects_declared_cpu_above_physical_capacity():
    async def run():
        server = AgentServer("edge", host="127.0.0.1", port=0)
        await server.start()
        try:
            inventory = ClusterInventory(
                (ClusterNode("edge", "127.0.0.1", server.port),)
            )
            system = SystemSpec(
                nodes=(
                    NodeSpec(
                        "edge",
                        resources=(ResourceSpec("cpu", 1_000_000),),
                    ),
                )
            )
            report = await validate_cluster(inventory, system=system)
            assert report.ready is False
            checks = {check.name: check for check in report.nodes[0].checks}
            capacity = checks["declared_cpu_capacity"]
            assert capacity.ok is False
            assert capacity.details["declared"] == 1_000_000
            assert capacity.details["effective"] == capacity.details["observed"]
            assert "exceeds observed physical capacity" in (capacity.error or "")
            assert any("declared_cpu_capacity" in error for error in report.errors)
        finally:
            await server.close()

    asyncio.run(run())
