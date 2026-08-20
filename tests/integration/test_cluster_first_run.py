from __future__ import annotations

import asyncio

from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.deployment import SSHResult
from darpan.runtime.real.cluster.first_run import run_cluster_first_run
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


class _ReadySSH:
    async def run(self, node, command: str) -> SSHResult:
        return SSHResult(returncode=0, stdout="ok\n", stderr="")

    async def upload(self, node, source, destination: str) -> SSHResult:
        return SSHResult(returncode=0, stdout="", stderr="")


def test_cluster_first_run_combines_host_discovery_and_active_acceptance():
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
            report = await run_cluster_first_run(
                inventory,
                system,
                source_node_id="edge",
                target_node_id="fog",
                command=("python3", "-c", "import time; time.sleep(3600)"),
                cpu_request=0.1,
                timeout_s=5,
                ssh_runner=_ReadySSH(),
            )
            assert report.ready_for_study is True
            assert report.ssh_check_skipped is False
            assert report.host_readiness is not None and report.host_readiness.ready
            assert report.discovery is not None
            assert report.acceptance is not None and report.acceptance.ready_for_study
            assert (
                report.discovery.environment_fingerprint
                == report.acceptance.preflight.environment_fingerprint
            )
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())


def test_cluster_first_run_requires_explicit_flag_to_skip_ssh_checks():
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
                nodes=(NodeSpec("edge"), NodeSpec("fog")),
                links=(LinkSpec("edge-fog", "edge", "fog"),),
            )
            report = await run_cluster_first_run(
                inventory,
                system,
                source_node_id="edge",
                target_node_id="fog",
                command=("python3", "-c", "import time; time.sleep(3600)"),
                timeout_s=5,
                skip_ssh_check=True,
            )
            assert report.ready_for_study is True
            assert report.ssh_check_skipped is True
            assert report.host_readiness is None
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())
