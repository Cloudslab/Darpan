from __future__ import annotations

import asyncio

from darpan.runtime.real.cluster.deployment import SSHResult
from darpan.runtime.real.cluster.inventory import ClusterNode
from darpan.runtime.real.cluster.readiness import inspect_host_readiness


class _Runner:
    def __init__(self, *, missing: str | None = None) -> None:
        self.missing = missing
        self.commands: list[str] = []

    async def run(self, node, command, *, stdin=None):
        del node, stdin
        self.commands.append(command)
        if self.missing and self.missing in command:
            return SSHResult(1, "", "missing")
        return SSHResult(0, "ok", "")

    async def upload(self, node, local, remote):
        raise AssertionError("readiness never uploads")


def test_host_readiness_checks_physical_prerequisites():
    async def run() -> None:
        node = ClusterNode(
            "edge",
            "10.0.0.1",
            physical_control=True,
            control_interfaces=("eth1",),
            control_cpu_max="/sys/fs/cgroup/demo/cpu.max",
        )
        runner = _Runner()
        report = await inspect_host_readiness(node, runner=runner)
        names = {check.name for check in report.checks}
        assert {"python", "systemd", "disk", "tc", "ip"}.issubset(names)
        assert "interface:eth1" in names
        assert "control_cpu_max" in names
        assert report.ssh_ready is True
        cpu_check = next(command for command in runner.commands if "cpu.max" in command)
        assert "systemctl show darpan-agent-edge.service" in cpu_check
        assert 'sudo -n -u "$service_user" test -w' in cpu_check

    asyncio.run(run())
