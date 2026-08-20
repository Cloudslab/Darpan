from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from darpan.runtime.real.cluster.deployment import (
    BootstrapOptions,
    OpenSSHRunner,
    SSHResult,
    apply_bootstrap_plan,
    build_bootstrap_plan,
)
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode


class _Runner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str]] = []
        self.stdin_payloads: list[tuple[str, bytes | None]] = []
        self.uploads: list[tuple[str, str, str]] = []

    async def run(self, node, command, *, stdin=None):
        self.commands.append((node.id, command))
        self.stdin_payloads.append((node.id, stdin))
        return SSHResult(0, "ok", "")

    async def upload(self, node, local: Path, remote: str):
        self.uploads.append((node.id, local.name, remote))
        return SSHResult(0, "", "")


def test_bootstrap_plan_is_redacted_and_renders_persistent_systemd(tmp_path, monkeypatch):
    wheel = tmp_path / "darpan.whl"
    wheel.write_bytes(b"wheel")
    monkeypatch.setenv("EDGE_TOKEN", "super-secret")
    inventory = ClusterInventory(
        (
            ClusterNode(
                "edge",
                "10.0.0.10",
                ssh_user="ubuntu",
                ssh_port=2222,
                ssh_identity_file="C:/keys/edge-key",
                token_env="EDGE_TOKEN",
                network_probe=True,
                artifact_forward=True,
                physical_control=True,
                control_interfaces=("eth1",),
                control_cpu_max=(
                    "/sys/fs/cgroup/system.slice/darpan-agent-edge.service/cpu.max"
                ),
            ),
        )
    )
    options = BootstrapOptions(wheel=wheel)
    plan = build_bootstrap_plan(inventory, options=options)
    node = plan.nodes[0]

    assert plan.schema == "darpan.cluster-bootstrap/v1"
    assert node.destination == "ubuntu@10.0.0.10"
    assert node.ssh_port == 2222
    assert node.ssh_identity_file == "C:/keys/edge-key"
    assert "super-secret" not in str(plan.to_dict())
    assert "DARPAN_AGENT_TOKEN=<resolved-at-apply-time>" in node.environment_template
    assert "--workspace-root /var/lib/darpan/workspaces/edge" in node.systemd_unit
    assert "--enable-network-probe" in node.systemd_unit
    assert "--enable-artifact-forward" in node.systemd_unit
    assert "--enable-physical-control" in node.systemd_unit
    assert "--control-cpu-max" in node.systemd_unit
    assert (
        "ExecStartPre=+/usr/bin/chown darpan:darpan "
        "/sys/fs/cgroup/system.slice/darpan-agent-edge.service/cpu.max"
        in node.systemd_unit
    )
    assert "AmbientCapabilities=CAP_NET_ADMIN" in node.systemd_unit
    assert "Delegate=cpu" in node.systemd_unit


def test_bootstrap_does_not_delegate_cpu_without_allowlisted_cpu_max():
    inventory = ClusterInventory(
        (
            ClusterNode(
                "edge",
                "127.0.0.1",
                physical_control=True,
                control_interfaces=("eth1",),
            ),
        )
    )

    rendered = build_bootstrap_plan(inventory).nodes[0].systemd_unit

    assert "Delegate=cpu" not in rendered


def test_bootstrap_apply_resolves_secret_only_at_execution_time(tmp_path, monkeypatch):
    async def run() -> None:
        wheel = tmp_path / "darpan.whl"
        wheel.write_bytes(b"wheel")
        monkeypatch.setenv("EDGE_TOKEN", "runtime-token")
        inventory = ClusterInventory(
            (
                ClusterNode(
                    "edge",
                    "127.0.0.1",
                    token_env="EDGE_TOKEN",
                ),
            )
        )
        options = BootstrapOptions(wheel=wheel)
        plan = build_bootstrap_plan(inventory, options=options)
        runner = _Runner()
        result = await apply_bootstrap_plan(
            inventory,
            plan,
            options=options,
            runner=runner,
        )
        assert result["nodes"][0]["installed"] is True
        assert runner.uploads == [("edge", "darpan.whl", "/tmp/darpan.whl")]
        assert any("systemctl enable" in command for _, command in runner.commands)
        assert any("systemctl restart" in command for _, command in runner.commands)
        # The secret is permitted only in the transient remote env-file install
        # command. It is never stored in the bootstrap plan itself.
        assert "runtime-token" not in str(plan.to_dict())
        assert all("runtime-token" not in command for _, command in runner.commands)
        assert any(
            payload is not None and b"runtime-token" in payload
            for _, payload in runner.stdin_payloads
        )

    asyncio.run(run())


def test_openssh_connection_is_noninteractive_and_supports_identity_file():
    node = ClusterNode(
        "edge",
        "127.0.0.1",
        ssh_identity_file="C:/keys/edge-key",
    )
    args = OpenSSHRunner._connection_args(node)
    assert args[:2] == ("-i", "C:/keys/edge-key")
    assert "BatchMode=yes" in args
    assert "StrictHostKeyChecking=no" in args
    assert f"UserKnownHostsFile={os.devnull}" in args


def test_bootstrap_tls_uses_standard_remote_paths_without_embedding_private_key(tmp_path):
    async def run() -> None:
        wheel = tmp_path / "darpan.whl"
        wheel.write_bytes(b"wheel")
        tls = tmp_path / "tls"
        tls.mkdir()
        (tls / "edge.crt").write_text("CERTIFICATE-DATA", encoding="utf-8")
        (tls / "edge.key").write_text("PRIVATE-KEY-DATA", encoding="utf-8")
        inventory = ClusterInventory(
            (
                ClusterNode(
                    "edge",
                    "127.0.0.1",
                    tls_ca="controller-ca.pem",
                    tls_cert="placeholder.crt",
                    tls_key="placeholder.key",
                ),
            )
        )
        options = BootstrapOptions(wheel=wheel, tls_source_dir=tls)
        plan = build_bootstrap_plan(inventory, options=options)
        rendered = plan.nodes[0].systemd_unit
        assert "--tls-cert /etc/darpan/tls/edge.crt" in rendered
        assert "--tls-key /etc/darpan/tls/edge.key" in rendered
        assert "PRIVATE-KEY-DATA" not in str(plan.to_dict())

        runner = _Runner()
        await apply_bootstrap_plan(inventory, plan, options=options, runner=runner)
        assert ("edge", "edge.crt", "/tmp/darpan-agent.crt") in runner.uploads
        assert ("edge", "edge.key", "/tmp/darpan-agent.key") in runner.uploads
        assert all("PRIVATE-KEY-DATA" not in command for _, command in runner.commands)

    asyncio.run(run())


def test_bootstrap_rejects_mismatched_tls_client_and_server_configuration(tmp_path):
    inventory = ClusterInventory(
        (ClusterNode("edge", "127.0.0.1", tls_ca="controller-ca.pem"),)
    )
    with pytest.raises(ValueError, match="server certificate"):
        build_bootstrap_plan(inventory)

    tls = tmp_path / "tls"
    tls.mkdir()
    (tls / "edge.crt").write_text("cert", encoding="utf-8")
    (tls / "edge.key").write_text("key", encoding="utf-8")
    plaintext_inventory = ClusterInventory((ClusterNode("edge", "127.0.0.1"),))
    with pytest.raises(ValueError, match="tls_ca"):
        build_bootstrap_plan(
            plaintext_inventory,
            options=BootstrapOptions(tls_source_dir=tls),
        )
