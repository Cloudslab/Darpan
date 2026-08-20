"""Auditable provisioning helpers for physical Darpan Agents.

The deployment layer is intentionally separate from the runtime protocol.  It
uses OpenSSH to provision an Agent host and writes a systemd service whose
arguments are derived from the frozen cluster inventory.  Plans never contain
secret token values; ``apply`` resolves tokens only at execution time.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from darpan._version import __version__

from .inventory import ClusterInventory, ClusterNode


@dataclass(frozen=True, slots=True)
class SSHResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class SSHRunner(Protocol):
    async def run(
        self,
        node: ClusterNode,
        command: str,
        *,
        stdin: bytes | None = None,
    ) -> SSHResult: ...

    async def upload(self, node: ClusterNode, local: Path, remote: str) -> SSHResult: ...


class OpenSSHRunner:
    """Small OpenSSH boundary kept injectable for deterministic tests."""

    @staticmethod
    def _destination(node: ClusterNode) -> str:
        return f"{node.ssh_user}@{node.host}" if node.ssh_user else node.host

    @staticmethod
    def _connection_args(node: ClusterNode) -> tuple[str, ...]:
        args = (
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={os.devnull}",
        )
        if node.ssh_identity_file is None:
            return args
        return ("-i", node.ssh_identity_file, *args)

    async def run(
        self,
        node: ClusterNode,
        command: str,
        *,
        stdin: bytes | None = None,
    ) -> SSHResult:
        process = await asyncio.create_subprocess_exec(
            "ssh",
            *self._connection_args(node),
            "-p",
            str(node.ssh_port),
            self._destination(node),
            command,
            stdin=(asyncio.subprocess.PIPE if stdin is not None else None),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(stdin)
        return SSHResult(
            int(process.returncode or 0),
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def upload(self, node: ClusterNode, local: Path, remote: str) -> SSHResult:
        process = await asyncio.create_subprocess_exec(
            "scp",
            *self._connection_args(node),
            "-P",
            str(node.ssh_port),
            str(local),
            f"{self._destination(node)}:{remote}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return SSHResult(
            int(process.returncode or 0),
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


@dataclass(frozen=True, slots=True)
class BootstrapOptions:
    service_user: str = "darpan"
    install_root: str = "/opt/darpan"
    config_root: str = "/etc/darpan"
    data_root: str = "/var/lib/darpan"
    package_spec: str = field(
        default_factory=lambda: f"darpan-continuum=={__version__}"
    )
    wheel: Path | None = None
    python_command: str = "python3"
    grant_docker_group: bool = False
    tls_source_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.service_user or any(ch.isspace() for ch in self.service_user):
            raise ValueError("bootstrap service_user must be a simple non-empty name")
        for value, name in (
            (self.install_root, "install_root"),
            (self.config_root, "config_root"),
            (self.data_root, "data_root"),
        ):
            if not value.startswith("/"):
                raise ValueError(f"bootstrap {name} must be an absolute path")
        if self.wheel is not None and not self.wheel.is_file():
            raise FileNotFoundError(self.wheel)
        if self.tls_source_dir is not None and not self.tls_source_dir.is_dir():
            raise FileNotFoundError(self.tls_source_dir)


@dataclass(frozen=True, slots=True)
class BootstrapNodePlan:
    node_id: str
    destination: str
    ssh_port: int
    ssh_identity_file: str | None
    service_name: str
    workspace_root: str
    install_source: str
    token_source_env: str | None
    systemd_unit: str
    environment_template: str
    commands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    schema: str
    darpan_version: str
    nodes: tuple[BootstrapNodePlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _agent_args(
    node: ClusterNode,
    *,
    workspace_root: str,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> list[str]:
    args = [
        "agent",
        "--node-id",
        node.id,
        "--host",
        "0.0.0.0",
        "--port",
        str(node.port),
        "--workspace-root",
        workspace_root,
    ]
    if node.token_env:
        args += ["--token-env", "DARPAN_AGENT_TOKEN"]
    if node.network_probe:
        args.append("--enable-network-probe")
    if node.artifact_forward:
        args.append("--enable-artifact-forward")
    if node.physical_control:
        args.append("--enable-physical-control")
        for interface in node.control_interfaces:
            args += ["--control-interface", interface]
        if node.control_cpu_max is not None:
            args += ["--control-cpu-max", node.control_cpu_max]
        if node.allow_replace_existing_qdisc:
            args.append("--allow-replace-existing-qdisc")
    cert = tls_cert if tls_cert is not None else node.tls_cert
    key = tls_key if tls_key is not None else node.tls_key
    if cert is not None:
        if key is None:
            raise ValueError(f"node {node.id} must define a TLS key with its certificate")
        args += ["--tls-cert", cert, "--tls-key", key]
    return args


def _systemd_unit(
    node: ClusterNode,
    options: BootstrapOptions,
    *,
    service_name: str,
    workspace_root: str,
) -> str:
    executable = f"{options.install_root}/venv/bin/darpan"
    tls_cert = None
    tls_key = None
    if options.tls_source_dir is not None:
        tls_cert = f"{options.config_root}/tls/{node.id}.crt"
        tls_key = f"{options.config_root}/tls/{node.id}.key"
    command = " ".join(
        shlex.quote(item)
        for item in [
            executable,
            *_agent_args(
                node,
                workspace_root=workspace_root,
                tls_cert=tls_cert,
                tls_key=tls_key,
            ),
        ]
    )
    lines = [
        "[Unit]",
        f"Description=Darpan Continuum Agent ({node.id})",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"User={options.service_user}",
        f"Group={options.service_user}",
        f"EnvironmentFile=-{options.config_root}/{node.id}.env",
    ]
    if node.physical_control and node.control_cpu_max is not None:
        cpu_max_path = shlex.quote(node.control_cpu_max)
        owner = shlex.quote(f"{options.service_user}:{options.service_user}")
        lines.append(f"ExecStartPre=+/usr/bin/chown {owner} {cpu_max_path}")
    lines += [
        f"ExecStart={command}",
        "Restart=on-failure",
        "RestartSec=2",
        "TimeoutStopSec=20",
    ]
    if node.physical_control:
        lines += [
            "AmbientCapabilities=CAP_NET_ADMIN",
            "CapabilityBoundingSet=CAP_NET_ADMIN",
        ]
        if node.control_cpu_max is not None:
            # Delegate only the CPU controller. The one-time privileged
            # ExecStartPre changes ownership of this unit's own allow-listed
            # cpu.max file; the long-running Agent remains unprivileged and
            # cannot affect sibling units or other controllers.
            lines.append("Delegate=cpu")
    lines += [
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        f"ReadWritePaths={options.data_root}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def build_bootstrap_plan(
    inventory: ClusterInventory,
    *,
    options: BootstrapOptions | None = None,
) -> BootstrapPlan:
    options = BootstrapOptions() if options is None else options
    install_source = (
        str(options.wheel.resolve()) if options.wheel is not None else options.package_spec
    )
    nodes: list[BootstrapNodePlan] = []
    for node in inventory.nodes:
        if node.tls_ca is not None and options.tls_source_dir is None and node.tls_cert is None:
            raise ValueError(
                f"node {node.id!r} configures TLS client validation but bootstrap has no "
                "Agent server certificate/key source"
            )
        if options.tls_source_dir is not None and node.tls_ca is None:
            raise ValueError(
                f"node {node.id!r} bootstrap uploads a TLS server certificate but inventory "
                "does not configure tls_ca for controller validation"
            )
        destination = f"{node.ssh_user}@{node.host}" if node.ssh_user else node.host
        service_name = f"darpan-agent-{node.id}.service"
        workspace_root = f"{options.data_root}/workspaces/{node.id}"
        unit = _systemd_unit(
            node,
            options,
            service_name=service_name,
            workspace_root=workspace_root,
        )
        environment = (
            "DARPAN_AGENT_TOKEN=<resolved-at-apply-time>\n" if node.token_env else ""
        )
        package_source = (
            f"/tmp/{options.wheel.name}"
            if options.wheel is not None
            else options.package_spec
        )
        commands = [
            f"sudo install -d -m 0755 {shlex.quote(options.install_root)}",
            f"sudo install -d -m 0750 {shlex.quote(options.config_root)}",
            f"sudo install -d -m 0750 {shlex.quote(options.config_root + '/tls')}",
            f"sudo install -d -m 0750 {shlex.quote(options.data_root)}",
            (
                f"id -u {shlex.quote(options.service_user)} >/dev/null 2>&1 || "
                f"sudo useradd --system --home {shlex.quote(options.data_root)} "
                f"--shell /usr/sbin/nologin {shlex.quote(options.service_user)}"
            ),
            (
                f"sudo {shlex.quote(options.python_command)} -m venv "
                f"{shlex.quote(options.install_root + '/venv')}"
            ),
            (
                f"sudo {shlex.quote(options.install_root + '/venv/bin/python')} -m pip "
                f"install --upgrade {shlex.quote(package_source)}"
            ),
            (
                f"sudo chown -R {shlex.quote(options.service_user)}:"
                f"{shlex.quote(options.service_user)} {shlex.quote(options.data_root)}"
            ),
        ]
        if options.grant_docker_group:
            commands.append(
                f"getent group docker >/dev/null 2>&1 && sudo usermod -aG docker "
                f"{shlex.quote(options.service_user)} || true"
            )
        commands += [
            "sudo systemctl daemon-reload",
            (
                f"sudo systemctl enable {shlex.quote(service_name)}"
                f" && sudo systemctl restart {shlex.quote(service_name)}"
            ),
        ]
        nodes.append(
            BootstrapNodePlan(
                node_id=node.id,
                destination=destination,
                ssh_port=node.ssh_port,
                ssh_identity_file=node.ssh_identity_file,
                service_name=service_name,
                workspace_root=workspace_root,
                install_source=install_source,
                token_source_env=node.token_env,
                systemd_unit=unit,
                environment_template=environment,
                commands=tuple(commands),
            )
        )
    return BootstrapPlan(
        schema="darpan.cluster-bootstrap/v1",
        darpan_version=__version__,
        nodes=tuple(nodes),
    )


async def _require_ok(result: SSHResult, *, node_id: str, operation: str) -> None:
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "remote command failed"
        raise RuntimeError(f"{node_id} {operation} failed: {message}")


async def _install_text(
    runner: SSHRunner,
    node: ClusterNode,
    *,
    path: str,
    content: str,
    mode: str,
) -> None:
    command = f"sudo install -m {mode} /dev/stdin {shlex.quote(path)}"
    await _require_ok(
        await runner.run(node, command, stdin=content.encode("utf-8")),
        node_id=node.id,
        operation=f"install {path}",
    )


async def apply_bootstrap_plan(
    inventory: ClusterInventory,
    plan: BootstrapPlan,
    *,
    options: BootstrapOptions | None = None,
    runner: SSHRunner | None = None,
) -> dict[str, Any]:
    options = BootstrapOptions() if options is None else options
    runner = OpenSSHRunner() if runner is None else runner
    by_id = {node.id: node for node in inventory.nodes}
    results: list[dict[str, Any]] = []
    for node_plan in plan.nodes:
        node = by_id[node_plan.node_id]
        if options.wheel is not None:
            remote_wheel = f"/tmp/{options.wheel.name}"
            uploaded = await runner.upload(
                node, options.wheel.resolve(), remote_wheel
            )
            await _require_ok(uploaded, node_id=node.id, operation="upload wheel")
        bootstrap_command = "set -eu\n" + "\n".join(node_plan.commands[:-2])
        await _require_ok(
            await runner.run(node, bootstrap_command),
            node_id=node.id,
            operation="bootstrap",
        )
        if options.tls_source_dir is not None:
            cert_source = options.tls_source_dir / f"{node.id}.crt"
            key_source = options.tls_source_dir / f"{node.id}.key"
            for source in (cert_source, key_source):
                if not source.is_file():
                    raise FileNotFoundError(source)
            for source, remote_tmp, remote_target, mode in (
                (
                    cert_source,
                    "/tmp/darpan-agent.crt",
                    f"{options.config_root}/tls/{node.id}.crt",
                    "0644",
                ),
                (
                    key_source,
                    "/tmp/darpan-agent.key",
                    f"{options.config_root}/tls/{node.id}.key",
                    "0600",
                ),
            ):
                uploaded = await runner.upload(node, source, remote_tmp)
                await _require_ok(uploaded, node_id=node.id, operation=f"upload {source.name}")
                await _require_ok(
                    await runner.run(
                        node,
                        "sudo install -m "
                        f"{mode} {shlex.quote(remote_tmp)} "
                        f"{shlex.quote(remote_target)} && "
                        f"rm -f {shlex.quote(remote_tmp)}",
                    ),
                    node_id=node.id,
                    operation=f"install {source.name}",
                )
        token_text = ""
        if node.token_env is not None:
            token = node.token()
            token_text = f"DARPAN_AGENT_TOKEN={token}\n"
        await _install_text(
            runner,
            node,
            path=f"{options.config_root}/{node.id}.env",
            content=token_text,
            mode="0600",
        )
        await _install_text(
            runner,
            node,
            path=f"/etc/systemd/system/{node_plan.service_name}",
            content=node_plan.systemd_unit,
            mode="0644",
        )
        activation_command = "set -eu\n" + "\n".join(node_plan.commands[-2:])
        await _require_ok(
            await runner.run(node, activation_command),
            node_id=node.id,
            operation="systemd activation",
        )
        results.append(
            {
                "node_id": node.id,
                "service_name": node_plan.service_name,
                "workspace_root": node_plan.workspace_root,
                "installed": True,
            }
        )
    return {"schema": "darpan.cluster-bootstrap-result/v1", "nodes": results}
