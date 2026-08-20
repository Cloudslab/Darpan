"""SSH and Agent readiness checks for first physical deployment."""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

from .deployment import OpenSSHRunner, SSHRunner
from .inventory import ClusterInventory, ClusterNode
from .validation import ClusterValidationReport, validate_cluster


@dataclass(frozen=True, slots=True)
class HostCheck:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HostReadiness:
    node_id: str
    ssh_ready: bool
    checks: tuple[HostCheck, ...]


@dataclass(frozen=True, slots=True)
class ClusterReadinessReport:
    ready: bool
    hosts: tuple[HostReadiness, ...]
    agent_validation: ClusterValidationReport | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _run_check(
    runner: SSHRunner,
    node: ClusterNode,
    name: str,
    command: str,
    *,
    required: bool = True,
) -> HostCheck:
    result = await runner.run(node, command)
    ok = result.returncode == 0
    details = {"stdout": result.stdout.strip()} if result.stdout.strip() else {}
    error = None
    if not ok and required:
        error = result.stderr.strip() or result.stdout.strip() or f"{name} unavailable"
    return HostCheck(name=name, ok=(ok or not required), details=details, error=error)


async def inspect_host_readiness(
    node: ClusterNode,
    *,
    runner: SSHRunner | None = None,
    python_command: str = "python3",
) -> HostReadiness:
    runner = OpenSSHRunner() if runner is None else runner
    checks: list[HostCheck] = []
    checks.append(
        await _run_check(
            runner,
            node,
            "python",
            f"command -v {shlex.quote(python_command)} && "
            f"{shlex.quote(python_command)} -c 'import sys; print(sys.version_info[:2])'",
        )
    )
    checks.append(
        await _run_check(runner, node, "systemd", "command -v systemctl")
    )
    checks.append(
        await _run_check(runner, node, "disk", "df -Pk / | tail -1")
    )
    checks.append(
        await _run_check(
            runner,
            node,
            "docker",
            "command -v docker && docker version --format '{{.Server.Version}}'",
            required=False,
        )
    )
    if node.physical_control:
        checks.append(await _run_check(runner, node, "tc", "command -v tc"))
        checks.append(await _run_check(runner, node, "ip", "command -v ip"))
        for interface in node.control_interfaces:
            checks.append(
                await _run_check(
                    runner,
                    node,
                    f"interface:{interface}",
                    f"ip link show dev {shlex.quote(interface)} >/dev/null",
                )
            )
        if node.control_cpu_max is not None:
            path = shlex.quote(node.control_cpu_max)
            service = shlex.quote(f"darpan-agent-{node.id}.service")
            checks.append(
                await _run_check(
                    runner,
                    node,
                    "control_cpu_max",
                    f"test -e {path} && "
                    f"service_user=$(systemctl show {service} --property=User --value) && "
                    'test -n "$service_user" && '
                    f"sudo -n -u \"$service_user\" test -w {path}",
                )
            )
    return HostReadiness(
        node_id=node.id,
        ssh_ready=all(check.ok for check in checks if check.error is not None or check.ok),
        checks=tuple(checks),
    )


async def check_deployment_readiness(
    inventory: ClusterInventory,
    *,
    runner: SSHRunner | None = None,
    require_agent: bool = False,
    system=None,
    max_clock_offset_s: float = 1.0,
) -> ClusterReadinessReport:
    runner = OpenSSHRunner() if runner is None else runner
    hosts = tuple(
        await asyncio.gather(
            *(inspect_host_readiness(node, runner=runner) for node in inventory.nodes)
        )
    )
    errors = [
        f"{host.node_id}:{check.name}: {check.error}"
        for host in hosts
        for check in host.checks
        if not check.ok and check.error
    ]
    validation = None
    if require_agent and not errors:
        validation = await validate_cluster(
            inventory,
            system=system,
            max_clock_offset_s=max_clock_offset_s,
        )
        errors.extend(validation.errors)
    ready = not errors and (validation is None or validation.ready)
    return ClusterReadinessReport(
        ready=ready,
        hosts=hosts,
        agent_validation=validation,
        errors=tuple(errors),
    )


def readiness_json(report: ClusterReadinessReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
