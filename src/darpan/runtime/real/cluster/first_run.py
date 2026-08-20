"""One-shot first-run orchestration for a newly deployed physical cluster."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from darpan.core.topology import SystemSpec

from .acceptance import ClusterAcceptanceReport, accept_cluster
from .deployment import SSHRunner
from .discovery import ClusterDiscovery, discover_cluster
from .inventory import ClusterInventory
from .readiness import ClusterReadinessReport, check_deployment_readiness


@dataclass(frozen=True, slots=True)
class PhysicalFirstRunReport:
    ready_for_study: bool
    ssh_check_skipped: bool
    host_readiness: ClusterReadinessReport | None
    discovery: ClusterDiscovery | None
    acceptance: ClusterAcceptanceReport | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_cluster_first_run(
    inventory: ClusterInventory,
    system: SystemSpec,
    *,
    source_node_id: str,
    target_node_id: str,
    command: tuple[str, ...],
    python_command: str = "python3",
    cpu_request: float | None = None,
    timeout_s: float = 10.0,
    max_clock_offset_s: float = 1.0,
    exercise_physical_control: bool = False,
    exercise_link_control: bool = False,
    skip_ssh_check: bool = False,
    ssh_runner: SSHRunner | None = None,
) -> PhysicalFirstRunReport:
    """Check, discover and actively accept one deployed cluster in one workflow."""

    host_readiness = None
    if not skip_ssh_check:
        host_readiness = await check_deployment_readiness(
            inventory,
            runner=ssh_runner,
            require_agent=True,
            system=system,
            max_clock_offset_s=max_clock_offset_s,
        )
        if not host_readiness.ready:
            return PhysicalFirstRunReport(
                ready_for_study=False,
                ssh_check_skipped=False,
                host_readiness=host_readiness,
                discovery=None,
                acceptance=None,
                errors=tuple(host_readiness.errors),
            )

    errors: list[str] = []
    discovery = await discover_cluster(inventory)
    acceptance = await accept_cluster(
        inventory,
        system,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        command=command,
        python_command=python_command,
        cpu_request=cpu_request,
        timeout_s=timeout_s,
        exercise_physical_control=exercise_physical_control,
        exercise_link_control=exercise_link_control,
    )
    errors.extend(acceptance.errors)
    accepted_environment = acceptance.preflight.environment_fingerprint
    if discovery.environment_fingerprint != accepted_environment:
        errors.append(
            "cluster environment changed between discovery and active acceptance"
        )
    ready = acceptance.ready_for_study and not errors
    return PhysicalFirstRunReport(
        ready_for_study=ready,
        ssh_check_skipped=skip_ssh_check,
        host_readiness=host_readiness,
        discovery=discovery,
        acceptance=acceptance,
        errors=tuple(errors),
    )
