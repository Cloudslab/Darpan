"""Physical Session construction from a validated cluster inventory."""

from __future__ import annotations

from darpan.api import Darpan
from darpan.core.topology import SystemSpec
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.executors.remote import RemoteExecutor
from darpan.runtime.session import Session

from .inventory import ClusterInventory
from .monitor import ClusterMonitor
from .probe import LinkProbeService


def validate_inventory_system_mapping(
    inventory: ClusterInventory, system: SystemSpec
) -> None:
    """Require every physical topology node to have an inventory Agent.

    Extra inventory nodes are allowed so one physical cluster can host several
    experiment topologies. Missing executors are not: silently running such a
    node on the controller would invalidate Physical Continuum evidence.
    """

    inventory_ids = {node.id for node in inventory.nodes}
    missing = sorted(node.id for node in system.nodes if node.id not in inventory_ids)
    if missing:
        raise ValueError(
            "physical system nodes are missing from cluster inventory: "
            + ", ".join(missing)
        )


def session_from_inventory(
    inventory: ClusterInventory,
    *,
    system: SystemSpec | None = None,
    monitor: bool = True,
    link_probes: bool = True,
    network_driver=None,
    physical_control: bool = False,
) -> Session:
    """Build a Real Session whose node IDs map directly to remote Agents.

    The helper centralizes cluster feature wiring so CLI experiments, RL
    environments, and validation drills cannot silently disagree about direct
    artifact forwarding, health monitoring, or physical link telemetry.
    """

    if system is not None:
        validate_inventory_system_mapping(inventory, system)
    clients = {node.id: inventory.client(node) for node in inventory.nodes}
    executors = {
        node.id: RemoteExecutor(
            clients[node.id],
            direct_artifact_forward=node.artifact_forward,
        )
        for node in inventory.nodes
    }
    if network_driver == "linux-agent":
        required = [node.id for node in inventory.nodes if not node.physical_control]
        if required:
            raise RuntimeError(
                "linux-agent ROUTE requires physical_control on inventory nodes: "
                + ", ".join(required)
            )
        from ..linux_network import AgentLinuxNetworkControlDriver

        network_driver = AgentLinuxNetworkControlDriver(clients)
    physical_driver = None
    if physical_control:
        required = [node.id for node in inventory.nodes if not node.physical_control]
        if required:
            raise RuntimeError(
                "physical_control requires inventory nodes to opt in: "
                + ", ".join(required)
            )
        from ..physical_control import AgentPhysicalControlDriver

        physical_driver = AgentPhysicalControlDriver(clients)
    session = Darpan.real(
        backend=RealBackend(
            executors=executors,
            require_explicit_executor=True,
            network_driver=network_driver,
            physical_control_driver=physical_driver,
        )
    )
    if monitor:
        session.add_service(ClusterMonitor(session, clients))
    if link_probes and any(node.network_probe for node in inventory.nodes):
        session.add_service(LinkProbeService(session, inventory, clients))
    return session
