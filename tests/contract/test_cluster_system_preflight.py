from __future__ import annotations

import pytest

from darpan.core.topology import NodeSpec, SystemSpec
from darpan.runtime.real.cluster.inventory import ClusterInventory, ClusterNode
from darpan.runtime.real.cluster.session import (
    session_from_inventory,
    validate_inventory_system_mapping,
)


def test_cluster_system_mapping_rejects_missing_physical_agent_before_session():
    inventory = ClusterInventory((ClusterNode("edge", "127.0.0.1", 9999),))
    system = SystemSpec(nodes=(NodeSpec("edge"), NodeSpec("cloud")))

    with pytest.raises(ValueError, match="cloud"):
        validate_inventory_system_mapping(inventory, system)
    with pytest.raises(ValueError, match="cloud"):
        session_from_inventory(inventory, system=system)
