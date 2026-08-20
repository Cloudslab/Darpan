from __future__ import annotations

from darpan.runtime.real.cluster.inventory import ClusterInventory


def test_cluster_inventory_direct_artifact_forward_is_explicit_opt_in(tmp_path):
    inventory_file = tmp_path / "cluster.yaml"
    inventory_file.write_text(
        """
nodes:
  - id: edge
    host: edge.local
    artifact_forward: true
  - id: cloud
    host: cloud.local
""".lstrip(),
        encoding="utf-8",
    )
    inventory = ClusterInventory.load(inventory_file)
    assert inventory.nodes[0].artifact_forward is True
    assert inventory.nodes[1].artifact_forward is False


def test_cluster_inventory_fails_fast_for_duplicate_nodes_and_partial_tls():
    import pytest

    from darpan.runtime.real.cluster.inventory import ClusterNode

    with pytest.raises(ValueError, match="node ids must be unique"):
        ClusterInventory(
            (
                ClusterNode("edge", "one.local"),
                ClusterNode("edge", "two.local"),
            )
        )
    with pytest.raises(ValueError, match="tls_cert/tls_key must be paired"):
        ClusterInventory(
            (
                ClusterNode("edge", "edge.local", tls_cert="cert.pem"),
            )
        )
