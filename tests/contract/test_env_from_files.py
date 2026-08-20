from __future__ import annotations

from darpan.adapters.rl import DarpanEnv


def test_darpan_env_accepts_configuration_files():
    env = DarpanEnv(
        system="configs/systems/edge_fog_cloud.yaml",
        application="configs/workloads/example.yaml",
        runtime="twin",
    )
    _, info = env.reset()
    assert info["decision_instance_id"] is not None
    env.close()
