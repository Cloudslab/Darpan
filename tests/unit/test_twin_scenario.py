from __future__ import annotations

import asyncio

from darpan import Darpan, DigitalTwin
from darpan.twin.models.network import NetworkDelayModel


def test_scenario_does_not_mutate_snapshot(small_system):
    async def build():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        twin = DigitalTwin()
        snapshot = twin.snapshot(session.state)
        scenario = twin.scenario(session.state).remove_node("edge-1")
        changed = scenario.materialize()
        assert changed.nodes["edge-1"].status == "offline"
        assert snapshot.state.nodes["edge-1"].status == "online"
        await session.close()

    asyncio.run(build())


def test_network_model_supports_multihop(small_system):
    async def build_state():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        prediction = NetworkDelayModel().predict(
            {"source": "edge-1", "target": "cloud-1", "size_bytes": 1000},
            session.state,
        )
        assert prediction.estimate < float("inf")
        assert prediction.metadata["path"] == ["edge-1", "fog-1", "cloud-1"]
        await session.close()

    asyncio.run(build_state())
