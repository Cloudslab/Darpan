from __future__ import annotations

import asyncio

from darpan import Darpan, DigitalTwin, Event
from darpan.core.event import EventKind


def test_future_fault_event_can_be_simulated(small_system):
    async def run():
        base = Darpan.twin()
        await base.start()
        await base.register_system(small_system)
        twin = DigitalTwin()
        scenario = twin.scenario(base.state)
        scenario.horizon_s = 10
        scenario.inject_event(
            Event(
                kind=EventKind.NODE_REMOVED,
                event_time=5.0,
                source="test.fault",
                subject="edge-1",
                payload={"node_id": "edge-1"},
            )
        )
        result = await twin.simulate(scenario)
        assert result.state.time == 10.0
        assert result.state.nodes["edge-1"].status == "offline"
        await base.close()

    asyncio.run(run())
