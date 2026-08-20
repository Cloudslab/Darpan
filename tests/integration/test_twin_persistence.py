from __future__ import annotations

import asyncio

from darpan import Action, Darpan, DigitalTwin, Event, Measurement
from darpan.core.event import EventKind
from darpan.twin.scenario import Scenario
from darpan.twin.snapshot import TwinSnapshot


def test_snapshot_round_trip_preserves_state_and_models(tmp_path, small_system, small_app):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        await session.apply(Action.place(f"{instance}:a", "edge-1"))
        twin = DigitalTwin()
        snapshot = twin.snapshot(session.state)
        path = snapshot.save(tmp_path / "snapshot.json")
        loaded = TwinSnapshot.load(path)
        assert loaded.state == snapshot.state
        assert loaded.virtual_time == snapshot.virtual_time
        assert loaded.model_states == snapshot.model_states
        await session.close()

    asyncio.run(run())


def test_scenario_round_trip_is_self_contained(tmp_path, small_system):
    async def run():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        scenario = (
            DigitalTwin()
            .scenario(session.state, seed=9)
            .change_link("edge-fog", latency_ms=99)
            .set_measurement(
                Measurement(
                    "temperature",
                    80,
                    target="edge-1",
                    source="test",
                )
            )
        )
        scenario.horizon_s = 5
        scenario.inject_event(
            Event(
                EventKind.NODE_OFFLINE,
                2.0,
                "test",
                payload={"node_id": "fog-1"},
            )
        )
        path = scenario.save(tmp_path / "scenario.json")
        loaded = Scenario.load(path)
        materialized = loaded.materialize()
        assert materialized.links["edge-fog"].spec.latency_ms == 99
        assert materialized.nodes["edge-1"].measurement("temperature").value == 80
        assert loaded.injected_events[0].kind == EventKind.NODE_OFFLINE
        assert loaded.horizon_s == 5
        assert loaded.seed == 9
        await session.close()

    asyncio.run(run())
