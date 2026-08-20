from __future__ import annotations

from darpan.core.event import Event, EventKind
from darpan.twin.models.network import NetworkDelayModel


def test_real_transfer_calibrates_endpoint_network_delay(small_system):
    from darpan import Darpan

    async def materialize_state():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        state = session.state
        await session.close()
        return state

    import asyncio

    state = asyncio.run(materialize_state())
    model = NetworkDelayModel(calibration_rate=1.0)
    query = {"source": "edge-1", "target": "fog-1", "size_bytes": 100_000}
    baseline = model.predict(query, state)
    observed = float(baseline.estimate) * 2.0

    model.observe(
        Event(
            kind=EventKind.DATA_TRANSFER_COMPLETED,
            event_time=1.0,
            source="runtime.real",
            payload={
                "source_node_id": "edge-1",
                "target_node_id": "fog-1",
                "size_bytes": 100_000,
                "duration_s": observed,
            },
        ),
        state,
    )
    calibrated = model.predict(query, state)

    assert calibrated.metadata["calibrated"] is True
    assert calibrated.metadata["samples"] == 1
    assert calibrated.estimate == observed
    assert calibrated.uncertainty < baseline.uncertainty + 0.2


def test_network_calibration_snapshot_roundtrip(small_system):
    from darpan import Darpan

    async def materialize_state():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        state = session.state
        await session.close()
        return state

    import asyncio

    state = asyncio.run(materialize_state())
    query = {"source": "edge-1", "target": "fog-1", "size_bytes": 1_000}
    model = NetworkDelayModel(calibration_rate=1.0)
    baseline = model.predict(query, state)
    model.observe(
        Event(
            kind=EventKind.DATA_TRANSFER_COMPLETED,
            event_time=1.0,
            source="runtime.real",
            payload={
                "source_node_id": "edge-1",
                "target_node_id": "fog-1",
                "size_bytes": 1_000,
                "duration_s": float(baseline.estimate) * 1.5,
            },
        ),
        state,
    )

    restored = NetworkDelayModel()
    restored.restore(model.snapshot())

    assert restored.predict(query, state).estimate == model.predict(query, state).estimate
    assert restored.min_scale == model.min_scale
    assert restored.max_scale == model.max_scale


def test_network_calibration_supports_large_physical_to_topology_scale(small_system):
    from darpan import Darpan

    async def materialize_state():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        state = session.state
        await session.close()
        return state

    import asyncio

    state = asyncio.run(materialize_state())
    model = NetworkDelayModel(calibration_rate=1.0)
    query = {"source": "edge-1", "target": "fog-1", "size_bytes": 100_000}
    baseline = model.predict(query, state)
    observed = float(baseline.estimate) * 300.0
    model.observe(
        Event(
            kind=EventKind.DATA_TRANSFER_COMPLETED,
            event_time=1.0,
            source="runtime.real",
            payload={
                "source_node_id": "edge-1",
                "target_node_id": "fog-1",
                "size_bytes": 100_000,
                "duration_s": observed,
            },
        ),
        state,
    )

    assert model.predict(query, state).estimate == observed


def test_network_calibration_restores_legacy_scale_bounds():
    model = NetworkDelayModel()
    model.restore({"calibration_rate": 0.25, "scales": {}})

    assert model.min_scale == 0.05
    assert model.max_scale == 20.0


def test_network_model_skips_contention_contaminated_calibration_sample(small_system):
    from darpan import Darpan

    async def materialize_state():
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        state = session.state
        await session.close()
        return state

    import asyncio

    state = asyncio.run(materialize_state())
    model = NetworkDelayModel(calibration_rate=1.0)
    query = {"source": "edge-1", "target": "fog-1", "size_bytes": 100_000}
    baseline = model.predict(query, state)
    model.observe(
        Event(
            kind=EventKind.DATA_TRANSFER_COMPLETED,
            event_time=1.0,
            source="runtime.real",
            payload={
                "source_node_id": "edge-1",
                "target_node_id": "fog-1",
                "size_bytes": 100_000,
                "duration_s": float(baseline.estimate) * 4.0,
                "network_calibration_eligible": False,
            },
        ),
        state,
    )
    prediction = model.predict(query, state)
    assert prediction.metadata["calibrated"] is False
    assert prediction.metadata["samples"] == 0
    assert prediction.estimate == baseline.estimate
