from __future__ import annotations

import asyncio

import pytest

from darpan import Darpan
from darpan.core.measurement import Measurement
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.telemetry import measurement_event


def test_link_measurements_materialize_in_state_and_drive_network_prediction():
    async def run():
        session = Darpan.twin()
        system = SystemSpec(
            nodes=(NodeSpec("edge"), NodeSpec("cloud", tier="cloud")),
            links=(
                LinkSpec(
                    "uplink",
                    "edge",
                    "cloud",
                    latency_ms=100,
                    bandwidth_mbps=10,
                ),
            ),
        )
        await session.start()
        await session.register_system(system)
        model = session.backend.models.get("network")
        before = model.predict(
            {"source": "edge", "target": "cloud", "size_bytes": 1_000_000},
            session.state,
        )
        await session.emit(
            measurement_event(
                Measurement(
                    "network.latency_ms",
                    10.0,
                    unit="ms",
                    target="uplink",
                    timestamp=session.clock.now(),
                    source="probe",
                ),
                source="probe",
            )
        )
        await session.emit(
            measurement_event(
                Measurement(
                    "network.bandwidth_mbps",
                    100.0,
                    unit="Mbps",
                    target="uplink",
                    timestamp=session.clock.now(),
                    source="probe",
                ),
                source="probe",
            )
        )
        assert session.state.links["uplink"].measurements["network.latency_ms"].value == 10.0
        after = model.predict(
            {"source": "edge", "target": "cloud", "size_bytes": 1_000_000},
            session.state,
        )
        assert before.estimate == pytest.approx(0.9)
        assert after.estimate == pytest.approx(0.09)
        assert after.metadata["telemetry"]["uplink"]["bandwidth_mbps"] == 100.0
        await session.close()

    asyncio.run(run())
