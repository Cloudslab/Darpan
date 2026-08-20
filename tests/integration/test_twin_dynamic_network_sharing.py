from __future__ import annotations

import asyncio

import pytest

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import EventKind
from darpan.core.measurement import Measurement
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.telemetry import measurement_event


def test_active_twin_transfer_reacts_to_live_bandwidth_measurement():
    async def run():
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 2),)),
                NodeSpec("cloud", resources=(ResourceSpec("cpu", 2),)),
            ),
            links=(LinkSpec("uplink", "edge", "cloud", bandwidth_mbps=8),),
        )
        app = ApplicationSpec(
            "dynamic-network",
            components=(
                ComponentSpec(
                    "source",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.0,
                ),
                ComponentSpec(
                    "target",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.0,
                ),
                ComponentSpec("keep-ready", work_units=0.0),
            ),
            flows=(
                FlowSpec(
                    "source",
                    "target",
                    data_size_bytes=1_000_000,
                    artifact="payload.bin",
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            await session.apply(Action.place(f"{instance}:source", "edge"))
            await session.advance()
            assert {item.component_id for item in session.state.ready_components()} == {
                "target",
                "keep-ready",
            }

            await session.apply(Action.place(f"{instance}:target", "cloud"))
            transfer_start = next(
                event.event_time
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_STARTED
            )
            backend = session.backend
            backend.schedule_event(
                measurement_event(
                    Measurement(
                        "network.bandwidth_mbps",
                        4.0,
                        unit="Mbit/s",
                        target="uplink",
                        timestamp=transfer_start + 0.25,
                        source="test",
                    ),
                    source="test",
                ),
                transfer_start + 0.25,
            )
            await session.advance()
            completed = next(
                event
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_COMPLETED
            )
            # 2 Mbit move in the first 0.25 s at 8 Mbit/s. The remaining
            # 6 Mbit need another 1.5 s after the link drops to 4 Mbit/s.
            assert completed.payload["duration_s"] == pytest.approx(1.75)
            assert completed.event_time - transfer_start == pytest.approx(1.75)
            assert session.state.links["uplink"].measurements[
                "network.bandwidth_mbps"
            ].value == 4.0
        finally:
            await session.close()

    asyncio.run(run())
