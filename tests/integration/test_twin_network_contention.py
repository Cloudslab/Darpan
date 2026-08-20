from __future__ import annotations

import asyncio

import pytest

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec


def test_twin_max_min_shares_bandwidth_on_shared_link():
    async def run():
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 4),)),
                NodeSpec("cloud", resources=(ResourceSpec("cpu", 4),)),
            ),
            links=(
                LinkSpec("shared", "edge", "cloud", latency_ms=0, bandwidth_mbps=8),
            ),
        )
        app = ApplicationSpec(
            "network-contention",
            components=tuple(
                ComponentSpec(
                    name,
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.0,
                )
                for name in ("s1", "s2", "t1", "t2")
            ),
            flows=(
                FlowSpec("s1", "t1", data_size_bytes=1_000_000, artifact="a.bin"),
                FlowSpec("s2", "t2", data_size_bytes=1_000_000, artifact="b.bin"),
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply_many(
            [Action.place(f"{instance}:s1", "edge"), Action.place(f"{instance}:s2", "edge")]
        )
        await session.advance()
        assert {item.component_id for item in session.state.ready_components()} == {"t1", "t2"}
        await session.apply_many(
            [Action.place(f"{instance}:t1", "cloud"), Action.place(f"{instance}:t2", "cloud")]
        )
        await session.advance()
        transfers = [
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_STARTED
        ]
        assert len(transfers) == 2
        assert transfers[1].event_time == pytest.approx(transfers[0].event_time)
        completed = [
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_COMPLETED
        ]
        assert len(completed) == 2
        assert completed[0].event_time == pytest.approx(completed[1].event_time)
        assert completed[0].payload["duration_s"] == pytest.approx(2.0)
        assert completed[1].payload["duration_s"] == pytest.approx(2.0)
        assert completed[0].payload["sharing_delay_s"] == pytest.approx(1.0)
        assert completed[1].payload["sharing_delay_s"] == pytest.approx(1.0)
        await session.close()

    asyncio.run(run())
