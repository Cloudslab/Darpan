from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import Event, EventKind
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec


def test_active_twin_artifact_transfer_fails_when_selected_link_goes_down():
    async def run():
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 2),)),
                NodeSpec("cloud", resources=(ResourceSpec("cpu", 2),)),
            ),
            links=(LinkSpec("uplink", "edge", "cloud", bandwidth_mbps=8),),
        )
        app = ApplicationSpec(
            "link-failure",
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
            await session.apply(Action.place(f"{instance}:target", "cloud"))
            started = next(
                event.event_time
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_STARTED
            )
            session.backend.schedule_event(
                Event(
                    kind=EventKind.LINK_REMOVED,
                    event_time=started + 0.25,
                    source="test",
                    subject="uplink",
                    payload={"link_id": "uplink"},
                ),
                started + 0.25,
            )
            await session.advance()
            assert any(
                event.kind == EventKind.DATA_TRANSFER_FAILED
                for event in session.event_log
            )
            target = session.state.components[f"{instance}:target"]
            assert target.status == "failed"
            assert session.state.links["uplink"].status == "down"
        finally:
            await session.close()

    asyncio.run(run())
