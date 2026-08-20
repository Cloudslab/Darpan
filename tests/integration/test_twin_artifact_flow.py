from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import EventKind


def test_twin_emits_canonical_events_for_declared_artifact_flow(small_system):
    async def run():
        app = ApplicationSpec(
            "artifact-twin",
            components=(ComponentSpec("source"), ComponentSpec("target")),
            flows=(
                FlowSpec(
                    "source",
                    "target",
                    data_size_bytes=50_000,
                    artifact="result.bin",
                    target_path="input.bin",
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(app)
        await session.apply(Action.place(f"{instance}:source", "edge-1"))
        await session.apply(Action.place(f"{instance}:target", "fog-1"))

        started = next(
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_STARTED
        )
        completed = next(
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_COMPLETED
        )
        component_started = next(
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_STARTED
            and event.subject == f"{instance}:target"
        )

        assert started.payload["artifact"] == "result.bin"
        assert completed.payload["target_path"] == "input.bin"
        assert completed.payload["size_bytes"] == 50_000
        assert completed.payload["predicted"] is True
        assert completed.event_time <= component_started.event_time
        await session.close()

    asyncio.run(run())
