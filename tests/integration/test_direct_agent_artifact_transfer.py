from __future__ import annotations

import asyncio
import sys

import pytest

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    FlowSpec,
    LinkSpec,
    NodeSpec,
    SystemSpec,
)
from darpan.core.event import EventKind
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.executors.remote import RemoteExecutor
from darpan.runtime.real.transport import AgentClient
from darpan.runtime.session import Session


class NoControllerReadClient(AgentClient):
    async def read_chunk(self, workspace_id, path, *, offset, limit):
        raise AssertionError("controller must not relay direct agent-to-agent artifact bytes")


def _app(size: int = 25_000) -> ApplicationSpec:
    return ApplicationSpec(
        "direct-artifact",
        components=(
            ComponentSpec(
                "source",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path('out.bin').write_bytes(b'x'*{size})",
                ),
            ),
            ComponentSpec(
                "target",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    f"assert len(Path('input.bin').read_bytes()) == {size}",
                ),
            ),
        ),
        flows=(
            FlowSpec(
                "source",
                "target",
                data_size_bytes=size,
                artifact="out.bin",
                target_path="input.bin",
            ),
        ),
    )


def test_direct_agent_artifact_transfer_uses_one_time_ticket_not_controller_relay():
    async def run():
        source_server = AgentServer(
            "source",
            host="127.0.0.1",
            port=0,
            token="source-secret",
            allow_artifact_forward=True,
        )
        target_server = AgentServer(
            "target",
            host="127.0.0.1",
            port=0,
            token="target-secret",
        )
        await source_server.start()
        await target_server.start()
        source_client = NoControllerReadClient(
            "127.0.0.1",
            source_server.port,
            token="source-secret",
            artifact_chunk_size=2048,
        )
        target_client = AgentClient(
            "127.0.0.1",
            target_server.port,
            token="target-secret",
            artifact_chunk_size=2048,
        )
        session = Session(
            RealBackend(
                executors={
                    "source": RemoteExecutor(
                        source_client,
                        direct_artifact_forward=True,
                    ),
                    "target": RemoteExecutor(target_client),
                },
                artifact_chunk_size=2048,
            )
        )
        try:
            await session.start()
            await session.register_system(
                SystemSpec(
                    nodes=(NodeSpec("source"), NodeSpec("target")),
                    links=(LinkSpec("wire", "source", "target"),),
                )
            )
            instance = await session.submit_application(_app())
            await session.apply(Action.place(f"{instance}:source", "source"))
            await session.wait_for(
                lambda event, state: event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:target",
                timeout=5,
            )
            await session.apply(Action.place(f"{instance}:target", "target"))
            await session.wait_for(
                lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
                timeout=5,
            )
            transfer = next(
                event
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_COMPLETED
            )
            assert transfer.payload["transport"] == "agent-direct"
            assert transfer.payload["logical_links"] == ["wire"]
            assert transfer.payload["size_bytes"] == 25_000
            assert transfer.payload["network_path_representative"] is True
            assert transfer.payload["network_calibration_eligible"] is True
            assert transfer.payload["measurement_scope"] == "source-agent-data-plane"
            assert transfer.payload["duration_s"] <= transfer.payload["control_duration_s"]
            assert transfer.payload["control_overhead_s"] >= 0.0
        finally:
            await session.close()
            await source_server.close()
            await target_server.close()

    asyncio.run(run())


def test_direct_transfer_ticket_is_bound_to_target_path():
    async def run():
        target = AgentServer(
            "target",
            host="127.0.0.1",
            port=0,
            token="target-secret",
        )
        await target.start()
        authenticated = AgentClient(
            "127.0.0.1",
            target.port,
            token="target-secret",
        )
        ticket = await authenticated.prepare_incoming_transfer(
            "workspace",
            "expected.bin",
            size_bytes=3,
        )
        ticket_client = AgentClient("127.0.0.1", target.port)
        try:
            with pytest.raises(RuntimeError, match="does not match target path"):
                await ticket_client.write_transfer_chunk(
                    ticket=ticket,
                    workspace_id="workspace",
                    path="other.bin",
                    offset=0,
                    data=b"abc",
                    truncate=True,
                    final=True,
                )
            await ticket_client.write_transfer_chunk(
                ticket=ticket,
                workspace_id="workspace",
                path="expected.bin",
                offset=0,
                data=b"abc",
                truncate=True,
                final=True,
            )
            assert await authenticated.get_file("workspace", "expected.bin") == b"abc"
        finally:
            await target.close()

    asyncio.run(run())


def test_direct_agent_transfer_supports_default_chunk_above_asyncio_line_limit():
    async def run():
        size = 100_000
        source_server = AgentServer(
            "source",
            host="127.0.0.1",
            port=0,
            allow_artifact_forward=True,
        )
        target_server = AgentServer("target", host="127.0.0.1", port=0)
        await source_server.start()
        await target_server.start()
        source_client = AgentClient("127.0.0.1", source_server.port)
        target_client = AgentClient("127.0.0.1", target_server.port)
        session = Session(
            RealBackend(
                executors={
                    "source": RemoteExecutor(source_client, direct_artifact_forward=True),
                    "target": RemoteExecutor(target_client),
                }
            )
        )
        try:
            await session.start()
            await session.register_system(
                SystemSpec(
                    nodes=(NodeSpec("source"), NodeSpec("target")),
                    links=(LinkSpec("wire", "source", "target"),),
                )
            )
            instance = await session.submit_application(_app(size=size))
            await session.apply(Action.place(f"{instance}:source", "source"))
            await session.wait_for(
                lambda event, state: event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:target",
                timeout=5,
            )
            await session.apply(Action.place(f"{instance}:target", "target"))
            completed = await session.wait_for(
                lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
                timeout=5,
            )
            assert completed.event.payload["success"] is True
            transfer = next(
                event
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_COMPLETED
            )
            assert transfer.payload["size_bytes"] == size
            assert transfer.payload["transport"] == "agent-direct"
        finally:
            await session.close()
            await source_server.close()
            await target_server.close()

    asyncio.run(run())
