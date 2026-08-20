from __future__ import annotations

import asyncio
import sys

import pytest

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    FlowSpec,
    NodeSpec,
    ResourceSpec,
    SystemSpec,
)
from darpan.core.event import EventKind
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.executors.remote import RemoteExecutor
from darpan.runtime.real.transport import AgentClient
from darpan.runtime.session import Session


def _artifact_app(size: int = 300_000) -> ApplicationSpec:
    source_code = (
        "from pathlib import Path; "
        f"Path('out.bin').write_bytes(b'x' * {size})"
    )
    target_code = (
        "from pathlib import Path; "
        "data=Path('input.bin').read_bytes(); "
        f"assert len(data)=={size}; "
        "assert data[:1] == b'x'; print('artifact-ok')"
    )
    return ApplicationSpec(
        "artifact-app",
        components=(
            ComponentSpec("producer", command=(sys.executable, "-c", source_code)),
            ComponentSpec("consumer", command=(sys.executable, "-c", target_code)),
        ),
        flows=(
            FlowSpec(
                "producer",
                "consumer",
                data_size_bytes=size,
                artifact="out.bin",
                target_path="input.bin",
            ),
        ),
    )


def _two_nodes() -> SystemSpec:
    return SystemSpec(
        nodes=(
            NodeSpec("n1", resources=(ResourceSpec("cpu", 1),)),
            NodeSpec("n2", resources=(ResourceSpec("cpu", 1),)),
        )
    )


def test_local_real_runtime_transfers_declared_file_artifact_between_nodes():
    async def run():
        session = Session(RealBackend())
        app = _artifact_app()
        await session.start()
        await session.register_system(_two_nodes())
        instance = await session.submit_application(app)
        await session.apply(Action.place(f"{instance}:producer", "n1"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{instance}:consumer",
            timeout=3,
        )
        await session.apply(Action.place(f"{instance}:consumer", "n2"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3,
        )
        started_transfers = [
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_STARTED
        ]
        assert len(started_transfers) == 1
        assert started_transfers[0].payload["size_bytes"] == 300_000
        transfers = [
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_COMPLETED
        ]
        assert len(transfers) == 1
        assert transfers[0].payload["size_bytes"] == 300_000
        producer_completed = next(
            event
            for event in session.event_log
            if event.kind == EventKind.COMPONENT_COMPLETED
            and event.subject == f"{instance}:producer"
        )
        assert producer_completed.payload["output_bytes"] == 300_000
        assert session.state.components[f"{instance}:consumer"].status == "completed"
        await session.close()

    asyncio.run(run())


def test_agent_artifact_transport_is_chunked_and_lossless():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient(
            "127.0.0.1",
            server.port,
            artifact_chunk_size=1024,
        )
        data = bytes(range(256)) * 40
        await client.put_file("workspace", "nested/data.bin", data)
        assert await client.stat_file("workspace", "nested/data.bin") == len(data)
        assert await client.get_file("workspace", "nested/data.bin") == data
        await server.close()

    asyncio.run(run())


def test_remote_agents_execute_cross_node_file_flow_with_chunked_transfer():
    async def run():
        first = AgentServer("n1", host="127.0.0.1", port=0)
        second = AgentServer("n2", host="127.0.0.1", port=0)
        await first.start()
        await second.start()
        backend = RealBackend(
            executors={
                "n1": RemoteExecutor(
                    AgentClient(
                        "127.0.0.1",
                        first.port,
                        artifact_chunk_size=2048,
                    )
                ),
                "n2": RemoteExecutor(
                    AgentClient(
                        "127.0.0.1",
                        second.port,
                        artifact_chunk_size=2048,
                    )
                ),
            }
        )
        session = Session(backend)
        app = _artifact_app(size=25_000)
        try:
            await session.start()
            await session.register_system(_two_nodes())
            instance = await session.submit_application(app)
            await session.apply(Action.place(f"{instance}:producer", "n1"))
            await session.wait_for(
                lambda event, state: event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:consumer",
                timeout=5,
            )
            await session.apply(Action.place(f"{instance}:consumer", "n2"))
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
            assert transfer.payload["size_bytes"] == 25_000
            assert transfer.payload["transport"] == "controller-stream"
            assert transfer.payload["network_path_representative"] is False
            assert transfer.payload["network_calibration_eligible"] is False
        finally:
            await session.close()
            await first.close()
            await second.close()

    asyncio.run(run())


def test_real_backend_streams_artifact_in_bounded_chunks_before_allocation(tmp_path):
    from darpan.runtime.real.executors.local import LocalExecutor

    class RecordingLocalExecutor(LocalExecutor):
        def __init__(self, root):
            super().__init__(workspace_root=root)
            self.read_sizes = []
            self.write_sizes = []

        async def read_chunk(self, workspace_id, path, *, offset, limit):
            data, size = await super().read_chunk(
                workspace_id,
                path,
                offset=offset,
                limit=limit,
            )
            self.read_sizes.append(len(data))
            return data, size

        async def write_chunk(
            self,
            workspace_id,
            path,
            *,
            offset,
            data,
            truncate=False,
        ):
            self.write_sizes.append(len(data))
            return await super().write_chunk(
                workspace_id,
                path,
                offset=offset,
                data=data,
                truncate=truncate,
            )

    async def run():
        source = RecordingLocalExecutor(tmp_path / "source")
        target = RecordingLocalExecutor(tmp_path / "target")
        backend = RealBackend(
            executors={"n1": source, "n2": target},
            artifact_chunk_size=1024,
        )
        from dataclasses import replace

        from darpan import ResourceRequest

        session = Session(backend)
        base_app = _artifact_app(size=10_000)
        app = ApplicationSpec(
            base_app.id,
            components=tuple(
                replace(component, resources=(ResourceRequest("cpu", 1),))
                for component in base_app.components
            ),
            flows=base_app.flows,
        )
        await session.start()
        await session.register_system(_two_nodes())
        instance = await session.submit_application(app)
        await session.apply(Action.place(f"{instance}:producer", "n1"))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:consumer"
            ),
            timeout=3,
        )
        before = session.event_count
        await session.apply(Action.place(f"{instance}:consumer", "n2"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3,
        )

        assert len(source.read_sizes) > 1
        assert max(source.read_sizes) <= 1024
        assert len(target.write_sizes) > 1
        assert max(target.write_sizes) <= 1024

        events = session.events_since(before)
        transfer_index = next(
            index
            for index, event in enumerate(events)
            if event.kind == EventKind.DATA_TRANSFER_COMPLETED
        )
        allocation_index = next(
            index
            for index, event in enumerate(events)
            if event.kind == EventKind.RESOURCE_ALLOCATED
            and event.subject == "n2"
        )
        started_index = next(
            index
            for index, event in enumerate(events)
            if event.kind == EventKind.COMPONENT_STARTED
            and event.subject == f"{instance}:consumer"
        )
        assert transfer_index < allocation_index < started_index
        await session.close()

    asyncio.run(run())


def test_agent_default_artifact_chunk_exceeds_asyncio_default_line_limit_safely():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port)
        data = bytes(range(256)) * 1200  # 307,200 bytes; base64 is far above 64 KiB.
        try:
            await client.put_file("workspace", "large-default-chunk.bin", data)
            assert await client.get_file("workspace", "large-default-chunk.bin") == data
        finally:
            await server.close()

    asyncio.run(run())


def test_agent_transport_rejects_chunks_above_protocol_limit() -> None:
    from darpan.runtime.real.backend import RealBackend
    from darpan.runtime.real.transport import MAX_AGENT_BINARY_PAYLOAD_BYTES, AgentClient

    too_large = MAX_AGENT_BINARY_PAYLOAD_BYTES + 1
    with pytest.raises(ValueError, match="artifact_chunk_size must be in"):
        AgentClient("127.0.0.1", artifact_chunk_size=too_large)
    with pytest.raises(ValueError, match="artifact_chunk_size must be in"):
        RealBackend(artifact_chunk_size=too_large)

    async def run() -> None:
        client = AgentClient("127.0.0.1")
        with pytest.raises(ValueError, match="artifact chunk exceeds"):
            await client.write_chunk(
                "workspace",
                "oversized.bin",
                offset=0,
                data=b"x" * too_large,
                truncate=True,
            )

    asyncio.run(run())
