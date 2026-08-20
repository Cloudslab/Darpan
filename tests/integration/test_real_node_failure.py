from __future__ import annotations

import asyncio

from darpan import Action, ApplicationSpec, ComponentSpec, Darpan, FlowSpec
from darpan.core.event import EventKind
from darpan.core.protocols.executor import ExecutionResult
from darpan.core.resource import ResourceRequest
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.cluster.monitor import ClusterMonitor


class FlappingClient:
    def __init__(self) -> None:
        self.up = True

    async def ping(self):
        if not self.up:
            raise ConnectionError("agent unreachable")
        return {"node_id": "edge-1"}

    async def telemetry(self):
        if not self.up:
            raise ConnectionError("agent unreachable")
        return {"node_id": "edge-1"}


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def execute(self, component):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class ArtifactExecutor:
    def __init__(self, *, block_reads: bool = False) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.block_reads = block_reads
        self.transfer_started = asyncio.Event()
        self.transfer_cancelled = False

    async def execute_in_workspace(self, component, workspace_id: str):
        if component.id == "source":
            self.files[(workspace_id, "payload.bin")] = b"x" * 4096
        return ExecutionResult(return_code=0, duration_s=0.01)

    async def stat_file(self, workspace_id: str, path: str) -> int:
        return len(self.files[(workspace_id, path)])

    async def read_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> tuple[bytes, int]:
        data = self.files[(workspace_id, path)]
        if self.block_reads:
            self.transfer_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.transfer_cancelled = True
                raise
        return data[offset : offset + limit], len(data)

    async def write_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        data: bytes,
        truncate: bool,
    ) -> None:
        key = (workspace_id, path)
        current = b"" if truncate else self.files.get(key, b"")
        if len(current) < offset:
            current += b"\0" * (offset - len(current))
        self.files[key] = current[:offset] + data + current[offset + len(data) :]


async def _mark_offline(session, client: FlappingClient) -> None:
    monitor = ClusterMonitor(
        session,
        {"edge-1": client},
        interval_s=60,
        emit_telemetry=False,
    )
    await monitor.poll_once()
    client.up = False
    await monitor.poll_once()


def test_real_running_component_is_cancelled_and_failed_on_agent_loss(small_system) -> None:
    async def run() -> None:
        executor = BlockingExecutor()
        backend = RealBackend(executors={"edge-1": executor})
        session = Darpan.real(backend=backend)
        client = FlappingClient()
        app = ApplicationSpec(
            "real-node-failure",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=1,
                ),
            ),
        )
        await session.start()
        try:
            await session.register_system(small_system)
            instance = await session.submit_application(app)
            task_id = f"{instance}:task"
            await session.apply(Action.place(task_id, "edge-1"))
            await asyncio.wait_for(executor.started.wait(), timeout=1)

            await _mark_offline(session, client)

            assert executor.cancelled is True
            assert session.state.nodes["edge-1"].status == "offline"
            assert session.state.components[task_id].status == "failed"
            assert session.state.nodes["edge-1"].resources["cpu"].allocated == 0
            failed = next(
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_FAILED
                and event.payload.get("instance_id") == task_id
            )
            assert failed.payload["failure_kind"] == "node_offline"

            client.up = True
            monitor = ClusterMonitor(
                session,
                {"edge-1": client},
                interval_s=60,
                emit_telemetry=False,
            )
            # Recreate the monitor's prior offline observation to exercise the
            # canonical recovery event without restarting the failed work.
            monitor._online["edge-1"] = False
            await monitor.poll_once()
            assert session.state.nodes["edge-1"].status == "online"
            assert session.state.components[task_id].status == "failed"
        finally:
            await session.close()

    asyncio.run(run())


def test_real_source_node_loss_cancels_inflight_artifact_transfer(small_system) -> None:
    async def run() -> None:
        source = ArtifactExecutor(block_reads=True)
        target = ArtifactExecutor()
        backend = RealBackend(
            executors={
                "edge-1": source,
                "cloud-1": target,
            }
        )
        session = Darpan.real(backend=backend)
        client = FlappingClient()
        app = ApplicationSpec(
            "real-transfer-node-failure",
            components=(
                ComponentSpec(
                    "source",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0,
                ),
                ComponentSpec(
                    "target",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0,
                ),
            ),
            flows=(
                FlowSpec(
                    "source",
                    "target",
                    data_size_bytes=4096,
                    artifact="payload.bin",
                ),
            ),
        )
        await session.start()
        try:
            await session.register_system(small_system)
            instance = await session.submit_application(app)
            await session.apply(Action.place(f"{instance}:source", "edge-1"))
            await session.wait_for(
                lambda event, state: event.kind == EventKind.COMPONENT_READY
                and event.subject == f"{instance}:target",
                timeout=1,
            )
            target_id = f"{instance}:target"
            await session.apply(Action.place(target_id, "cloud-1"))
            await asyncio.wait_for(source.transfer_started.wait(), timeout=1)

            await _mark_offline(session, client)

            assert source.transfer_cancelled is True
            assert session.state.components[target_id].status == "failed"
            transfer_failed = next(
                event
                for event in session.event_log
                if event.kind == EventKind.DATA_TRANSFER_FAILED
                and event.payload.get("target_instance_id") == target_id
            )
            assert transfer_failed.payload["failure_kind"] == "node_offline"
            assert transfer_failed.payload["network_calibration_eligible"] is False
            assert not any(
                event.kind == EventKind.DATA_TRANSFER_COMPLETED
                and event.payload.get("target_instance_id") == target_id
                for event in session.event_log
            )
        finally:
            await session.close()

    asyncio.run(run())
