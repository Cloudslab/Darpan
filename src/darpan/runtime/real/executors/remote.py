"""Executor adapter routing component execution to a Darpan node agent."""

from __future__ import annotations

import asyncio

from darpan.core.application import ComponentSpec
from darpan.core.protocols.executor import ExecutionResult
from darpan.core.serialization import to_primitive

from ..transport import AgentClient


class RemoteExecutionHandle:
    """One Agent-side execution whose lifetime survives controller RPC calls."""

    wait_attempts = 3
    wait_retry_delay_s = 0.2

    def __init__(self, client: AgentClient, execution_id: str) -> None:
        self.client = client
        self.execution_id = execution_id

    async def _release(self) -> None:
        for attempt in range(self.wait_attempts):
            try:
                await self.client.release_execution(self.execution_id)
                return
            except OSError:
                if attempt + 1 == self.wait_attempts:
                    return
                await asyncio.sleep(self.wait_retry_delay_s * (attempt + 1))
            except Exception:
                # Release is cleanup after a result was already received. An old
                # Agent or a disappearing node must not discard that valid result.
                return

    async def wait(self) -> ExecutionResult:
        try:
            connection_failures = 0
            while True:
                try:
                    response = await self.client.wait_execution(self.execution_id)
                    break
                except TimeoutError:
                    # ``execute.wait`` is an idempotent long-poll. A timeout
                    # means "poll again", not that the remote process failed.
                    continue
                except OSError:
                    connection_failures += 1
                    if connection_failures == self.wait_attempts:
                        raise
                    await asyncio.sleep(
                        self.wait_retry_delay_s * connection_failures
                    )
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self.client.cancel_execution(self.execution_id))
            finally:
                await asyncio.shield(self._release())
                raise
        try:
            if response.get("cancelled", False):
                raise RuntimeError(
                    f"remote execution was cancelled: {self.execution_id}"
                )
            return RemoteExecutor._result(response["result"])
        finally:
            await self._release()

    async def cancel(self) -> bool:
        try:
            result = await self.client.cancel_execution(self.execution_id)
            return bool(result.get("cancelled", False))
        finally:
            await self._release()


class RemoteExecutor:
    is_remote_agent = True

    def __init__(
        self,
        client: AgentClient,
        *,
        direct_artifact_forward: bool = False,
    ) -> None:
        self.client = client
        self.direct_artifact_forward = direct_artifact_forward

    @staticmethod
    def _result(result) -> ExecutionResult:
        return ExecutionResult(
            return_code=int(result["return_code"]),
            duration_s=float(result["duration_s"]),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            output_bytes=int(result.get("output_bytes", 0)),
            measurements=dict(result.get("measurements", {})),
        )

    async def start_execution(
        self,
        component: ComponentSpec,
        *,
        workspace_id: str | None = None,
    ) -> RemoteExecutionHandle:
        execution_id = await self.client.start_execution(
            to_primitive(component),
            workspace_id=workspace_id,
        )
        return RemoteExecutionHandle(self.client, execution_id)

    async def _execute(
        self,
        component: ComponentSpec,
        *,
        workspace_id: str | None = None,
    ) -> ExecutionResult:
        handle = await self.start_execution(component, workspace_id=workspace_id)
        return await handle.wait()

    async def execute(self, component: ComponentSpec) -> ExecutionResult:
        return await self._execute(component)

    async def execute_in_workspace(
        self,
        component: ComponentSpec,
        workspace_id: str,
    ) -> ExecutionResult:
        return await self._execute(component, workspace_id=workspace_id)

    async def put_file(self, workspace_id: str, path: str, data: bytes) -> None:
        await self.client.put_file(workspace_id, path, data)

    async def get_file(self, workspace_id: str, path: str) -> bytes:
        return await self.client.get_file(workspace_id, path)

    async def stat_file(self, workspace_id: str, path: str) -> int:
        return await self.client.stat_file(workspace_id, path)

    async def delete_file(self, workspace_id: str, path: str) -> bool:
        return await self.client.delete_file(workspace_id, path)

    async def write_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        data: bytes,
        truncate: bool = False,
    ) -> int:
        return await self.client.write_chunk(
            workspace_id,
            path,
            offset=offset,
            data=data,
            truncate=truncate,
        )

    async def read_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> tuple[bytes, int]:
        return await self.client.read_chunk(
            workspace_id,
            path,
            offset=offset,
            limit=limit,
        )

    def can_direct_transfer_to(self, target) -> bool:
        return (
            self.direct_artifact_forward
            and isinstance(target, RemoteExecutor)
        )

    async def copy_file_to(
        self,
        source_workspace: str,
        source_path: str,
        target,
        target_workspace: str,
        target_path: str,
        *,
        chunk_size: int,
        expected_size: int | None = None,
    ) -> tuple[int, float | None]:
        if not self.can_direct_transfer_to(target):
            raise RuntimeError("direct artifact transfer is not enabled for this peer")
        endpoint = target.client.direct_transfer_endpoint()
        if target.client.ssl_context is not None and endpoint.get("ca_pem") is None:
            raise RuntimeError(
                "direct TLS artifact transfer requires target AgentClient.tls_ca_pem"
            )
        size = (
            await self.stat_file(source_workspace, source_path)
            if expected_size is None
            else int(expected_size)
        )
        last_error: OSError | None = None
        for forward_attempt in range(2):
            ticket = await target.client.prepare_incoming_transfer(
                target_workspace,
                target_path,
                size_bytes=size,
            )
            try:
                transferred, duration_s = await self.client.forward_artifact(
                    source_workspace,
                    source_path,
                    target_host=str(endpoint["host"]),
                    target_port=int(endpoint["port"]),
                    target_workspace=target_workspace,
                    target_path=target_path,
                    ticket=ticket,
                    target_ca_pem=endpoint.get("ca_pem"),
                    target_server_hostname=endpoint.get("server_hostname"),
                    chunk_size=chunk_size,
                )
            except OSError as exc:
                last_error = exc
                for check_attempt in range(3):
                    await asyncio.sleep(0.2 * (check_attempt + 1))
                    try:
                        observed_size = await target.stat_file(
                            target_workspace,
                            target_path,
                        )
                    except (OSError, RuntimeError):
                        continue
                    if observed_size == size:
                        # The Agent completed the copy but its completion response
                        # was lost. Size is the existing artifact contract; report
                        # no source-side duration so fidelity calibration excludes it.
                        return size, None
                    if observed_size > size:
                        raise OSError(
                            "direct artifact transfer exceeded its declared size: "
                            f"expected {size}, got {observed_size}"
                        ) from exc
                if forward_attempt == 0:
                    continue
                raise
            if transferred != size:
                raise OSError(
                    "direct artifact transfer size mismatch: "
                    f"expected {size}, got {transferred}"
                )
            return transferred, duration_s
        assert last_error is not None
        raise last_error
