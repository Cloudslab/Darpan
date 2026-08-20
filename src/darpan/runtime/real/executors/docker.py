"""Docker CLI executor with artifact-capable per-component workspaces."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter

from darpan.core.application import ComponentSpec
from darpan.core.protocols.executor import ExecutionResult

from .local import LocalExecutor


class DockerExecutor:
    """Execute container components without making Docker a Core dependency.

    Workspace-aware execution mounts the same isolated host directory used by
    Darpan artifact transfer at ``/darpan/workspace``.  This makes file flows
    work identically for local processes and containers and keeps all artifact
    bytes outside the container lifecycle.
    """

    container_workspace = "/darpan/workspace"

    def __init__(
        self,
        *,
        executable: str = "docker",
        workspace_root: str | Path | None = None,
    ) -> None:
        self.executable = executable
        self.workspace = LocalExecutor(workspace_root=workspace_root)

    async def execute(self, component: ComponentSpec) -> ExecutionResult:
        return await self._execute(component, workspace_id=None)

    async def execute_in_workspace(
        self,
        component: ComponentSpec,
        workspace_id: str,
    ) -> ExecutionResult:
        return await self._execute(component, workspace_id=workspace_id)

    async def _execute(
        self,
        component: ComponentSpec,
        *,
        workspace_id: str | None,
    ) -> ExecutionResult:
        if not component.image:
            raise ValueError("DockerExecutor requires component.image")
        command = [self.executable, "run", "--rm"]
        cpu_request = next(
            (request.amount for request in component.resources if request.name == "cpu"),
            None,
        )
        if cpu_request is not None and cpu_request > 0:
            command.extend(["--cpus", str(cpu_request)])
        if workspace_id is not None:
            host_workspace = self.workspace.workspace_path(workspace_id)
            command.extend(
                [
                    "--volume",
                    f"{host_workspace}:{self.container_workspace}",
                    "--workdir",
                    self.container_workspace,
                ]
            )
        command.extend([component.image, *component.command])
        started = perf_counter()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            raise
        return ExecutionResult(
            return_code=int(process.returncode or 0),
            duration_s=perf_counter() - started,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            output_bytes=len(stdout),
        )

    async def put_file(self, workspace_id: str, path: str, data: bytes) -> None:
        await self.workspace.put_file(workspace_id, path, data)

    async def get_file(self, workspace_id: str, path: str) -> bytes:
        return await self.workspace.get_file(workspace_id, path)

    async def stat_file(self, workspace_id: str, path: str) -> int:
        return await self.workspace.stat_file(workspace_id, path)

    async def delete_file(self, workspace_id: str, path: str) -> bool:
        return await self.workspace.delete_file(workspace_id, path)

    async def write_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        data: bytes,
        truncate: bool = False,
    ) -> int:
        return await self.workspace.write_chunk(
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
        return await self.workspace.read_chunk(
            workspace_id,
            path,
            offset=offset,
            limit=limit,
        )
