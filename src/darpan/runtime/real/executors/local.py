"""Local subprocess executor with isolated per-component workspaces."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path, PurePosixPath
from time import perf_counter

from darpan.core.application import ComponentSpec
from darpan.core.protocols.executor import ExecutionResult


class LocalExecutor:
    def __init__(self, *, workspace_root: str | Path | None = None) -> None:
        self._temporary_root = None
        if workspace_root is None:
            self._temporary_root = tempfile.TemporaryDirectory(prefix="darpan-workspaces-")
            workspace_root = self._temporary_root.name
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _workspace(self, workspace_id: str) -> Path:
        digest = hashlib.sha256(workspace_id.encode()).hexdigest()
        path = self.workspace_root / digest
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_path(self, workspace_id: str) -> Path:
        """Return the isolated host path backing one component workspace."""

        return self._workspace(workspace_id)

    @staticmethod
    def _relative(path: str) -> Path:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or str(pure) in {"", "."}:
            raise ValueError("artifact path must be a safe relative path")
        return Path(*pure.parts)

    def _artifact_path(self, workspace_id: str, path: str) -> Path:
        return self._workspace(workspace_id) / self._relative(path)

    async def execute(self, component: ComponentSpec) -> ExecutionResult:
        return await self._execute(component, cwd=None)

    async def execute_in_workspace(
        self,
        component: ComponentSpec,
        workspace_id: str,
    ) -> ExecutionResult:
        return await self._execute(component, cwd=self._workspace(workspace_id))

    async def _execute(
        self,
        component: ComponentSpec,
        *,
        cwd: Path | None,
    ) -> ExecutionResult:
        started = perf_counter()
        if not component.command:
            return ExecutionResult(return_code=0, duration_s=0.0)
        process = await asyncio.create_subprocess_exec(
            *component.command,
            cwd=None if cwd is None else str(cwd),
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
        duration = perf_counter() - started
        return ExecutionResult(
            return_code=int(process.returncode or 0),
            duration_s=duration,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            output_bytes=len(stdout),
        )

    async def put_file(self, workspace_id: str, path: str, data: bytes) -> None:
        target = self._artifact_path(workspace_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def get_file(self, workspace_id: str, path: str) -> bytes:
        return self._artifact_path(workspace_id, path).read_bytes()

    async def stat_file(self, workspace_id: str, path: str) -> int:
        return self._artifact_path(workspace_id, path).stat().st_size

    async def delete_file(self, workspace_id: str, path: str) -> bool:
        target = self._artifact_path(workspace_id, path)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        return True

    async def write_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        data: bytes,
        truncate: bool = False,
    ) -> int:
        target = self._artifact_path(workspace_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if truncate else "r+b"
        if not target.exists() and mode == "r+b":
            mode = "wb"
        with target.open(mode) as handle:
            handle.seek(offset)
            handle.write(data)
        return target.stat().st_size

    async def read_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> tuple[bytes, int]:
        target = self._artifact_path(workspace_id, path)
        size = target.stat().st_size
        with target.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(limit)
        return data, size
