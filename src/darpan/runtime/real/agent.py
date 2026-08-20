"""Lightweight Darpan agent server for physical Continuum nodes."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import platform
import shutil
import ssl
import statistics
import tempfile
from pathlib import Path
from time import monotonic, perf_counter, time
from typing import Any
from uuid import uuid4

from darpan._version import __version__
from darpan.core.codec import component_spec_from_dict
from darpan.core.serialization import to_primitive

from .executors.docker import DockerExecutor
from .executors.local import LocalExecutor
from .telemetry import (
    detect_cgroup_version,
    detect_cpu_capacity,
    detect_memory_capacity_bytes,
    environment_snapshot,
)
from .transport import AGENT_STREAM_LIMIT, MAX_AGENT_BINARY_PAYLOAD_BYTES

AGENT_PROTOCOL_VERSION = 5


class AgentServer:
    def __init__(
        self,
        node_id: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        workspace_root: str | Path | None = None,
        token: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        allow_network_probe: bool = False,
        allow_artifact_forward: bool = False,
        physical_control=None,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.token = token
        self.ssl_context = ssl_context
        self.allow_network_probe = allow_network_probe
        self.allow_artifact_forward = allow_artifact_forward
        self.physical_control = physical_control
        self._workload_enabled = True
        self._temporary_root = None
        if workspace_root is None:
            self._temporary_root = tempfile.TemporaryDirectory(
                prefix=f"darpan-agent-{node_id}-"
            )
            workspace_root = self._temporary_root.name
        self.workspace_root = Path(workspace_root).resolve()
        self.local = LocalExecutor(workspace_root=self.workspace_root)
        self.docker = DockerExecutor(workspace_root=self.workspace_root)
        self._server: asyncio.Server | None = None
        self._executions: dict[str, asyncio.Task] = {}
        self._incoming_transfers: dict[str, dict[str, Any]] = {}
        self._control_lease_task: asyncio.Task | None = None
        self._control_lease_deadline: float | None = None
        self._control_auto_restore: dict[str, Any] | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle,
            self.host,
            self.port,
            ssl=self.ssl_context,
            limit=AGENT_STREAM_LIMIT,
        )
        if self._server.sockets:
            self.port = int(self._server.sockets[0].getsockname()[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def _restore_physical_control(self, *, reason: str) -> dict[str, Any]:
        if self.physical_control is None:
            return {"ok": True, "reason": reason, "disabled": True}
        restore = await self.physical_control.restore()
        verify = await self.physical_control.verify_restored()
        self._workload_enabled = True
        result = {
            "ok": bool(restore.get("ok", False)) and bool(verify.get("ok", False)),
            "reason": reason,
            "restore": restore,
            "verify": verify,
            "workload_enabled": True,
            "report": self.physical_control.report(),
        }
        return result

    async def _lease_expired(self, lease_s: float) -> None:
        lease_task = asyncio.current_task()
        try:
            await asyncio.sleep(lease_s)
            self._control_auto_restore = await self._restore_physical_control(
                reason="lease_expired"
            )
        except asyncio.CancelledError:
            raise
        finally:
            # A renewal cancels the old timer and installs its replacement
            # before the cancelled task gets a chance to run ``finally``.
            # Only the timer that is still current may clear the shared lease
            # state; otherwise the replacement becomes untracked and can no
            # longer be renewed or cancelled safely.
            if self._control_lease_task is lease_task:
                self._control_lease_deadline = None
                self._control_lease_task = None

    def _arm_control_lease(self, lease_s: float) -> dict[str, Any]:
        if self.physical_control is None:
            raise PermissionError("physical control is disabled on this Agent")
        if not 5 <= lease_s <= 3600:
            raise ValueError("physical-control lease_s must be in [5, 3600]")
        if self._control_lease_task is not None:
            self._control_lease_task.cancel()
        self._control_lease_deadline = monotonic() + lease_s
        self._control_lease_task = asyncio.create_task(self._lease_expired(lease_s))
        return {"lease_s": lease_s, "deadline_monotonic": self._control_lease_deadline}

    async def close(self) -> None:
        if self._control_lease_task is not None:
            self._control_lease_task.cancel()
            await asyncio.gather(self._control_lease_task, return_exceptions=True)
            self._control_lease_task = None
        if self.physical_control is not None:
            await self._restore_physical_control(reason="agent_close")
        for task in self._executions.values():
            task.cancel()
        if self._executions:
            await asyncio.gather(*self._executions.values(), return_exceptions=True)
            self._executions.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._incoming_transfers.clear()

    def _authenticate(self, request: dict[str, Any]) -> None:
        # Direct artifact chunks use a short-lived, path-bound upload ticket
        # issued by an already authenticated controller request. The ticket is
        # validated again in dispatch; it intentionally replaces the long-lived
        # target agent token for this one transfer.
        if request.get("method") == "artifact.transfer.put_chunk":
            ticket = request.get("payload", {}).get("ticket")
            if isinstance(ticket, str) and ticket:
                return
        if self.token is None:
            return
        supplied = request.get("token")
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.token):
            raise PermissionError("agent authentication failed")

    async def _execute_payload(self, payload: dict[str, Any]):
        if not self._workload_enabled:
            raise RuntimeError("agent workload plane is disabled by physical control")
        component = component_spec_from_dict(payload["component"])
        workspace_id = payload.get("workspace_id")
        if component.image:
            if workspace_id is None:
                execution = await self.docker.execute(component)
            else:
                execution = await self.docker.execute_in_workspace(
                    component,
                    str(workspace_id),
                )
        elif workspace_id is None:
            execution = await self.local.execute(component)
        else:
            execution = await self.local.execute_in_workspace(
                component,
                str(workspace_id),
            )
        return to_primitive(execution)


    def _purge_expired_transfers(self) -> None:
        now = monotonic()
        expired = [
            ticket
            for ticket, transfer in self._incoming_transfers.items()
            if float(transfer["expires_at"]) <= now
        ]
        for ticket in expired:
            self._incoming_transfers.pop(ticket, None)

    def _prepare_incoming_transfer(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._purge_expired_transfers()
        size_bytes = int(payload["size_bytes"])
        expires_s = float(payload.get("expires_s", 60.0))
        if size_bytes < 0:
            raise ValueError("transfer size_bytes cannot be negative")
        if not 0 < expires_s <= 300:
            raise ValueError("transfer expires_s must be in (0, 300]")
        ticket = str(payload.get("ticket") or uuid4())
        existing = self._incoming_transfers.get(ticket)
        if existing is not None:
            if (
                existing["workspace_id"] != str(payload["workspace_id"])
                or existing["path"] != str(payload["path"])
                or int(existing["size_bytes"]) != size_bytes
            ):
                raise ValueError("transfer ticket was reused with different parameters")
            return {"ticket": ticket, "size_bytes": size_bytes, "expires_s": expires_s}
        self._incoming_transfers[ticket] = {
            "workspace_id": str(payload["workspace_id"]),
            "path": str(payload["path"]),
            "size_bytes": size_bytes,
            "expires_at": monotonic() + expires_s,
        }
        return {"ticket": ticket, "size_bytes": size_bytes, "expires_s": expires_s}

    async def _put_transfer_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._purge_expired_transfers()
        ticket = str(payload.get("ticket", ""))
        transfer = self._incoming_transfers.get(ticket)
        if transfer is None:
            raise PermissionError("invalid or expired artifact transfer ticket")
        workspace_id = str(payload["workspace_id"])
        path = str(payload["path"])
        if workspace_id != transfer["workspace_id"] or path != transfer["path"]:
            raise PermissionError("artifact transfer ticket does not match target path")
        offset = int(payload.get("offset", 0))
        data = base64.b64decode(payload.get("data", ""))
        if len(data) > MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact transfer chunk exceeds the 4194304-byte protocol limit")
        expected = int(transfer["size_bytes"])
        if offset < 0 or offset + len(data) > expected:
            raise ValueError("artifact transfer chunk exceeds declared size")
        final = bool(payload.get("final", False))
        if final and offset + len(data) != expected:
            raise ValueError("final artifact transfer chunk does not complete declared size")
        size = await self.local.write_chunk(
            workspace_id,
            path,
            offset=offset,
            data=data,
            truncate=bool(payload.get("truncate", False)),
        )
        if final:
            self._incoming_transfers.pop(ticket, None)
        return {"size": size, "completed": final}

    async def _forward_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_artifact_forward:
            raise PermissionError(
                "direct artifact forwarding is disabled; start the source agent "
                "with --enable-artifact-forward"
            )
        chunk_size = int(payload.get("chunk_size", 256 * 1024))
        if not 1 <= chunk_size <= 4 * 1024 * 1024:
            raise ValueError("artifact forwarding chunk_size must be in [1, 4194304]")
        source_workspace = str(payload["source_workspace"])
        source_path = str(payload["source_path"])
        size = await self.local.stat_file(source_workspace, source_path)

        from .transport import AgentClient, client_tls_context_from_pem

        ca_pem = payload.get("target_ca_pem")
        context = None
        if ca_pem is not None:
            context = client_tls_context_from_pem(str(ca_pem))
        target = AgentClient(
            str(payload["target_host"]),
            int(payload["target_port"]),
            ssl_context=context,
            server_hostname=(
                str(payload.get("target_server_hostname") or payload["target_host"])
                if context is not None
                else None
            ),
        )
        target_workspace = str(payload["target_workspace"])
        target_path = str(payload["target_path"])
        ticket = str(payload["ticket"])
        transfer_started = perf_counter()
        if size == 0:
            await target.write_transfer_chunk(
                ticket=ticket,
                workspace_id=target_workspace,
                path=target_path,
                offset=0,
                data=b"",
                truncate=True,
                final=True,
            )
            return {
                "size_bytes": 0,
                "duration_s": max(0.0, perf_counter() - transfer_started),
            }
        offset = 0
        while offset < size:
            data, reported_size = await self.local.read_chunk(
                source_workspace,
                source_path,
                offset=offset,
                limit=min(chunk_size, size - offset),
            )
            if reported_size != size:
                raise OSError("artifact changed size during direct transfer")
            if not data:
                raise OSError("empty artifact chunk before direct-transfer EOF")
            final = offset + len(data) == size
            await target.write_transfer_chunk(
                ticket=ticket,
                workspace_id=target_workspace,
                path=target_path,
                offset=offset,
                data=data,
                truncate=offset == 0,
                final=final,
            )
            offset += len(data)
        return {
            "size_bytes": size,
            "duration_s": max(0.0, perf_counter() - transfer_started),
        }

    async def _network_agent_probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_network_probe:
            raise PermissionError(
                "agent-to-agent network probing is disabled; start the source agent "
                "with --enable-network-probe"
            )

        host = str(payload["host"])
        port = int(payload.get("port", 8765))
        samples = int(payload.get("samples", 3))
        payload_bytes = int(payload.get("payload_bytes", 256 * 1024))
        timeout = float(payload.get("timeout", 5.0))
        if not host:
            raise ValueError("network probe host cannot be empty")
        if not 1 <= port <= 65535:
            raise ValueError("network probe port must be in [1, 65535]")
        if not 1 <= samples <= 20:
            raise ValueError("network probe samples must be in [1, 20]")
        if not 0 <= payload_bytes <= 4 * 1024 * 1024:
            raise ValueError("network probe payload_bytes must be in [0, 4194304]")
        if not 0 < timeout <= 30:
            raise ValueError("network probe timeout must be in (0, 30]")

        from .transport import AgentClient, client_tls_context_from_pem

        ca_pem = payload.get("target_ca_pem")
        context = None
        if ca_pem is not None:
            context = client_tls_context_from_pem(str(ca_pem))
        target = AgentClient(
            host,
            port,
            timeout=timeout,
            token=(
                None
                if payload.get("target_token") is None
                else str(payload["target_token"])
            ),
            ssl_context=context,
            server_hostname=(
                str(payload.get("target_server_hostname") or host)
                if context is not None
                else None
            ),
        )

        rtt_ms: list[float] = []
        for _ in range(samples):
            started = perf_counter()
            await target.ping()
            rtt_ms.append((perf_counter() - started) * 1000.0)

        throughput_mbps: list[float] = []
        if payload_bytes > 0:
            probe_data = "x" * payload_bytes
            for _ in range(samples):
                started = perf_counter()
                result = await target.request("probe.echo", {"data": probe_data})
                elapsed = max(perf_counter() - started, 1e-9)
                if int(result.get("received_bytes", -1)) != payload_bytes:
                    raise OSError("target agent reported an incomplete probe payload")
                throughput_mbps.append(payload_bytes * 8 / elapsed / 1_000_000)

        return {
            "source_node_id": self.node_id,
            "target_host": host,
            "target_port": port,
            "samples": samples,
            "payload_bytes": payload_bytes,
            "rtt_ms": statistics.median(rtt_ms),
            "rtt_std_ms": statistics.pstdev(rtt_ms) if len(rtt_ms) > 1 else 0.0,
            "rtt_samples_ms": rtt_ms,
            "bandwidth_mbps": (
                statistics.median(throughput_mbps) if throughput_mbps else None
            ),
            "bandwidth_samples_mbps": throughput_mbps,
        }

    def _telemetry(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.workspace_root)
        try:
            load_1m, load_5m, load_15m = os.getloadavg()
        except (AttributeError, OSError):
            load_1m = load_5m = load_15m = 0.0
        return {
            "node_id": self.node_id,
            "cpu_count": os.cpu_count(),
            "cpu_capacity": detect_cpu_capacity(),
            "memory_capacity_bytes": detect_memory_capacity_bytes(),
            "cgroup_version": detect_cgroup_version(),
            "active_executions": sum(
                1 for task in self._executions.values() if not task.done()
            ),
            "load_1m": load_1m,
            "load_5m": load_5m,
            "load_15m": load_15m,
            "workspace_total_bytes": usage.total,
            "workspace_free_bytes": usage.free,
        }

    async def _dispatch(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "ping":
            return {
                "node_id": self.node_id,
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "darpan_version": __version__,
                "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                "environment": environment_snapshot(),
                "wall_time_s": time(),
                "monotonic_s": monotonic(),
                "features": {
                    "network_probe": self.allow_network_probe,
                    "artifact_forward": self.allow_artifact_forward,
                    "tls": self.ssl_context is not None,
                    "physical_control": self.physical_control is not None,
                    "docker": shutil.which("docker") is not None,
                },
            }
        if method == "telemetry":
            return self._telemetry()
        if method == "control.inspect":
            if self.physical_control is None:
                return {
                    "enabled": False,
                    "workload_enabled": self._workload_enabled,
                }
            capabilities = self.physical_control.capabilities()
            readiness_fn = getattr(self.physical_control, "readiness", None)
            readiness = {} if readiness_fn is None else dict(readiness_fn())
            return {
                "enabled": True,
                "workload_enabled": self._workload_enabled,
                "lease_active": self._control_lease_task is not None,
                "lease_deadline_monotonic": self._control_lease_deadline,
                "last_auto_restore": self._control_auto_restore,
                "netem": bool(capabilities.netem),
                "route": bool(capabilities.route),
                "cpu_capacity": bool(capabilities.cpu_capacity),
                "interfaces": list(capabilities.interfaces),
                "cpu_max_path": capabilities.cpu_max_path,
                "readiness": readiness,
            }
        if method == "control.lease":
            return self._arm_control_lease(float(payload.get("lease_s", 60.0)))
        if method == "control.workload":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            self._arm_control_lease(float(payload.get("lease_s", 60.0)))
            enabled = bool(payload["enabled"])
            if not enabled:
                for task in tuple(self._executions.values()):
                    if not task.done():
                        task.cancel()
                if self._executions:
                    await asyncio.gather(
                        *tuple(self._executions.values()), return_exceptions=True
                    )
                self._executions.clear()
            self._workload_enabled = enabled
            return {
                "enabled": enabled,
                "mode": "agent-workload-plane-unavailable",
            }
        if method == "control.netem":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            self._arm_control_lease(float(payload.get("lease_s", 60.0)))
            return await self.physical_control.apply_netem(
                str(payload["interface"]),
                latency_ms=(
                    None if payload.get("latency_ms") is None else float(payload["latency_ms"])
                ),
                bandwidth_mbps=(
                    None
                    if payload.get("bandwidth_mbps") is None
                    else float(payload["bandwidth_mbps"])
                ),
                loss_pct=(
                    None if payload.get("loss_pct") is None else float(payload["loss_pct"])
                ),
            )
        if method == "control.cpu_capacity":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            self._arm_control_lease(float(payload.get("lease_s", 60.0)))
            return await self.physical_control.set_cpu_capacity(float(payload["cpus"]))
        if method == "control.route.bind":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            self._arm_control_lease(float(payload.get("lease_s", 60.0)))
            return await self.physical_control.bind_route(
                destination=str(payload["destination"]),
                via=str(payload["via"]),
                interface=str(payload["interface"]),
            )
        if method == "control.route.clear":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            self._arm_control_lease(float(payload.get("lease_s", 60.0)))
            return await self.physical_control.clear_route(
                destination=str(payload["destination"])
            )
        if method == "control.restore":
            if self.physical_control is None:
                raise PermissionError("physical control is disabled on this Agent")
            if self._control_lease_task is not None:
                self._control_lease_task.cancel()
                await asyncio.gather(self._control_lease_task, return_exceptions=True)
                self._control_lease_task = None
                self._control_lease_deadline = None
            return await self._restore_physical_control(reason="controller_restore")
        if method == "probe.echo":
            data = payload.get("data", "")
            if not isinstance(data, str):
                raise TypeError("probe.echo data must be a string")
            return {"received_bytes": len(data.encode())}
        if method == "network.agent_probe":
            return await self._network_agent_probe(payload)
        if method == "execute":
            return await self._execute_payload(payload)
        if method == "execute.start":
            if not self._workload_enabled:
                raise RuntimeError("agent workload plane is disabled by physical control")
            execution_id = str(payload.get("execution_id") or uuid4())
            if execution_id not in self._executions:
                self._executions[execution_id] = asyncio.create_task(
                    self._execute_payload(payload)
                )
            return {"execution_id": execution_id}
        if method == "execute.wait":
            execution_id = str(payload["execution_id"])
            task = self._executions.get(execution_id)
            if task is None:
                raise KeyError(f"unknown execution: {execution_id}")
            try:
                result = await task
                return {"execution_id": execution_id, "result": result, "cancelled": False}
            except asyncio.CancelledError:
                return {"execution_id": execution_id, "cancelled": True}
        if method == "execute.release":
            execution_id = str(payload["execution_id"])
            task = self._executions.get(execution_id)
            if task is None:
                return {"execution_id": execution_id, "released": False}
            if not task.done():
                raise RuntimeError(f"cannot release active execution: {execution_id}")
            self._executions.pop(execution_id, None)
            return {"execution_id": execution_id, "released": True}
        if method == "execute.cancel":
            execution_id = str(payload["execution_id"])
            task = self._executions.get(execution_id)
            if task is None or task.done():
                return {"execution_id": execution_id, "cancelled": False}
            task.cancel()
            await asyncio.sleep(0)
            return {"execution_id": execution_id, "cancelled": True}
        if method == "artifact.transfer.prepare":
            return self._prepare_incoming_transfer(payload)
        if method == "artifact.transfer.put_chunk":
            return await self._put_transfer_chunk(payload)
        if method == "artifact.forward":
            return await self._forward_artifact(payload)
        if method == "artifact.put_chunk":
            data = base64.b64decode(payload.get("data", ""))
            if len(data) > MAX_AGENT_BINARY_PAYLOAD_BYTES:
                raise ValueError("artifact chunk exceeds the 4194304-byte protocol limit")
            size = await self.local.write_chunk(
                str(payload["workspace_id"]),
                str(payload["path"]),
                offset=int(payload.get("offset", 0)),
                data=data,
                truncate=bool(payload.get("truncate", False)),
            )
            return {"size": size}
        if method == "artifact.get_chunk":
            limit = int(payload.get("limit", 256 * 1024))
            if not 0 <= limit <= MAX_AGENT_BINARY_PAYLOAD_BYTES:
                raise ValueError("artifact read limit must be in [0, 4194304]")
            data, size = await self.local.read_chunk(
                str(payload["workspace_id"]),
                str(payload["path"]),
                offset=int(payload.get("offset", 0)),
                limit=limit,
            )
            return {"data": base64.b64encode(data).decode("ascii"), "size": size}
        if method == "artifact.stat":
            size = await self.local.stat_file(
                str(payload["workspace_id"]),
                str(payload["path"]),
            )
            return {"size": size}
        if method == "artifact.delete":
            deleted = await self.local.delete_file(
                str(payload["workspace_id"]),
                str(payload["path"]),
            )
            return {"deleted": deleted}
        raise ValueError(f"unknown agent method: {method}")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            request = json.loads(raw)
            self._authenticate(request)
            result = await self._dispatch(
                str(request.get("method")),
                dict(request.get("payload", {})),
            )
            response: dict[str, Any] = {
                "id": request.get("id"),
                "ok": True,
                "result": result,
            }
        except Exception as exc:  # agent boundary must return structured failures
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
