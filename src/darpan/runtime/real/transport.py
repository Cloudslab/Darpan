"""Newline-delimited JSON transport for Darpan node agents.

Large artifacts are transferred in bounded base64 chunks rather than embedded
in one unbounded request/response.  Optional shared-token authentication keeps
an accidentally exposed research agent from accepting anonymous control
requests; TLS can be layered on this transport without changing the protocol.
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

# asyncio StreamReader defaults to 64 KiB, but Darpan deliberately allows bounded
# artifact chunks up to 4 MiB. Base64 expands those chunks by roughly 4/3, so
# the newline-delimited JSON transport needs an explicit bounded frame limit.
MAX_AGENT_BINARY_PAYLOAD_BYTES = 4 * 1024 * 1024
AGENT_STREAM_LIMIT = 8 * 1024 * 1024
# ``artifact.forward`` is a control request whose response is intentionally
# delayed until the source Agent has streamed the complete artifact to the
# target Agent.  It therefore needs a much longer deadline than ordinary
# metadata/control RPCs, especially when a physical experiment applies link
# shaping to the same interface used by the Agent data plane.
ARTIFACT_FORWARD_TIMEOUT_S = 300.0
# Agent execution-control acknowledgements are idempotent and should return
# immediately after registering the request.  A short retry window prevents a
# lost acknowledgement on a WAN-connected cluster from stalling an otherwise
# completed scheduling window for five seconds at a time.
EXECUTION_CONTROL_ACK_TIMEOUT_S = 1.0
# ``execute.wait`` is a long-poll.  Reissuing the idempotent wait periodically
# keeps a delayed WAN response from hiding an already-completed process for up
# to the Agent client's general 30-second deadline.
EXECUTION_WAIT_POLL_TIMEOUT_S = 2.0


class AgentClient:
    def __init__(
        self,
        host: str,
        port: int = 8765,
        *,
        timeout: float = 30.0,
        artifact_chunk_size: int = 256 * 1024,
        token: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        server_hostname: str | None = None,
        tls_ca_pem: str | None = None,
    ) -> None:
        if not 1 <= artifact_chunk_size <= MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact_chunk_size must be in [1, 4194304]")
        self.host = host
        self.port = port
        self.timeout = timeout
        self.artifact_chunk_size = artifact_chunk_size
        self.token = token
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self.tls_ca_pem = tls_ca_pem

    async def _request_once(
        self,
        method: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        raw = b""
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    ssl=self.ssl_context,
                    server_hostname=(
                        self.server_hostname if self.ssl_context is not None else None
                    ),
                    limit=AGENT_STREAM_LIMIT,
                ),
                timeout,
            )
            request = {
                "id": str(uuid4()),
                "method": method,
                "payload": dict(payload or {}),
            }
            if self.token is not None:
                request["token"] = self.token
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout)
        finally:
            if writer is not None:
                writer.close()
                try:
                    await asyncio.wait_for(
                        writer.wait_closed(),
                        timeout=min(timeout, 1.0),
                    )
                except (OSError, TimeoutError):
                    # Closing a completed TCP stream is best-effort cleanup. On
                    # Windows, wait_closed() can raise WinError 121 after the
                    # complete response has already been read. Never discard a
                    # valid Agent response because teardown itself failed.
                    pass
        if not raw:
            raise ConnectionError(f"agent {self.host}:{self.port} closed connection")
        response = json.loads(raw)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "agent request failed"))
        return dict(response.get("result", {}))

    async def request(
        self,
        method: str,
        payload: Mapping[str, Any] | None = None,
        *,
        attempts: int = 1,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if attempts <= 0:
            raise ValueError("agent request attempts must be positive")
        request_timeout = self.timeout if timeout is None else float(timeout)
        if request_timeout <= 0:
            raise ValueError("agent request timeout must be positive")
        for attempt in range(attempts):
            try:
                return await self._request_once(
                    method,
                    payload,
                    timeout=request_timeout,
                )
            except OSError as exc:
                if attempt + 1 == attempts:
                    raise OSError(
                        exc.errno,
                        f"agent RPC {method} failed after {attempts} attempt(s): {exc}",
                    ) from exc
                await asyncio.sleep(0.2 * (attempt + 1))
        raise AssertionError("unreachable Agent request retry state")

    def _short_timeout(self) -> float:
        return min(self.timeout, 5.0)

    def _execution_control_ack_timeout(self) -> float:
        return min(self._short_timeout(), EXECUTION_CONTROL_ACK_TIMEOUT_S)

    async def ping(self) -> dict[str, Any]:
        return await self.request("ping", attempts=3, timeout=self._short_timeout())

    async def telemetry(self) -> dict[str, Any]:
        return await self.request("telemetry", attempts=3, timeout=self._short_timeout())

    async def control_inspect(self) -> dict[str, Any]:
        return await self.request("control.inspect", attempts=3, timeout=self._short_timeout())

    async def control_lease(self, *, lease_s: float = 60.0) -> dict[str, Any]:
        return await self.request(
            "control.lease",
            {"lease_s": lease_s},
            attempts=3,
            timeout=self._short_timeout(),
        )

    async def control_workload(self, *, enabled: bool, lease_s: float = 60.0) -> dict[str, Any]:
        return await self.request(
            "control.workload",
            {"enabled": enabled, "lease_s": lease_s},
            attempts=3,
        )

    async def control_netem(
        self,
        *,
        interface: str,
        latency_ms: float | None = None,
        bandwidth_mbps: float | None = None,
        loss_pct: float | None = None,
        lease_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.request(
            "control.netem",
            {
                "interface": interface,
                "latency_ms": latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
                "loss_pct": loss_pct,
                "lease_s": lease_s,
            },
            attempts=3,
        )

    async def control_cpu_capacity(self, cpus: float, *, lease_s: float = 60.0) -> dict[str, Any]:
        return await self.request(
            "control.cpu_capacity",
            {"cpus": cpus, "lease_s": lease_s},
            attempts=3,
        )

    async def control_route_bind(
        self,
        *,
        destination: str,
        via: str,
        interface: str,
        lease_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.request(
            "control.route.bind",
            {
                "destination": destination,
                "via": via,
                "interface": interface,
                "lease_s": lease_s,
            },
            attempts=3,
        )

    async def control_route_clear(
        self, *, destination: str, lease_s: float = 60.0
    ) -> dict[str, Any]:
        return await self.request(
            "control.route.clear",
            {"destination": destination, "lease_s": lease_s},
            attempts=3,
        )

    async def control_restore(self) -> dict[str, Any]:
        return await self.request("control.restore", attempts=3)

    async def start_execution(
        self,
        component: Mapping[str, Any],
        *,
        workspace_id: str | None = None,
    ) -> str:
        execution_id = str(uuid4())
        payload: dict[str, Any] = {
            "component": dict(component),
            "execution_id": execution_id,
        }
        if workspace_id is not None:
            payload["workspace_id"] = workspace_id
        result = await self.request(
            "execute.start",
            payload,
            attempts=3,
            timeout=self._execution_control_ack_timeout(),
        )
        returned_id = str(result["execution_id"])
        if returned_id != execution_id:
            raise RuntimeError(
                f"Agent returned execution id {returned_id!r}, expected {execution_id!r}"
            )
        return returned_id

    async def wait_execution(self, execution_id: str) -> dict[str, Any]:
        return await self.request(
            "execute.wait",
            {"execution_id": execution_id},
            timeout=min(self.timeout, EXECUTION_WAIT_POLL_TIMEOUT_S),
        )

    async def release_execution(self, execution_id: str) -> bool:
        result = await self.request(
            "execute.release",
            {"execution_id": execution_id},
            attempts=3,
            timeout=self._execution_control_ack_timeout(),
        )
        return bool(result.get("released", False))

    async def cancel_execution(self, execution_id: str) -> bool:
        result = await self.request(
            "execute.cancel",
            {"execution_id": execution_id},
            attempts=3,
            timeout=self._execution_control_ack_timeout(),
        )
        return bool(result.get("cancelled", False))

    async def write_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        data: bytes,
        truncate: bool = False,
    ) -> int:
        if len(data) > MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact chunk exceeds the 4194304-byte protocol limit")
        result = await self.request(
            "artifact.put_chunk",
            {
                "workspace_id": workspace_id,
                "path": path,
                "offset": offset,
                "truncate": truncate,
                "data": base64.b64encode(data).decode("ascii"),
            },
            attempts=3,
            timeout=self._short_timeout(),
        )
        return int(result["size"])

    async def read_chunk(
        self,
        workspace_id: str,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> tuple[bytes, int]:
        if not 0 <= limit <= MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact read limit must be in [0, 4194304]")
        result = await self.request(
            "artifact.get_chunk",
            {
                "workspace_id": workspace_id,
                "path": path,
                "offset": offset,
                "limit": limit,
            },
            attempts=3,
            timeout=self._short_timeout(),
        )
        return base64.b64decode(result.get("data", "")), int(result["size"])

    async def put_file(self, workspace_id: str, path: str, data: bytes) -> None:
        if not data:
            await self.write_chunk(
                workspace_id,
                path,
                offset=0,
                data=b"",
                truncate=True,
            )
            return
        for offset in range(0, len(data), self.artifact_chunk_size):
            chunk = data[offset : offset + self.artifact_chunk_size]
            await self.write_chunk(
                workspace_id,
                path,
                offset=offset,
                data=chunk,
                truncate=offset == 0,
            )

    async def get_file(self, workspace_id: str, path: str) -> bytes:
        offset = 0
        chunks: list[bytes] = []
        while True:
            chunk, size = await self.read_chunk(
                workspace_id,
                path,
                offset=offset,
                limit=self.artifact_chunk_size,
            )
            chunks.append(chunk)
            offset += len(chunk)
            if offset >= size:
                break
            if not chunk:
                raise OSError("agent returned an empty artifact chunk before EOF")
        return b"".join(chunks)

    async def stat_file(self, workspace_id: str, path: str) -> int:
        result = await self.request(
            "artifact.stat",
            {"workspace_id": workspace_id, "path": path},
            attempts=3,
            timeout=self._short_timeout(),
        )
        return int(result["size"])

    async def delete_file(self, workspace_id: str, path: str) -> bool:
        result = await self.request(
            "artifact.delete",
            {"workspace_id": workspace_id, "path": path},
            attempts=3,
            timeout=self._short_timeout(),
        )
        return bool(result.get("deleted", False))

    async def prepare_incoming_transfer(
        self,
        workspace_id: str,
        path: str,
        *,
        size_bytes: int,
        expires_s: float = 60.0,
    ) -> str:
        ticket = str(uuid4())
        result = await self.request(
            "artifact.transfer.prepare",
            {
                "ticket": ticket,
                "workspace_id": workspace_id,
                "path": path,
                "size_bytes": int(size_bytes),
                "expires_s": float(expires_s),
            },
            attempts=3,
            timeout=self._short_timeout(),
        )
        returned_ticket = str(result["ticket"])
        if returned_ticket != ticket:
            raise RuntimeError(
                f"Agent returned transfer ticket {returned_ticket!r}, expected {ticket!r}"
            )
        return returned_ticket

    async def write_transfer_chunk(
        self,
        *,
        ticket: str,
        workspace_id: str,
        path: str,
        offset: int,
        data: bytes,
        truncate: bool = False,
        final: bool = False,
    ) -> int:
        if len(data) > MAX_AGENT_BINARY_PAYLOAD_BYTES:
            raise ValueError("artifact transfer chunk exceeds the 4194304-byte protocol limit")
        result = await self.request(
            "artifact.transfer.put_chunk",
            {
                "ticket": ticket,
                "workspace_id": workspace_id,
                "path": path,
                "offset": int(offset),
                "truncate": bool(truncate),
                "final": bool(final),
                "data": base64.b64encode(data).decode("ascii"),
            },
        )
        return int(result["size"])

    async def forward_artifact(
        self,
        source_workspace: str,
        source_path: str,
        *,
        target_host: str,
        target_port: int,
        target_workspace: str,
        target_path: str,
        ticket: str,
        target_ca_pem: str | None = None,
        target_server_hostname: str | None = None,
        chunk_size: int | None = None,
    ) -> tuple[int, float]:
        payload: dict[str, Any] = {
            "source_workspace": source_workspace,
            "source_path": source_path,
            "target_host": target_host,
            "target_port": int(target_port),
            "target_workspace": target_workspace,
            "target_path": target_path,
            "ticket": ticket,
            "chunk_size": int(chunk_size or self.artifact_chunk_size),
        }
        if target_ca_pem is not None:
            payload["target_ca_pem"] = target_ca_pem
        if target_server_hostname is not None:
            payload["target_server_hostname"] = target_server_hostname
        result = await self.request(
            "artifact.forward",
            payload,
            timeout=max(self.timeout, ARTIFACT_FORWARD_TIMEOUT_S),
        )
        return int(result["size_bytes"]), float(result["duration_s"])

    def direct_transfer_endpoint(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "ca_pem": self.tls_ca_pem,
            "server_hostname": self.server_hostname,
        }

    async def probe_agent(
        self,
        host: str,
        port: int,
        *,
        token: str | None = None,
        ca_pem: str | None = None,
        server_hostname: str | None = None,
        samples: int = 3,
        payload_bytes: int = 256 * 1024,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Ask this agent to measure its path to another Darpan agent.

        The measurement originates on the remote node rather than on the
        controller, so the resulting RTT/throughput describes the physical
        source->target path that a Continuum link is meant to represent.
        """

        payload: dict[str, Any] = {
            "host": host,
            "port": int(port),
            "samples": int(samples),
            "payload_bytes": int(payload_bytes),
            "timeout": float(timeout),
        }
        if token is not None:
            payload["target_token"] = token
        if ca_pem is not None:
            payload["target_ca_pem"] = ca_pem
        if server_hostname is not None:
            payload["target_server_hostname"] = server_hostname
        return await self.request("network.agent_probe", payload)


def client_tls_context(ca_file: str) -> ssl.SSLContext:
    """Create a certificate-verifying client context for Darpan agents."""

    return ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)


def client_tls_context_from_pem(ca_pem: str) -> ssl.SSLContext:
    """Create a client TLS context from PEM certificate text."""

    return ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cadata=ca_pem)


def server_tls_context(cert_file: str, key_file: str) -> ssl.SSLContext:
    """Create a TLS server context for a Darpan agent."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return context
