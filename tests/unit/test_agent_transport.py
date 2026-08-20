from __future__ import annotations

import asyncio
import json

from darpan.runtime.real.transport import ARTIFACT_FORWARD_TIMEOUT_S, AgentClient


def test_agent_response_survives_windows_wait_closed_error(monkeypatch):
    class Reader:
        async def readline(self):
            return (
                json.dumps(
                    {
                        "ok": True,
                        "result": {"node_id": "edge"},
                    }
                )
                + "\n"
            ).encode()

    class Writer:
        def __init__(self):
            self.closed = False
            self.written = b""

        def write(self, data):
            self.written += data

        async def drain(self):
            return None

        def close(self):
            self.closed = True

        async def wait_closed(self):
            raise OSError(121, "The semaphore timeout period has expired")

    writer = Writer()

    async def open_connection(*args, **kwargs):
        del args, kwargs
        return Reader(), writer

    monkeypatch.setattr(asyncio, "open_connection", open_connection)

    async def run():
        result = await AgentClient("agent.example", timeout=0.1).ping()
        assert result == {"node_id": "edge"}
        assert writer.closed is True
        request = json.loads(writer.written)
        assert request["method"] == "ping"

    asyncio.run(run())


def test_idempotent_agent_request_retries_a_lost_response(monkeypatch):
    class Reader:
        def __init__(self, attempt):
            self.attempt = attempt

        async def readline(self):
            if self.attempt == 1:
                raise OSError(121, "simulated stalled socket")
            return b'{"ok": true, "result": {"size": 17}}\n'

    class Writer:
        def write(self, data):
            self.written = data

        async def drain(self):
            return None

        def close(self):
            return None

        async def wait_closed(self):
            return None

    attempts = 0

    async def open_connection(*args, **kwargs):
        nonlocal attempts
        del args, kwargs
        attempts += 1
        return Reader(attempts), Writer()

    monkeypatch.setattr(asyncio, "open_connection", open_connection)

    async def run():
        result = await AgentClient("agent.example", timeout=0.1).request(
            "artifact.stat",
            {"workspace_id": "workspace", "path": "artifact.bin"},
            attempts=2,
            timeout=0.1,
        )
        assert result == {"size": 17}
        assert attempts == 2

    asyncio.run(run())


def test_artifact_forward_uses_long_data_plane_timeout(monkeypatch):
    client = AgentClient("source.example", timeout=0.1)
    observed = {}

    async def request(method, payload, **kwargs):
        observed.update(method=method, payload=payload, kwargs=kwargs)
        return {"size_bytes": 1024, "duration_s": 12.5}

    monkeypatch.setattr(client, "request", request)

    async def run():
        result = await client.forward_artifact(
            "source-workspace",
            "source.bin",
            target_host="target.example",
            target_port=8765,
            target_workspace="target-workspace",
            target_path="target.bin",
            ticket="ticket",
        )
        assert result == (1024, 12.5)
        assert observed["method"] == "artifact.forward"
        assert observed["kwargs"]["timeout"] == ARTIFACT_FORWARD_TIMEOUT_S

    asyncio.run(run())


def test_execution_wait_uses_bounded_long_poll_timeout(monkeypatch):
    client = AgentClient("agent.example", timeout=30.0)
    observed = {}

    async def request(method, payload=None, **kwargs):
        observed.update({"method": method, "payload": payload, **kwargs})
        return {"execution_id": "execution-1", "cancelled": False}

    monkeypatch.setattr(client, "request", request)

    async def run():
        await client.wait_execution("execution-1")
        assert observed["method"] == "execute.wait"
        assert observed["payload"] == {"execution_id": "execution-1"}
        assert observed["timeout"] == 2.0

    asyncio.run(run())


def test_physical_control_requests_retry_transient_transport_errors(monkeypatch):
    client = AgentClient("agent.example", timeout=0.1)
    observed = []

    async def request(method, payload=None, **kwargs):
        observed.append((method, payload, kwargs))
        return {"restore_verified": True}

    monkeypatch.setattr(client, "request", request)

    async def run():
        await client.control_netem(interface="eth1", latency_ms=80.0)
        await client.control_restore()
        assert observed[0][0] == "control.netem"
        assert observed[0][2]["attempts"] == 3
        assert observed[1][0] == "control.restore"
        assert observed[1][2]["attempts"] == 3

    asyncio.run(run())
