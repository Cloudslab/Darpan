from __future__ import annotations

import asyncio
import subprocess

from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.executors.remote import RemoteExecutor
from darpan.runtime.real.transport import (
    AgentClient,
    client_tls_context,
    client_tls_context_from_pem,
    server_tls_context,
)


def test_agent_transport_supports_verified_tls(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )

    async def run():
        server = AgentServer(
            "tls-node",
            host="127.0.0.1",
            port=0,
            ssl_context=server_tls_context(str(cert), str(key)),
        )
        await server.start()
        client = AgentClient(
            "127.0.0.1",
            server.port,
            ssl_context=client_tls_context(str(cert)),
            server_hostname="localhost",
        )
        assert (await client.ping())["node_id"] == "tls-node"
        await server.close()

    asyncio.run(run())


def test_direct_agent_artifact_forwarding_preserves_tls_verification(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    ca_pem = cert.read_text(encoding="utf-8")

    async def run():
        source_server = AgentServer(
            "source",
            host="127.0.0.1",
            port=0,
            ssl_context=server_tls_context(str(cert), str(key)),
            allow_artifact_forward=True,
        )
        target_server = AgentServer(
            "target",
            host="127.0.0.1",
            port=0,
            ssl_context=server_tls_context(str(cert), str(key)),
        )
        await source_server.start()
        await target_server.start()
        source = RemoteExecutor(
            AgentClient(
                "127.0.0.1",
                source_server.port,
                ssl_context=client_tls_context_from_pem(ca_pem),
                server_hostname="localhost",
                tls_ca_pem=ca_pem,
            ),
            direct_artifact_forward=True,
        )
        target = RemoteExecutor(
            AgentClient(
                "127.0.0.1",
                target_server.port,
                ssl_context=client_tls_context_from_pem(ca_pem),
                server_hostname="localhost",
                tls_ca_pem=ca_pem,
            )
        )
        try:
            await source.put_file("source-ws", "data.bin", b"tls-direct")
            size, duration_s = await source.copy_file_to(
                "source-ws",
                "data.bin",
                target,
                "target-ws",
                "received.bin",
                chunk_size=4,
            )
            assert size == len(b"tls-direct")
            assert duration_s > 0
            assert await target.get_file("target-ws", "received.bin") == b"tls-direct"
        finally:
            await source_server.close()
            await target_server.close()

    asyncio.run(run())
