from __future__ import annotations

import asyncio

from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.transport import AgentClient


def test_agent_ping_and_execution():
    async def run():
        server = AgentServer("test-node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port)
        ping = await client.ping()
        assert ping["node_id"] == "test-node"
        result = await client.request(
            "execute",
            {
                "component": {
                    "id": "hello",
                    "command": ["python", "-c", "print('hello')"],
                }
            },
        )
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]
        await server.close()

    asyncio.run(run())
