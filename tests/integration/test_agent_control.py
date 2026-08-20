from __future__ import annotations

import asyncio
import sys

import pytest

from darpan.core.application import ComponentSpec
from darpan.core.serialization import to_primitive
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.executors.remote import RemoteExecutionHandle, RemoteExecutor
from darpan.runtime.real.transport import AgentClient


def test_agent_optional_authentication_and_telemetry():
    async def run():
        server = AgentServer("secure", host="127.0.0.1", port=0, token="secret")
        await server.start()
        unauthenticated = AgentClient("127.0.0.1", server.port)
        with pytest.raises(RuntimeError, match="authentication failed"):
            await unauthenticated.ping()

        client = AgentClient("127.0.0.1", server.port, token="secret")
        ping = await client.ping()
        telemetry = await client.telemetry()
        assert ping["node_id"] == "secure"
        assert telemetry["node_id"] == "secure"
        assert telemetry["workspace_total_bytes"] > 0
        assert telemetry["workspace_free_bytes"] >= 0
        await server.close()

    asyncio.run(run())


def test_agent_execution_can_be_cancelled():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port, timeout=5)
        component = ComponentSpec(
            "slow",
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
        )
        execution_id = await client.start_execution(to_primitive(component))
        assert await client.cancel_execution(execution_id) is True
        result = await client.wait_execution(execution_id)
        assert result["cancelled"] is True
        await server.close()

    asyncio.run(run())


def test_completed_execution_is_replayable_until_controller_releases_it():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port, timeout=5)
        component = ComponentSpec(
            "quick",
            command=(sys.executable, "-c", "print('durable result')"),
        )
        execution_id = await client.start_execution(to_primitive(component))
        first = await client.wait_execution(execution_id)
        second = await client.wait_execution(execution_id)
        assert first == second
        assert await client.release_execution(execution_id) is True
        assert await client.release_execution(execution_id) is False
        with pytest.raises(RuntimeError, match="unknown execution"):
            await client.wait_execution(execution_id)
        await server.close()

    asyncio.run(run())


def test_replayed_start_with_same_execution_id_does_not_duplicate_work():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port, timeout=5)
        component = ComponentSpec(
            "quick",
            command=(sys.executable, "-c", "print('once')"),
        )
        payload = {
            "component": to_primitive(component),
            "execution_id": "controller-execution-1",
        }
        first = await client.request("execute.start", payload)
        second = await client.request("execute.start", payload)
        assert first == second == {"execution_id": "controller-execution-1"}
        assert len(server._executions) == 1
        result = await client.wait_execution("controller-execution-1")
        assert result["result"]["stdout"].splitlines() == ["once"]
        assert await client.release_execution("controller-execution-1") is True
        await server.close()

    asyncio.run(run())


def test_remote_handle_retries_wait_without_starting_a_second_execution():
    class FlakyClient:
        def __init__(self):
            self.wait_calls = 0
            self.release_calls = 0

        async def wait_execution(self, execution_id):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise OSError(121, "simulated lost completion response")
            return {
                "execution_id": execution_id,
                "cancelled": False,
                "result": {
                    "return_code": 0,
                    "duration_s": 0.1,
                    "stdout": "ok\n",
                    "stderr": "",
                    "output_bytes": 0,
                    "measurements": {},
                },
            }

        async def release_execution(self, execution_id):
            self.release_calls += 1
            return True

    async def run():
        client = FlakyClient()
        handle = RemoteExecutionHandle(client, "execution-1")
        result = await handle.wait()
        assert result.success
        assert client.wait_calls == 2
        assert client.release_calls == 1

    asyncio.run(run())


def test_remote_executor_propagates_task_cancellation_to_agent():
    async def run():
        server = AgentServer("node", host="127.0.0.1", port=0)
        await server.start()
        client = AgentClient("127.0.0.1", server.port, timeout=5)
        executor = RemoteExecutor(client)
        component = ComponentSpec(
            "slow",
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
        )
        task = asyncio.create_task(executor.execute(component))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # A new execution proves the agent remains responsive after cancellation.
        result = await executor.execute(
            ComponentSpec("quick", command=(sys.executable, "-c", "print('ok')"))
        )
        assert result.success
        assert "ok" in result.stdout
        await server.close()

    asyncio.run(run())


def test_agent_physical_control_is_opt_in_and_workload_gate_is_real():
    class Control:
        def capabilities(self):
            class Capabilities:
                netem = True
                route = True
                cpu_capacity = True
                interfaces = ("eth-test",)
                cpu_max_path = "/test/cpu.max"

            return Capabilities()

        async def apply_netem(self, interface, **kwargs):
            return {"interface": interface, **kwargs}

        async def set_cpu_capacity(self, cpus):
            return {"cpus": cpus}

        async def bind_route(self, **kwargs):
            return kwargs

        async def clear_route(self, **kwargs):
            return kwargs

        async def restore(self):
            return {"ok": True, "restored": []}

        async def verify_restored(self):
            return {"ok": True, "mismatches": []}

        def report(self):
            return {"schema": "fake"}

    async def run():
        server = AgentServer(
            "node",
            host="127.0.0.1",
            port=0,
            physical_control=Control(),
        )
        await server.start()
        client = AgentClient("127.0.0.1", server.port, timeout=5)
        try:
            inspect = await client.control_inspect()
            assert inspect["enabled"] is True
            assert inspect["route"] is True
            await client.control_workload(enabled=False)
            component = ComponentSpec("blocked", command=(sys.executable, "-c", "pass"))
            with pytest.raises(RuntimeError, match="workload plane is disabled"):
                await client.start_execution(to_primitive(component))
            await client.control_workload(enabled=True)
            execution_id = await client.start_execution(to_primitive(component))
            result = await client.wait_execution(execution_id)
            assert result["result"]["return_code"] == 0
            restored = await client.control_restore()
            assert restored["ok"] is True
        finally:
            await server.close()

    asyncio.run(run())


def test_agent_control_lease_expiry_auto_restores_workload_plane():
    class Control:
        def __init__(self):
            self.restore_calls = 0

        def capabilities(self):
            class Capabilities:
                netem = False
                route = False
                cpu_capacity = False
                interfaces = ()
                cpu_max_path = None

            return Capabilities()

        async def restore(self):
            self.restore_calls += 1
            return {"ok": True, "restored": ["workload"]}

        async def verify_restored(self):
            return {"ok": True, "mismatches": []}

        def report(self):
            return {"schema": "fake"}

    async def run():
        control = Control()
        server = AgentServer(
            "node",
            host="127.0.0.1",
            port=0,
            physical_control=control,
        )
        server._workload_enabled = False
        await server._lease_expired(0.0)
        assert control.restore_calls == 1
        assert server._workload_enabled is True
        assert server._control_auto_restore["reason"] == "lease_expired"

    asyncio.run(run())


def test_agent_control_lease_renewal_keeps_replacement_timer_tracked():
    class Control:
        def capabilities(self):
            class Capabilities:
                netem = False
                route = False
                cpu_capacity = False
                interfaces = ()
                cpu_max_path = None

            return Capabilities()

        async def restore(self):
            return {"ok": True, "restored": []}

        async def verify_restored(self):
            return {"ok": True, "mismatches": []}

        def report(self):
            return {"schema": "fake"}

    async def run():
        server = AgentServer(
            "node",
            host="127.0.0.1",
            port=0,
            physical_control=Control(),
        )
        server._arm_control_lease(5.0)
        first = server._control_lease_task
        assert first is not None

        server._arm_control_lease(5.0)
        replacement = server._control_lease_task
        assert replacement is not None and replacement is not first
        await asyncio.gather(first, return_exceptions=True)

        assert server._control_lease_task is replacement
        assert server._control_lease_deadline is not None
        await server.close()

    asyncio.run(run())
