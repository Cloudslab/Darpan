from __future__ import annotations

import asyncio

from darpan.runtime.real.executors.remote import RemoteExecutionHandle, RemoteExecutor


def test_remote_wait_repolls_after_long_poll_timeouts():
    class Client:
        def __init__(self):
            self.waits = 0
            self.released = False

        async def wait_execution(self, execution_id):
            assert execution_id == "execution-1"
            self.waits += 1
            if self.waits <= 4:
                raise TimeoutError("simulated long-poll timeout")
            return {
                "cancelled": False,
                "result": {
                    "return_code": 0,
                    "duration_s": 2.5,
                    "stdout": "done",
                    "stderr": "",
                    "output_bytes": 0,
                    "measurements": {},
                },
            }

        async def release_execution(self, execution_id):
            assert execution_id == "execution-1"
            self.released = True
            return True

    async def run():
        client = Client()
        result = await RemoteExecutionHandle(client, "execution-1").wait()
        assert result.success is True
        assert result.duration_s == 2.5
        assert client.waits == 5
        assert client.released is True

    asyncio.run(run())


def test_direct_transfer_recovers_a_lost_completion_response_by_target_size():
    class SourceClient:
        ssl_context = None

        async def forward_artifact(self, *args, **kwargs):
            del args, kwargs
            raise OSError(121, "simulated lost artifact completion")

    class TargetClient:
        ssl_context = None

        def direct_transfer_endpoint(self):
            return {
                "host": "target.example",
                "port": 8765,
                "ca_pem": None,
                "server_hostname": None,
            }

        async def prepare_incoming_transfer(self, *args, **kwargs):
            del args, kwargs
            return "ticket-1"

        async def stat_file(self, workspace_id, path):
            assert workspace_id == "target-workspace"
            assert path == "input.bin"
            return 4096

    async def run():
        source = RemoteExecutor(SourceClient(), direct_artifact_forward=True)
        target = RemoteExecutor(TargetClient(), direct_artifact_forward=True)
        size, duration = await source.copy_file_to(
            "source-workspace",
            "output.bin",
            target,
            "target-workspace",
            "input.bin",
            chunk_size=1024,
            expected_size=4096,
        )
        assert size == 4096
        assert duration is None

    asyncio.run(run())
