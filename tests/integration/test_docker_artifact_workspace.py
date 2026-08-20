from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

from darpan.core.application import ComponentSpec
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.executors.docker import DockerExecutor
from darpan.runtime.real.transport import AgentClient


def _fake_docker(path: Path) -> Path:
    script = path / "fake-docker"
    script.write_text(
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
assert args[:2] == ["run", "--rm"]
index = 2
host_workspace = None
while index < len(args) and args[index].startswith("--"):
    flag = args[index]
    if flag == "--volume":
        host_workspace = args[index + 1].split(":", 1)[0]
        index += 2
    elif flag == "--workdir":
        index += 2
    else:
        raise SystemExit(f"unsupported fake docker flag: {flag}")
image = args[index]
assert image
command = args[index + 1:]
result = subprocess.run(command, cwd=host_workspace, check=False)
raise SystemExit(result.returncode)
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_agent_container_execution_uses_same_artifact_workspace(tmp_path):
    async def run():
        fake = _fake_docker(tmp_path)
        root = tmp_path / "agent-workspaces"
        server = AgentServer("edge", host="127.0.0.1", port=0, workspace_root=root)
        server.docker = DockerExecutor(executable=str(fake), workspace_root=root)
        await server.start()
        client = AgentClient("127.0.0.1", server.port)
        workspace_id = "container-artifact"
        await client.put_file(workspace_id, "input.txt", b"hello container")
        component = ComponentSpec(
            "container",
            image="fake:image",
            command=(
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "Path('output.txt').write_bytes(Path('input.txt').read_bytes() + b'!')",
            ),
        )
        execution_id = await client.start_execution(
            {
                "id": component.id,
                "kind": component.kind,
                "image": component.image,
                "command": list(component.command),
                "resources": [],
                "work_units": component.work_units,
                "labels": {},
            },
            workspace_id=workspace_id,
        )
        response = await client.wait_execution(execution_id)
        assert response["result"]["return_code"] == 0
        assert await client.get_file(workspace_id, "output.txt") == b"hello container!"
        await server.close()

    asyncio.run(run())
