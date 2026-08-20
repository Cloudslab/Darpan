from __future__ import annotations

import asyncio
import json

from darpan.cli.cluster import _validate
from darpan.runtime.real.agent import AgentServer


class Args:
    cluster: str
    system: str | None = None
    exercise_data_plane = False
    payload_bytes = 1024
    timeout = 5.0
    output: str


def test_cluster_validate_cli_writes_durable_inputs_and_checksums(tmp_path):
    async def run():
        server = AgentServer("edge", host="127.0.0.1", port=0)
        await server.start()
        try:
            cluster = tmp_path / "cluster.yaml"
            cluster.write_text(
                f"nodes:\n  - id: edge\n    host: 127.0.0.1\n    port: {server.port}\n",
                encoding="utf-8",
            )
            args = Args()
            args.cluster = str(cluster)
            args.output = str(tmp_path / "report")
            await _validate(args)
            report = tmp_path / "report"
            payload = json.loads((report / "cluster-validation.json").read_text())
            assert payload["ready"] is True
            assert (report / "inputs" / "cluster.yaml").is_file()
            checksums = json.loads((report / "inputs" / "checksums.json").read_text())
            assert len(checksums["cluster.yaml"]) == 64
        finally:
            await server.close()

    asyncio.run(run())
