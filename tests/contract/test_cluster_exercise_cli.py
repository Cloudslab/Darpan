from __future__ import annotations

import asyncio
import json
import sys

from darpan.cli.cluster import _exercise
from darpan.runtime.real.agent import AgentServer


class Args:
    cluster: str
    system: str
    source = "edge"
    target = "fog"
    python_command = sys.executable
    cpu = 0.1
    timeout = 5.0
    output: str


def test_cluster_exercise_cli_writes_durable_runtime_report(tmp_path):
    async def run() -> None:
        edge = AgentServer("edge", host="127.0.0.1", port=0)
        fog = AgentServer("fog", host="127.0.0.1", port=0)
        await edge.start()
        await fog.start()
        try:
            cluster = tmp_path / "cluster.yaml"
            cluster.write_text(
                "nodes:\n"
                f"  - id: edge\n    host: 127.0.0.1\n    port: {edge.port}\n"
                f"  - id: fog\n    host: 127.0.0.1\n    port: {fog.port}\n",
                encoding="utf-8",
            )
            system = tmp_path / "system.yaml"
            system.write_text(
                "nodes:\n"
                "  - id: edge\n    resources:\n      - name: cpu\n        capacity: 0.25\n"
                "  - id: fog\n    resources:\n      - name: cpu\n        capacity: 0.25\n"
                "links:\n"
                "  - id: edge-fog\n    source: edge\n    target: fog\n",
                encoding="utf-8",
            )
            args = Args()
            args.cluster = str(cluster)
            args.system = str(system)
            args.output = str(tmp_path / "report")
            await _exercise(args)

            output = tmp_path / "report"
            report = json.loads((output / "cluster-exercise.json").read_text())
            assert report["ready"] is True
            restart = next(
                step for step in report["steps"] if step["name"] == "restart_source"
            )
            assert restart["details"]["restart_mode"] == "process"
            assert restart["details"]["canonical_downtime_s"] >= 0
            migrate = next(
                step for step in report["steps"] if step["name"] == "migrate_target"
            )
            assert migrate["details"]["migration_mode"] == "restart"
            assert migrate["details"]["canonical_downtime_s"] >= 0
            assert (output / "inputs" / "cluster.yaml").is_file()
            assert (output / "inputs" / "system.yaml").is_file()
            assert (output / "provenance.json").is_file()
            checksums = json.loads((output / "inputs" / "checksums.json").read_text())
            assert len(checksums["cluster.yaml"]) == 64
            assert len(checksums["system.yaml"]) == 64
        finally:
            await fog.close()
            await edge.close()

    asyncio.run(run())
