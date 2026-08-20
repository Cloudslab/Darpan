from __future__ import annotations

import asyncio
from pathlib import Path

from darpan.experiment.artifact import verify_artifact
from darpan.experiment.study import validate_physical_study
from darpan.runtime.real.agent import AgentServer


class _Control:
    def capabilities(self):
        class Capabilities:
            netem = True
            route = True
            cpu_capacity = False
            interfaces = ("eth-test",)
            cpu_max_path = None

        return Capabilities()

    async def restore(self):
        return {"ok": True, "restored": []}

    async def verify_restored(self):
        return {"ok": True, "mismatches": []}

    def report(self):
        return {"schema": "fake"}


def test_study_validate_checks_plan_cluster_clock_and_control(tmp_path: Path) -> None:
    async def run() -> None:
        server = AgentServer(
            "edge",
            host="127.0.0.1",
            port=0,
            physical_control=_Control(),
        )
        await server.start()
        try:
            (tmp_path / "cluster.yaml").write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    host: 127.0.0.1\n"
                f"    port: {server.port}\n"
                "    physical_control: true\n"
                "    control_interfaces: [eth-test]\n",
                encoding="utf-8",
            )
            (tmp_path / "system.yaml").write_text(
                "nodes:\n  - id: edge\n",
                encoding="utf-8",
            )
            campaign = tmp_path / "campaign.yaml"
            campaign.write_text(
                "name: physical-ready\n"
                "output: campaign-results\n"
                "jobs:\n"
                "  - id: preflight\n"
                "    kind: cluster_preflight\n"
                "    cluster: cluster.yaml\n"
                "    system: system.yaml\n",
                encoding="utf-8",
            )
            output = tmp_path / "readiness"
            report = await validate_physical_study(
                campaign,
                max_clock_offset_s=0.5,
                output=output,
            )
            assert report.ready is True
            assert len(report.plan_fingerprint) == 64
            assert report.checks[0]["clock"][0]["absolute_offset_s"] < 0.5
            assert report.checks[0]["environment_fingerprint"]
            verified = verify_artifact(output)
            assert verified["verified"] is True
            assert verified["schema"] == "darpan.study-readiness/v1"
        finally:
            await server.close()

    asyncio.run(run())
