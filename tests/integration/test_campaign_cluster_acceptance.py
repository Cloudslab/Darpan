from __future__ import annotations

import asyncio
from pathlib import Path

from darpan.experiment.campaign import CampaignRunner, CampaignSpec
from darpan.runtime.real.agent import AgentServer


def test_campaign_cluster_acceptance_is_durable_physical_evidence(tmp_path: Path):
    async def run() -> None:
        source = AgentServer(
            "edge",
            host="127.0.0.1",
            port=0,
            allow_network_probe=True,
            allow_artifact_forward=True,
        )
        target = AgentServer("fog", host="127.0.0.1", port=0)
        await source.start()
        await target.start()
        try:
            (tmp_path / "cluster.yaml").write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    host: 127.0.0.1\n"
                f"    port: {source.port}\n"
                "    network_probe: true\n"
                "    artifact_forward: true\n"
                "  - id: fog\n"
                "    host: 127.0.0.1\n"
                f"    port: {target.port}\n",
                encoding="utf-8",
            )
            (tmp_path / "system.yaml").write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    resources: {cpu: 1}\n"
                "  - id: fog\n"
                "    resources: {cpu: 1}\n"
                "links:\n"
                "  - id: edge-fog\n"
                "    source: edge\n"
                "    target: fog\n",
                encoding="utf-8",
            )
            campaign = tmp_path / "campaign.yaml"
            campaign.write_text(
                "name: acceptance-campaign\n"
                "output: results\n"
                "jobs:\n"
                "  - id: acceptance\n"
                "    kind: cluster_acceptance\n"
                "    evidence_for: [deployment-acceptance]\n"
                "    cluster: cluster.yaml\n"
                "    system: system.yaml\n"
                "    source: edge\n"
                "    target: fog\n"
                "    python_command: python3\n"
                "    cpu: 0.1\n"
                "    timeout_s: 5\n",
                encoding="utf-8",
            )

            async def no_experiment(_spec):
                raise AssertionError("acceptance jobs do not execute ExperimentSpec")

            payload = await CampaignRunner(
                CampaignSpec.load(campaign), execute=no_experiment
            ).run()
            result = payload["jobs"]["acceptance"]["result"]
            assert result["ready_for_study"] is True
            assert result["lifecycle"]["ready"] is True
            assert result["steps"][0]["name"] == "scale_out_in"

        finally:
            await target.close()
            await source.close()

    asyncio.run(run())
