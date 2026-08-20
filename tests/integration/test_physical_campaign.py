from __future__ import annotations

import asyncio
import json
import sys

from darpan.cli.run import run_experiment_spec
from darpan.experiment.campaign import CampaignRunner, CampaignSpec
from darpan.experiment.suite import PaperSuite
from darpan.runtime.real.agent import AgentServer


def test_campaign_runs_physical_preflight_and_runtime_exercise(tmp_path):
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
            cluster = tmp_path / "cluster.yaml"
            cluster.write_text(
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
            system = tmp_path / "system.yaml"
            system.write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    resources:\n"
                "      cpu: 0.25\n"
                "  - id: fog\n"
                "    resources:\n"
                "      cpu: 0.25\n"
                "links:\n"
                "  - id: edge-fog\n"
                "    source: edge\n"
                "    target: fog\n",
                encoding="utf-8",
            )
            suite = tmp_path / "suite.yaml"
            suite.write_text(
                "name: physical\nversion: '1'\nitems:\n"
                "  - id: cluster\n    kind: cluster\n    path: cluster.yaml\n"
                "  - id: system\n    kind: system\n    path: system.yaml\n",
                encoding="utf-8",
            )
            lock = tmp_path / "suite.lock.json"
            lock.write_text(
                json.dumps(PaperSuite.load(suite).lock_payload()),
                encoding="utf-8",
            )
            campaign = tmp_path / "campaign.yaml"
            campaign.write_text(
                "name: physical\n"
                "output: results\n"
                "suite: suite.yaml\n"
                "suite_lock: suite.lock.json\n"
                "jobs:\n"
                "  - id: preflight\n"
                "    kind: cluster_preflight\n"
                "    cluster: suite:cluster\n"
                "    system: suite:system\n"
                "    exercise_data_plane: true\n"
                "    payload_bytes: 1024\n"
                "  - id: lifecycle\n"
                "    kind: cluster_exercise\n"
                "    cluster: suite:cluster\n"
                "    system: suite:system\n"
                "    source: edge\n"
                "    target: fog\n"
                f"    python_command: {sys.executable}\n"
                "    cpu: 0.1\n"
                "    timeout_s: 5\n",
                encoding="utf-8",
            )

            async def execute(spec):
                return await run_experiment_spec(spec, emit_output=False)

            payload = await CampaignRunner(
                CampaignSpec.load(campaign), execute=execute
            ).run()
            assert payload["successful_jobs"] == 2
            assert payload["jobs"]["preflight"]["result"]["ready"] is True
            exercise = payload["jobs"]["lifecycle"]["result"]
            assert exercise["ready"] is True
            assert tuple(step["name"] for step in exercise["steps"]) == (
                "place_source",
                "restart_source",
                "migrate_target",
                "stop_target",
            )
            assert (tmp_path / "results" / "jobs" / "preflight.json").is_file()
            assert (tmp_path / "results" / "jobs" / "lifecycle.json").is_file()
        finally:
            await target.close()
            await source.close()

    asyncio.run(run())
