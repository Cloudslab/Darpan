from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from darpan.cli.run import run_experiment_spec
from darpan.experiment.spec import ExperimentSpec
from darpan.runtime.real.agent import AgentServer


class _Control:
    def __init__(self) -> None:
        self.restored = False

    def capabilities(self):
        class Capabilities:
            netem = False
            route = False
            cpu_capacity = True
            interfaces = ()
            cpu_max_path = "/test/cpu.max"

        return Capabilities()

    async def apply_netem(self, *args, **kwargs):
        raise AssertionError("not used")

    async def set_cpu_capacity(self, cpus):
        return {"cpus": cpus}

    async def bind_route(self, **kwargs):
        raise AssertionError("not used")

    async def clear_route(self, **kwargs):
        raise AssertionError("not used")

    async def restore(self):
        self.restored = True
        return {"ok": True, "restored": ["cpu.max"]}

    async def verify_restored(self):
        return {"ok": self.restored, "mismatches": []}

    def report(self):
        return {"schema": "fake-linux-control/v1", "restored": self.restored}


def test_real_scenario_artifact_contains_verified_physical_restoration(tmp_path: Path) -> None:
    async def run() -> None:
        control = _Control()
        server = AgentServer(
            "edge",
            host="127.0.0.1",
            port=0,
            physical_control=control,
        )
        await server.start()
        try:
            (tmp_path / "cluster.yaml").write_text(
                "nodes:\n"
                "  - id: edge\n"
                "    host: 127.0.0.1\n"
                f"    port: {server.port}\n"
                "    physical_control: true\n",
                encoding="utf-8",
            )
            (tmp_path / "system.yaml").write_text(
                "name: one-node\n"
                "nodes:\n"
                "  - id: edge\n"
                "    resources:\n"
                "      - name: cpu\n"
                "        capacity: 1\n",
                encoding="utf-8",
            )
            (tmp_path / "app.yaml").write_text(
                "id: demo\n"
                "components:\n"
                "  - id: task\n"
                f"    command: [{json.dumps(sys.executable)}, -c, pass]\n"
                "    resources:\n"
                "      - name: cpu\n"
                "        amount: 0.25\n",
                encoding="utf-8",
            )
            (tmp_path / "scenario.yaml").write_text(
                "name: cpu-control\n"
                "events:\n"
                "  - at: 0\n"
                "    kind: measurement.observed\n"
                "    target: edge\n"
                "    name: compute.cpu_capacity\n"
                "    value: 0.5\n"
                "    unit: count\n",
                encoding="utf-8",
            )
            experiment = tmp_path / "experiment.yaml"
            experiment.write_text(
                "system: system.yaml\n"
                "application: app.yaml\n"
                "runtime: real\n"
                "cluster: cluster.yaml\n"
                "physical_control: true\n"
                "scenario: scenario.yaml\n"
                "policy: first-fit\n"
                "metrics: [application_latency_s]\n"
                "output: results\n",
                encoding="utf-8",
            )
            await run_experiment_spec(ExperimentSpec.load(experiment), emit_output=False)
            payload = json.loads(
                (tmp_path / "results/run-0001/physical-control.json").read_text(
                    encoding="utf-8"
                )
            )
            assert payload["restore_verified"] is True
            assert payload["actions"][0]["event_kind"] == "measurement.observed"
            assert payload["restore_results"]["edge"]["ok"] is True
            assert control.restored is True
        finally:
            await server.close()

    asyncio.run(run())
