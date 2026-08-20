from __future__ import annotations

import asyncio

from darpan.core.application import ApplicationSpec, ComponentSpec, FlowSpec
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.baselines import LatencyAwarePolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


def test_latency_aware_policy_prefers_lower_input_transfer_path() -> None:
    async def run() -> None:
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("camera", 1),)),
                NodeSpec("cloud", resources=(ResourceSpec("cpu", 2),)),
                NodeSpec("fog", resources=(ResourceSpec("cpu", 2),)),
            ),
            links=(
                LinkSpec("edge-fog", "edge", "fog", latency_ms=2, bandwidth_mbps=100),
                LinkSpec(
                    "edge-cloud",
                    "edge",
                    "cloud",
                    latency_ms=50,
                    bandwidth_mbps=100,
                ),
            ),
        )
        app = ApplicationSpec(
            "pipeline",
            components=(
                ComponentSpec(
                    "source",
                    resources=(ResourceRequest("camera", 1),),
                    work_units=0.1,
                ),
                ComponentSpec(
                    "analyze",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.1,
                ),
            ),
            flows=(FlowSpec("source", "analyze", data_size_bytes=1_000_000),),
        )
        session = Session(TwinBackend())
        try:
            result = await ExperimentRunner(session, metrics=[ApplicationLatency()]).run(
                system,
                app,
                LatencyAwarePolicy(),
            )
            analyze = next(
                item
                for item in session.state.components.values()
                if item.component_id == "analyze"
            )
            assert result.feasible is True
            assert analyze.node_id == "fog"
        finally:
            await session.close()

    asyncio.run(run())
