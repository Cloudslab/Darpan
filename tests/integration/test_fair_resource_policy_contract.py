from __future__ import annotations

import asyncio

from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.core.workload import ArrivalSpec, WorkloadSpec
from darpan.experiment.baselines import FirstFitPolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


def test_first_fit_policy_admits_overlapping_jobs_to_fair_cpu():
    async def run():
        system = SystemSpec(
            nodes=(
                NodeSpec(
                    "edge",
                    resources=(
                        ResourceSpec("cpu", 1.0, attributes={"scheduling": "fair"}),
                    ),
                ),
            )
        )
        app = ApplicationSpec(
            "job",
            components=(
                ComponentSpec(
                    "work",
                    resources=(ResourceRequest("cpu", 1.0),),
                    work_units=1.0,
                ),
            ),
        )
        workload = WorkloadSpec(
            applications=(app,),
            arrivals=(ArrivalSpec("job", 0.0, count=2),),
        )
        session = Session(TwinBackend())
        runner = ExperimentRunner(session, metrics=(ApplicationLatency(),))
        try:
            result = await runner.run_workload(
                system,
                workload,
                FirstFitPolicy(),
                timeout=5,
            )
            starts = [
                event
                for event in session.event_log
                if event.kind == "component.started"
            ]
            assert len(starts) == 2
            assert starts[0].event_time == starts[1].event_time == 0.0
            assert result.metrics["application_latency_s"] == 2.0
        finally:
            await session.close()

    asyncio.run(run())
