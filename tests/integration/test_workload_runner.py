from __future__ import annotations

import asyncio

from darpan import Darpan
from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.event import EventKind
from darpan.core.resource import ResourceRequest
from darpan.core.workload import ArrivalSpec, WorkloadSpec
from darpan.experiment.baselines import FirstFitPolicy
from darpan.experiment.metric import ApplicationLatency, EventCount
from darpan.experiment.runner import ExperimentRunner


def _workload() -> WorkloadSpec:
    app = ApplicationSpec(
        "job",
        (
            ComponentSpec(
                "task",
                resources=(ResourceRequest("cpu", 1),),
                work_units=2.0,
            ),
        ),
    )
    return WorkloadSpec(
        (app,),
        (
            ArrivalSpec("job", at_s=0.0, count=1),
            ArrivalSpec("job", at_s=0.5, count=1),
        ),
        name="overlap",
    )


def test_twin_workload_preserves_timed_overlapping_arrivals(small_system):
    async def run():
        session = Darpan.twin()
        runner = ExperimentRunner(
            session,
            metrics=[
                ApplicationLatency(),
                EventCount(EventKind.APPLICATION_COMPLETED, name="completed"),
            ],
        )
        result = await runner.run_workload(
            small_system,
            _workload(),
            FirstFitPolicy(),
            timeout=2,
        )
        submitted = [
            event
            for event in session.event_log
            if event.kind == EventKind.APPLICATION_SUBMITTED
        ]
        assert [event.event_time for event in submitted] == [0.0, 0.5]
        assert result.metrics["completed"] == 2
        assert session.state.time >= 2.0
        await session.close()

    asyncio.run(run())


def test_real_workload_uses_same_arrival_contract(small_system):
    async def run():
        session = Darpan.real()
        runner = ExperimentRunner(
            session,
            metrics=[EventCount(EventKind.APPLICATION_COMPLETED, name="completed")],
        )
        workload = _workload()
        # Keep the physical test fast while still exercising delayed arrivals.
        workload = WorkloadSpec(
            workload.applications,
            (
                ArrivalSpec("job", at_s=0.0),
                ArrivalSpec("job", at_s=0.02),
            ),
            name="real-overlap",
        )
        result = await runner.run_workload(
            small_system,
            workload,
            FirstFitPolicy(),
            timeout=3,
        )
        assert result.metrics["completed"] == 2
        submitted = [
            event
            for event in session.event_log
            if event.kind == EventKind.APPLICATION_SUBMITTED
        ]
        assert len(submitted) == 2
        assert submitted[1].event_time >= submitted[0].event_time
        await session.close()

    asyncio.run(run())


def test_real_workload_serializes_same_time_delayed_arrivals(small_system):
    async def run():
        app = ApplicationSpec(
            "simultaneous-job",
            (
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=0.01,
                ),
            ),
        )
        workload = WorkloadSpec(
            (app,),
            (ArrivalSpec("simultaneous-job", at_s=0.02, count=4),),
            name="real-same-time-arrivals",
        )
        session = Darpan.real()
        runner = ExperimentRunner(
            session,
            metrics=[EventCount(EventKind.APPLICATION_COMPLETED, name="completed")],
        )
        result = await runner.run_workload(
            small_system,
            workload,
            FirstFitPolicy(),
            timeout=3,
        )
        assert result.metrics["completed"] == 4

        events = list(session.event_log)
        for submitted in (
            event for event in events if event.kind == EventKind.APPLICATION_SUBMITTED
        ):
            component_id = f"{submitted.subject}:task"
            created_index = next(
                index
                for index, event in enumerate(events)
                if event.kind == EventKind.COMPONENT_CREATED
                and event.subject == component_id
            )
            ready_index = next(
                index
                for index, event in enumerate(events)
                if event.kind == EventKind.COMPONENT_READY
                and event.subject == component_id
            )
            assert created_index < ready_index
        await session.close()

    asyncio.run(run())
