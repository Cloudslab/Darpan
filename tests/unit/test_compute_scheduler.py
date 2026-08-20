from darpan.core.resource import ResourceState
from darpan.core.state import ContinuumState, NodeState
from darpan.twin.compute_scheduler import MaxMinComputeScheduler, TwinComputeJob


def _state(capacity: float = 1.0) -> ContinuumState:
    return ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={
                    "cpu": ResourceState(
                        "cpu",
                        capacity,
                        attributes={"scheduling": "fair"},
                    )
                },
            )
        }
    )


def test_two_jobs_share_cpu_symmetrically() -> None:
    state = _state()
    scheduler = MaxMinComputeScheduler()
    scheduler.add(
        TwinComputeJob("a", "edge", max_cpu=1.0, work_cpu_seconds=1.0, started_at=0.0),
        now=0.0,
        state=state,
    )
    scheduler.add(
        TwinComputeJob("b", "edge", max_cpu=1.0, work_cpu_seconds=1.0, started_at=0.0),
        now=0.0,
        state=state,
    )

    assert scheduler.next_event_at(now=0.0) == 2.0
    assert {job.id for job in scheduler.advance(now=2.0, state=state)} == {"a", "b"}


def test_late_join_rebalances_existing_job() -> None:
    state = _state()
    scheduler = MaxMinComputeScheduler()
    scheduler.add(
        TwinComputeJob("a", "edge", max_cpu=1.0, work_cpu_seconds=1.0, started_at=0.0),
        now=0.0,
        state=state,
    )
    scheduler.add(
        TwinComputeJob("b", "edge", max_cpu=1.0, work_cpu_seconds=1.0, started_at=0.5),
        now=0.5,
        state=state,
    )

    assert scheduler.next_event_at(now=0.5) == 1.5
    assert [job.id for job in scheduler.advance(now=1.5, state=state)] == ["a"]
    assert scheduler.next_event_at(now=1.5) == 2.0
    assert [job.id for job in scheduler.advance(now=2.0, state=state)] == ["b"]
