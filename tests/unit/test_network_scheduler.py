from __future__ import annotations

import pytest

from darpan.core.state import ContinuumState, LinkState, NodeState
from darpan.core.topology import LinkSpec
from darpan.twin.network_scheduler import MaxMinNetworkScheduler, TwinTransfer


def _state(*, bandwidth_mbps: float = 8.0) -> ContinuumState:
    return ContinuumState(
        nodes={
            "edge": NodeState("edge", "edge"),
            "cloud": NodeState("cloud", "cloud"),
        },
        links={
            "shared": LinkState(
                LinkSpec(
                    "shared",
                    "edge",
                    "cloud",
                    latency_ms=0,
                    bandwidth_mbps=bandwidth_mbps,
                )
            )
        },
    )


def _transfer(name: str, *, started_at: float = 0.0) -> TwinTransfer:
    return TwinTransfer(
        id=name,
        links=("shared",),
        size_bytes=1_000_000,
        started_at=started_at,
        ready_at=started_at,
        work_bits=8_000_000,
        baseline_duration_s=1.0,
    )


def test_max_min_scheduler_shares_one_link_symmetrically():
    scheduler = MaxMinNetworkScheduler()
    state = _state()
    scheduler.add(_transfer("a"), now=0.0, state=state)
    scheduler.add(_transfer("b"), now=0.0, state=state)
    assert scheduler.next_event_at(now=0.0) == pytest.approx(2.0)
    completed = scheduler.advance(now=2.0, state=state)
    assert {item.id for item in completed} == {"a", "b"}


def test_max_min_scheduler_rebalances_when_transfer_joins_late():
    scheduler = MaxMinNetworkScheduler()
    state = _state()
    scheduler.add(_transfer("a"), now=0.0, state=state)
    scheduler.add(_transfer("b", started_at=0.5), now=0.5, state=state)
    assert scheduler.next_event_at(now=0.5) == pytest.approx(1.5)
    completed = scheduler.advance(now=1.5, state=state)
    assert [item.id for item in completed] == ["a"]
    assert scheduler.next_event_at(now=1.5) == pytest.approx(2.0)
    completed = scheduler.advance(now=2.0, state=state)
    assert [item.id for item in completed] == ["b"]


def test_max_min_scheduler_reacts_to_live_bandwidth_change():
    scheduler = MaxMinNetworkScheduler()
    scheduler.add(_transfer("a"), now=0.0, state=_state(bandwidth_mbps=8.0))
    assert scheduler.next_event_at(now=0.0) == pytest.approx(1.0)
    scheduler.topology_changed(now=0.25, state=_state(bandwidth_mbps=4.0))
    # 2 Mbit were sent in the first 0.25 s; 6 Mbit remain at 4 Mbit/s.
    assert scheduler.next_event_at(now=0.25) == pytest.approx(1.75)
    completed = scheduler.advance(now=1.75, state=_state(bandwidth_mbps=4.0))
    assert [item.id for item in completed] == ["a"]


def test_scheduler_completes_sub_bit_residual_at_nonzero_clock():
    scheduler = MaxMinNetworkScheduler()
    state = _state(bandwidth_mbps=1000.0)
    started_at = 0.09927468800000001
    ready_at = started_at + 0.01
    transfer = TwinTransfer(
        id="residual",
        links=("shared",),
        size_bytes=131_072,
        started_at=started_at,
        ready_at=ready_at,
        work_bits=1_048_576,
        baseline_duration_s=0.011048576,
    )

    scheduler.add(transfer, now=started_at, state=state)
    assert scheduler.next_event_at(now=started_at) == pytest.approx(ready_at)
    assert scheduler.advance(now=ready_at, state=state) == ()

    completion_at = scheduler.next_event_at(now=ready_at)
    assert completion_at is not None
    completed = scheduler.advance(now=completion_at, state=state)

    assert [item.id for item in completed] == ["residual"]
    assert scheduler.next_event_at(now=completion_at) is None


def test_scheduler_completes_tiny_flow_when_completion_rounds_at_large_clock():
    scheduler = MaxMinNetworkScheduler()
    state = _state(bandwidth_mbps=1000.0)
    started_at = 20.118999220335667
    ready_at = started_at + 0.01
    transfer = TwinTransfer(
        id="tiny-residual",
        links=("shared",),
        size_bytes=128,
        started_at=started_at,
        ready_at=ready_at,
        work_bits=1024,
        baseline_duration_s=0.010001024,
    )

    scheduler.add(transfer, now=started_at, state=state)
    assert scheduler.advance(now=ready_at, state=state) == ()

    completion_at = scheduler.next_event_at(now=ready_at)
    assert completion_at is not None
    completed = scheduler.advance(now=completion_at, state=state)

    assert [item.id for item in completed] == ["tiny-residual"]
    assert scheduler.next_event_at(now=completion_at) is None
