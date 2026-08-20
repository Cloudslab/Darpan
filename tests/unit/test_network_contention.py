from __future__ import annotations

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState, LinkState, NodeState
from darpan.core.topology import LinkSpec
from darpan.twin.models.network import NetworkDelayModel


def _state():
    link = LinkSpec("shared", "edge", "cloud", latency_ms=0, bandwidth_mbps=8)
    return ContinuumState(
        nodes={"edge": NodeState("edge", "edge"), "cloud": NodeState("cloud", "cloud")},
        links={"shared": LinkState(link)},
    )


def test_network_model_reserves_shared_path_bandwidth_exclusively():
    model = NetworkDelayModel()
    first = model.predict(
        {"source": "edge", "target": "cloud", "size_bytes": 1_000_000},
        _state(),
    )
    assert first.metadata["links"] == ["shared"]
    assert first.estimate == 1.0
    second = model.predict(
        {
            "source": "edge",
            "target": "cloud",
            "size_bytes": 1_000_000,
            "earliest_start": 0.0,
            "reservations": (
                {"start": 0.0, "end": 1.0, "links": ["shared"]},
            ),
        },
        _state(),
    )
    assert second.metadata["contention_wait_s"] == 1.0
    assert second.metadata["base_transfer_s"] == 1.0
    assert second.estimate == 2.0


def test_network_calibration_is_scoped_to_explicit_route():
    state = ContinuumState(
        nodes={
            name: NodeState(name, "edge")
            for name in ("edge", "a", "b", "cloud")
        },
        links={
            "edge-a": LinkState(
                LinkSpec("edge-a", "edge", "a", bandwidth_mbps=8)
            ),
            "a-cloud": LinkState(
                LinkSpec("a-cloud", "a", "cloud", bandwidth_mbps=8)
            ),
            "edge-b": LinkState(
                LinkSpec("edge-b", "edge", "b", bandwidth_mbps=8)
            ),
            "b-cloud": LinkState(
                LinkSpec("b-cloud", "b", "cloud", bandwidth_mbps=8)
            ),
        },
    )
    model = NetworkDelayModel(calibration_rate=1.0)
    size = 1_000_000
    route_a = {
        "source": "edge",
        "target": "cloud",
        "size_bytes": size,
        "path": ["edge", "a", "cloud"],
        "links": ["edge-a", "a-cloud"],
    }
    route_b = {
        "source": "edge",
        "target": "cloud",
        "size_bytes": size,
        "path": ["edge", "b", "cloud"],
        "links": ["edge-b", "b-cloud"],
    }
    baseline_a = model.predict(route_a, state)
    baseline_b = model.predict(route_b, state)
    model.observe(
        Event(
            kind=EventKind.DATA_TRANSFER_COMPLETED,
            event_time=2.0,
            source="real",
            payload={
                "source_node_id": "edge",
                "target_node_id": "cloud",
                "size_bytes": size,
                "duration_s": baseline_a.estimate * 2.0,
                "route_path": ["edge", "a", "cloud"],
                "route_links": ["edge-a", "a-cloud"],
            },
        ),
        state,
    )
    calibrated_a = model.predict(route_a, state)
    untouched_b = model.predict(route_b, state)
    assert calibrated_a.estimate == baseline_a.estimate * 2.0
    assert calibrated_a.metadata["calibrated"] is True
    assert untouched_b.estimate == baseline_b.estimate
    assert untouched_b.metadata["calibrated"] is False
