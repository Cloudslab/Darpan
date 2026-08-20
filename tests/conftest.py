from __future__ import annotations

import pytest

from darpan import (
    ApplicationSpec,
    ComponentSpec,
    FlowSpec,
    LinkSpec,
    NodeSpec,
    ResourceRequest,
    ResourceSpec,
    SystemSpec,
)


@pytest.fixture
def small_system():
    return SystemSpec(
        nodes=(
            NodeSpec(
                "edge-1",
                resources=(ResourceSpec("cpu", 2), ResourceSpec("memory", 4)),
            ),
            NodeSpec(
                "fog-1",
                tier="fog",
                resources=(ResourceSpec("cpu", 4), ResourceSpec("memory", 8)),
            ),
            NodeSpec(
                "cloud-1",
                tier="cloud",
                resources=(ResourceSpec("cpu", 8), ResourceSpec("memory", 16)),
            ),
        ),
        links=(
            LinkSpec("edge-fog", "edge-1", "fog-1", latency_ms=5, bandwidth_mbps=100),
            LinkSpec("fog-cloud", "fog-1", "cloud-1", latency_ms=20, bandwidth_mbps=1000),
        ),
    )


@pytest.fixture
def small_app():
    return ApplicationSpec(
        "demo",
        components=(
            ComponentSpec(
                "a",
                command=("python", "-c", "print('a')"),
                resources=(ResourceRequest("cpu", 1),),
                work_units=0.2,
            ),
            ComponentSpec(
                "b",
                command=("python", "-c", "print('b')"),
                resources=(ResourceRequest("cpu", 1),),
                work_units=0.3,
            ),
        ),
        flows=(FlowSpec("a", "b", data_size_bytes=1024),),
    )
