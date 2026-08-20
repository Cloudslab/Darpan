from __future__ import annotations

import asyncio

from darpan.core.state import (
    ComponentInstanceState,
    ContinuumState,
    FlowRouteBinding,
    LinkState,
    NodeState,
)
from darpan.core.topology import LinkSpec
from darpan.runtime.real.linux_network import AgentLinuxNetworkControlDriver


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def control_route_bind(self, **kwargs):
        self.calls.append(("bind", dict(kwargs)))
        return kwargs

    async def control_route_clear(self, **kwargs):
        self.calls.append(("clear", dict(kwargs)))
        return kwargs


def test_linux_network_driver_maps_darpan_path_to_host_route() -> None:
    async def run() -> None:
        edge = _Client()
        state = ContinuumState(
            nodes={
                "edge": NodeState("edge", "edge"),
                "fog": NodeState("fog", "fog"),
                "cloud": NodeState(
                    "cloud",
                    "cloud",
                    labels={"route_ipv4": "10.0.0.9"},
                ),
            },
            links={
                "edge-fog": LinkState(
                    LinkSpec(
                        "edge-fog",
                        "edge",
                        "fog",
                        labels={
                            "physical_source_interface": "eth-route",
                            "physical_target_ipv4": "10.0.0.2",
                        },
                    )
                ),
                "fog-cloud": LinkState(LinkSpec("fog-cloud", "fog", "cloud")),
            },
            components={
                "app:source": ComponentInstanceState(
                    "app:source",
                    "demo",
                    "app",
                    "source",
                    status="running",
                    node_id="edge",
                ),
                "app:target": ComponentInstanceState(
                    "app:target",
                    "demo",
                    "app",
                    "target",
                    status="running",
                    node_id="cloud",
                ),
            },
        )
        binding = FlowRouteBinding(
            application_id="demo",
            application_instance_id="app",
            source_component_id="source",
            target_component_id="target",
            path=("edge", "fog", "cloud"),
            links=("edge-fog", "fog-cloud"),
            updated_at=1,
        )
        driver = AgentLinuxNetworkControlDriver({"edge": edge})
        await driver.bind_route(binding, state)
        assert edge.calls == [
            (
                "bind",
                {
                    "destination": "10.0.0.9/32",
                    "via": "10.0.0.2",
                    "interface": "eth-route",
                },
            )
        ]
        await driver.clear_route("app", "source", "target", state)
        assert edge.calls[-1] == (
            "clear",
            {"destination": "10.0.0.9/32"},
        )

    asyncio.run(run())
