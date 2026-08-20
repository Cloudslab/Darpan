"""Exercise Darpan's stable six-action controller surface on the Twin runtime."""

from __future__ import annotations

import asyncio
import json

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    Darpan,
    FlowSpec,
    LinkSpec,
    NodeSpec,
    ResourceRequest,
    ResourceSpec,
    SystemSpec,
)
from darpan.core.action import ActionKind
from darpan.core.event import EventKind


def _system() -> SystemSpec:
    nodes = tuple(
        NodeSpec(name, resources=(ResourceSpec("cpu", 4),))
        for name in ("edge", "fog-a", "fog-b", "cloud")
    )
    links = (
        LinkSpec("edge-a", "edge", "fog-a", latency_ms=1, bandwidth_mbps=100),
        LinkSpec("a-cloud", "fog-a", "cloud", latency_ms=1, bandwidth_mbps=100),
        LinkSpec("edge-b", "edge", "fog-b", latency_ms=5, bandwidth_mbps=50),
        LinkSpec("b-cloud", "fog-b", "cloud", latency_ms=5, bandwidth_mbps=50),
    )
    return SystemSpec(nodes=nodes, links=links)


def _application() -> ApplicationSpec:
    return ApplicationSpec(
        "control-surface",
        components=(
            ComponentSpec(
                "api",
                kind="service",
                work_units=0,
                resources=(ResourceRequest("cpu", 1),),
            ),
            ComponentSpec(
                "sink",
                kind="stream",
                work_units=0,
                resources=(ResourceRequest("cpu", 1),),
            ),
        ),
        flows=(FlowSpec("api", "sink", kind="stream"),),
    )


async def _run() -> dict[str, object]:
    session = Darpan.twin()
    await session.start()
    try:
        await session.register_system(_system())
        app_instance = await session.submit_application(_application())
        api = f"{app_instance}:api"
        sink = f"{app_instance}:sink"

        assert await session.apply(Action.place(api, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))

        assert await session.apply(
            Action.route(app_instance, "api", "sink", ["edge", "fog-a", "cloud"])
        )
        assert await session.apply(Action.route(app_instance, "api", "sink", []))

        assert await session.apply(Action.scale(api, 2))
        replica = f"{api}#replica-1"
        assert await session.apply(Action.place(replica, "fog-a"))
        await session.wait_for(
            lambda event, state: (
                event.kind == EventKind.COMPONENT_SCALED
                and event.payload.get("desired_replicas") == 2
            ),
            timeout=2,
        )
        assert await session.apply(Action.scale(api, 1))

        assert await session.apply(Action.restart(api))
        assert await session.apply(Action.migrate(api, "fog-b"))
        assert await session.apply(
            Action.route(app_instance, "api", "sink", ["fog-b", "cloud"])
        )

        assert await session.apply(Action.stop(api))
        assert await session.apply(Action.stop(sink))

        completion = next(
            event
            for event in reversed(tuple(session.event_log))
            if event.kind == EventKind.APPLICATION_COMPLETED
        )
        stable_actions = sorted(
            {
                ActionKind.PLACE,
                ActionKind.STOP,
                ActionKind.RESTART,
                ActionKind.MIGRATE,
                ActionKind.SCALE,
                ActionKind.ROUTE,
            }
        )
        return {
            "stable_actions": stable_actions,
            "twin_supported_actions": sorted(session.supported_action_kinds or ()),
            "application_success": bool(completion.payload.get("success")),
            "final_api_node": session.state.components[api].node_id,
            "api_attempt": session.state.components[api].attempt,
            "desired_replicas": session.state.desired_replicas(api),
            "route_path": list(
                session.state.flow_route(app_instance, "api", "sink").path
            ),
        }
    finally:
        await session.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run()), sort_keys=True))
