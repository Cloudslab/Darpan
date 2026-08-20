from __future__ import annotations

import asyncio

import pytest

from darpan import (
    Action,
    ApplicationSpec,
    ComponentSpec,
    Darpan,
    DigitalTwin,
    FlowSpec,
    LinkSpec,
    NodeSpec,
    ResourceRequest,
    ResourceSpec,
    SystemSpec,
)
from darpan.core.action import ActionKind
from darpan.core.event import Event, EventKind
from darpan.runtime.real.backend import RealBackend
from darpan.twin.snapshot import TwinSnapshot


def _routed_system() -> SystemSpec:
    return SystemSpec(
        nodes=tuple(
            NodeSpec(name, resources=(ResourceSpec("cpu", 4),))
            for name in ("edge", "fog-a", "fog-b", "cloud")
        ),
        links=(
            LinkSpec("edge-a", "edge", "fog-a", latency_ms=1, bandwidth_mbps=100),
            LinkSpec("a-cloud", "fog-a", "cloud", latency_ms=1, bandwidth_mbps=100),
            LinkSpec("edge-b", "edge", "fog-b", latency_ms=10, bandwidth_mbps=10),
            LinkSpec("b-cloud", "fog-b", "cloud", latency_ms=10, bandwidth_mbps=10),
        ),
    )


def _artifact_app() -> ApplicationSpec:
    return ApplicationSpec(
        "route-artifact",
        components=(
            ComponentSpec("source", work_units=0.1, resources=(ResourceRequest("cpu", 1),)),
            ComponentSpec("sink", work_units=0.1, resources=(ResourceRequest("cpu", 1),)),
        ),
        flows=(
            FlowSpec(
                "source",
                "sink",
                data_size_bytes=1_000_000,
                artifact="payload.bin",
            ),
        ),
    )


def test_twin_route_binding_drives_future_transfer_path_and_placement() -> None:
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(_artifact_app())
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert session.state.components[source].status == "completed"
        assert session.state.components[sink].status == "ready"

        route = Action.route(
            app_instance,
            "source",
            "sink",
            ["edge", "fog-b", "cloud"],
        )
        assert await session.apply(route)
        binding = session.state.flow_route(app_instance, "source", "sink")
        assert binding is not None
        assert binding.path == ("edge", "fog-b", "cloud")
        assert binding.links == ("edge-b", "b-cloud")

        assert not await session.apply(Action.place(sink, "fog-a"))
        assert await session.apply(Action.place(sink, "cloud"))
        transfer = next(
            event
            for event in session.event_log
            if event.kind == EventKind.DATA_TRANSFER_STARTED
            and event.payload.get("target_instance_id") == sink
        )
        assert transfer.payload["route_bound"] is True
        assert transfer.payload["path"] == ["edge", "fog-b", "cloud"]
        assert transfer.payload["links"] == ["edge-b", "b-cloud"]

        assert await session.apply(Action.route(app_instance, "source", "sink", []))
        assert session.state.flow_route(app_instance, "source", "sink") is None
        await session.close()

    asyncio.run(run())


class _RecordingNetworkDriver:
    def __init__(self) -> None:
        self.bound = []
        self.cleared = []

    async def bind_route(self, binding, state) -> None:
        del state
        self.bound.append(binding)

    async def clear_route(self, app_instance, source, target, state) -> None:
        del state
        self.cleared.append((app_instance, source, target))


def test_real_route_requires_and_invokes_physical_network_driver() -> None:
    async def run() -> None:
        app = ApplicationSpec(
            "route-service",
            components=(
                ComponentSpec("source", kind="service", work_units=0),
                ComponentSpec("sink", kind="stream", work_units=0),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        driver = _RecordingNetworkDriver()
        session = Darpan.real(backend=RealBackend(network_driver=driver))
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(app)
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))

        assert await session.apply(
            Action.route(app_instance, "source", "sink", ["edge", "fog-a", "cloud"])
        )
        assert len(driver.bound) == 1
        assert driver.bound[0].links == ("edge-a", "a-cloud")
        assert session.state.flow_route(app_instance, "source", "sink") is not None

        assert await session.apply(Action.route(app_instance, "source", "sink", []))
        assert driver.cleared == [(app_instance, "source", "sink")]
        assert session.state.flow_route(app_instance, "source", "sink") is None
        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())


def test_route_validation_rejects_bad_or_ambiguous_paths_and_scaled_endpoints() -> None:
    async def run() -> None:
        app = ApplicationSpec(
            "route-service",
            components=(
                ComponentSpec("source", kind="service", work_units=0),
                ComponentSpec("sink", kind="stream", work_units=0),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(app)
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))

        assert not await session.apply(
            Action.route(app_instance, "source", "sink", ["fog-a", "cloud"])
        )
        assert not await session.apply(
            Action.route(app_instance, "source", "missing", ["edge", "fog-a"])
        )
        assert await session.apply(Action.scale(source, 2))
        replica = f"{source}#replica-1"
        assert await session.apply(Action.place(replica, "edge"))
        assert not await session.apply(
            Action.route(app_instance, "source", "sink", ["edge", "fog-a", "cloud"])
        )

        reasons = [
            str(event.payload.get("reason"))
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.ROUTE
        ]
        assert any("must start at source node" in reason for reason in reasons)
        assert any("has no flow" in reason for reason in reasons)
        assert any("unscaled endpoints" in reason for reason in reasons)
        assert await session.apply(Action.scale(source, 1))
        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())


def test_route_binding_survives_snapshot_and_broken_route_blocks_future_placement() -> None:
    async def run() -> None:
        session = Darpan.twin()
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(_artifact_app())
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(
            Action.route(app_instance, "source", "sink", ["edge", "fog-b", "cloud"])
        )
        snapshot = DigitalTwin().snapshot(session.state)
        restored = TwinSnapshot.from_dict(snapshot.to_dict())
        binding = restored.state.flow_route(app_instance, "source", "sink")
        assert binding is not None and binding.links == ("edge-b", "b-cloud")

        await session.emit(
            Event(
                kind=EventKind.LINK_REMOVED,
                event_time=session.clock.now(),
                source="test",
                subject="edge-b",
                payload={"link_id": "edge-b"},
            )
        )
        assert not await session.apply(Action.place(sink, "cloud"))
        rejection = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_REJECTED
            and event.payload.get("kind") == ActionKind.PLACE
        ][-1]
        assert "explicit route is unavailable" in str(rejection.payload["reason"])
        await session.close()

    asyncio.run(run())


class _FailingNetworkDriver:
    async def bind_route(self, binding, state) -> None:
        del binding, state
        raise RuntimeError("network control failed")

    async def clear_route(self, app_instance, source, target, state) -> None:
        del app_instance, source, target, state
        raise RuntimeError("network control failed")


def test_real_route_driver_failure_does_not_commit_canonical_binding() -> None:
    async def run() -> None:
        app = ApplicationSpec(
            "route-service",
            components=(
                ComponentSpec("source", kind="service", work_units=0),
                ComponentSpec("sink", kind="stream", work_units=0),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        session = Darpan.real(
            backend=RealBackend(network_driver=_FailingNetworkDriver())
        )
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(app)
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))

        with pytest.raises(RuntimeError, match="network control failed"):
            await session.apply(
                Action.route(
                    app_instance,
                    "source",
                    "sink",
                    ["edge", "fog-a", "cloud"],
                )
            )
        assert session.state.flow_route(app_instance, "source", "sink") is None
        failed = [
            event
            for event in session.event_log
            if event.kind == EventKind.ACTION_FAILED
            and event.payload.get("kind") == ActionKind.ROUTE
        ]
        assert len(failed) == 1
        await session.close()

    asyncio.run(run())


def test_explicit_route_must_be_cleared_before_scaling_endpoint() -> None:
    async def run() -> None:
        app = ApplicationSpec(
            "route-scale",
            components=(
                ComponentSpec("source", kind="service", work_units=0),
                ComponentSpec("sink", kind="stream", work_units=0),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(app)
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))
        assert await session.apply(
            Action.route(
                app_instance,
                "source",
                "sink",
                ["edge", "fog-a", "cloud"],
            )
        )
        assert not await session.apply(Action.scale(source, 2))
        assert await session.apply(Action.route(app_instance, "source", "sink", []))
        assert await session.apply(Action.scale(source, 2))
        replica = f"{source}#replica-1"
        assert await session.apply(Action.place(replica, "edge"))
        assert await session.apply(Action.scale(source, 1))
        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())


def test_route_can_fail_over_to_alternate_path_after_link_loss() -> None:
    async def run() -> None:
        app = ApplicationSpec(
            "route-failover",
            components=(
                ComponentSpec("source", kind="service", work_units=0),
                ComponentSpec("sink", kind="stream", work_units=0),
            ),
            flows=(FlowSpec("source", "sink", kind="stream"),),
        )
        session = Darpan.twin()
        await session.start()
        await session.register_system(_routed_system())
        app_instance = await session.submit_application(app)
        source = f"{app_instance}:source"
        sink = f"{app_instance}:sink"
        assert await session.apply(Action.place(source, "edge"))
        assert await session.apply(Action.place(sink, "cloud"))
        assert await session.apply(
            Action.route(
                app_instance,
                "source",
                "sink",
                ["edge", "fog-a", "cloud"],
            )
        )
        await session.emit(
            Event(
                kind=EventKind.LINK_REMOVED,
                event_time=session.clock.now(),
                source="test",
                subject="a-cloud",
                payload={"link_id": "a-cloud"},
            )
        )
        assert await session.apply(
            Action.route(
                app_instance,
                "source",
                "sink",
                ["edge", "fog-b", "cloud"],
            )
        )
        binding = session.state.flow_route(app_instance, "source", "sink")
        assert binding is not None
        assert binding.links == ("edge-b", "b-cloud")
        assert await session.apply(Action.stop(source))
        assert await session.apply(Action.stop(sink))
        await session.close()

    asyncio.run(run())
