from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from darpan.core.event import EventKind
from darpan.core.resource import ResourceSpec
from darpan.core.topology import LinkSpec, NodeSpec, SystemSpec
from darpan.experiment.scenario_plan import ScenarioPlan, ScenarioPlanEvent
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.physical_control import AgentPhysicalControlDriver
from darpan.runtime.session import Session


@dataclass
class _Client:
    calls: list[tuple[str, object]] = field(default_factory=list)
    last_auto_restore: dict | None = None
    lease_active: bool = False
    renewal_failures_remaining: int = 0

    async def control_lease(self, *, lease_s: float = 60.0):
        if self.renewal_failures_remaining:
            self.renewal_failures_remaining -= 1
            raise ConnectionError("transient lease renewal failure")
        self.calls.append(("lease", lease_s))
        self.lease_active = True
        return {"lease_s": lease_s}

    async def control_inspect(self):
        return {
            "lease_active": self.lease_active,
            "last_auto_restore": self.last_auto_restore,
        }

    async def control_workload(self, *, enabled: bool, lease_s: float = 60.0):
        self.calls.append(("workload", enabled))
        return {"enabled": enabled}

    async def control_netem(self, **kwargs):
        self.calls.append(("netem", dict(kwargs)))
        return dict(kwargs)

    async def control_cpu_capacity(self, cpus: float, *, lease_s: float = 60.0):
        self.calls.append(("cpu", cpus))
        return {"cpus": cpus}

    async def control_restore(self):
        self.calls.append(("restore", True))
        self.lease_active = False
        return {"ok": True}


async def _run_physical_plan() -> tuple[Session, _Client, _Client]:
    edge = _Client()
    fog = _Client()
    driver = AgentPhysicalControlDriver({"edge": edge, "fog": fog})
    session = Session(RealBackend(physical_control_driver=driver))
    system = SystemSpec(
        nodes=(
            NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
            NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
        ),
        links=(
            LinkSpec(
                "edge-fog",
                "edge",
                "fog",
                latency_ms=5,
                bandwidth_mbps=100,
                labels={
                    "physical_control_scope": "interface",
                    "physical_source_interface": "eth-edge",
                    "physical_target_interface": "eth-fog",
                },
            ),
        ),
    )
    plan = ScenarioPlan(
        name="physical",
        events=(
            ScenarioPlanEvent(0, EventKind.NODE_OFFLINE, {"node_id": "edge"}),
            ScenarioPlanEvent(
                0,
                EventKind.LINK_CHANGED,
                {"link_id": "edge-fog", "latency_ms": 25, "bandwidth_mbps": 20},
            ),
            ScenarioPlanEvent(
                0,
                EventKind.MEASUREMENT_OBSERVED,
                {
                    "target": "fog",
                    "name": "compute.cpu_capacity",
                    "value": 0.5,
                    "unit": "count",
                },
            ),
        ),
    )
    await session.start()
    await session.register_system(system)
    await plan.apply(session)
    return session, edge, fog


def test_real_scenario_controls_physical_plane_before_canonical_state() -> None:
    async def run() -> None:
        session, edge, fog = await _run_physical_plan()
        assert session.state.nodes["edge"].status == "offline"
        assert session.state.links["edge-fog"].spec.latency_ms == 25
        assert session.state.links["edge-fog"].spec.bandwidth_mbps == 20
        assert session.state.nodes["fog"].measurements["compute.cpu_capacity"].value == 0.5
        assert edge.calls[0][0] == "lease"
        assert fog.calls[0][0] == "lease"
        assert ("workload", False) in edge.calls
        assert any(call[0] == "netem" for call in edge.calls)
        assert any(call[0] == "netem" for call in fog.calls)
        assert ("cpu", 0.5) in fog.calls
        await session.close()
        assert edge.calls[-1] == ("restore", True)
        assert fog.calls[-1] == ("restore", True)
        report = session.backend.physical_control_report
        assert report is not None
        assert report["restore_verified"] is True
        assert len(report["actions"]) == 4

    asyncio.run(run())


def test_physical_control_heartbeat_survives_transient_renewal_failure() -> None:
    async def run() -> None:
        client = _Client()
        driver = AgentPhysicalControlDriver({"fog": client}, lease_s=0.2)
        await driver.start()
        client.renewal_failures_remaining = 1
        await asyncio.sleep(0.16)
        report = driver.report()["heartbeat"]
        assert driver._heartbeat is not None
        assert not driver._heartbeat.done()
        assert report["renewal_successes"]["fog"] >= 2
        assert len(report["failures"]) == 1
        assert report["continuity_verified"] is True
        await driver.restore()
        assert driver.restore_verified is True

    asyncio.run(run())


def test_physical_control_reports_agent_auto_restore_as_continuity_failure() -> None:
    async def run() -> None:
        client = _Client()
        driver = AgentPhysicalControlDriver({"cloud": client}, lease_s=0.2)
        await driver.start()
        client.last_auto_restore = {"ok": True, "reason": "lease_expired"}
        await asyncio.sleep(0.08)
        await driver.restore()
        report = driver.report()["heartbeat"]
        assert report["continuity_verified"] is False
        assert any(
            row["reason"] == "new_agent_auto_restore" for row in report["continuity_violations"]
        )
        assert driver.restore_verified is False

    asyncio.run(run())
