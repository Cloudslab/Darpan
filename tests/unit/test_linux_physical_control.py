from __future__ import annotations

import asyncio
from pathlib import Path

from darpan.runtime.real.linux_control import LinuxPhysicalControlBackend


def test_linux_control_is_allowlisted_and_restores(tmp_path: Path) -> None:
    async def run() -> None:
        commands: list[tuple[str, ...]] = []
        netem_active = False

        async def runner(command: tuple[str, ...]):
            nonlocal netem_active
            commands.append(command)
            if "-j" in command and "qdisc" in command:
                payload = '[{"kind":"netem","root":true}]' if netem_active else "[]"
                return 0, payload, ""
            if "-j" in command and "route" in command:
                return 0, "[]", ""
            if command[:4] == ("tc", "qdisc", "replace", "dev"):
                netem_active = True
            if command[:4] == ("tc", "qdisc", "del", "dev"):
                netem_active = False
            return 0, "", ""

        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("max 100000\n", encoding="utf-8")
        control = LinuxPhysicalControlBackend(
            interfaces=("eth-test",),
            cpu_max_path=cpu_max,
            runner=runner,
            tc_path="tc",
            ip_path="ip",
        )
        applied = await control.apply_netem(
            "eth-test",
            latency_ms=20,
            bandwidth_mbps=50,
        )
        assert applied["observed_qdisc"] == [{"kind": "netem", "root": True}]
        await control.bind_route(
            destination="10.0.0.9/32",
            via="10.0.0.2",
            interface="eth-test",
        )
        cpu_applied = await control.set_cpu_capacity(0.5)
        assert cpu_applied["observed_cpu_max"] == "50000 100000"
        assert cpu_max.read_text(encoding="utf-8").strip() == "50000 100000"

        restored = await control.restore()
        verified = await control.verify_restored()
        assert restored["ok"] is True
        assert verified["ok"] is True
        assert cpu_max.read_text(encoding="utf-8").strip() == "max 100000"
        assert any(command[:4] == ("tc", "qdisc", "replace", "dev") for command in commands)
        assert any(command[:3] == ("ip", "route", "replace") for command in commands)

    asyncio.run(run())


def test_linux_control_rejects_unlisted_interface() -> None:
    async def run() -> None:
        control = LinuxPhysicalControlBackend(
            interfaces=("eth-safe",),
            runner=lambda command: None,  # never reached
            tc_path="tc",
        )
        try:
            await control.apply_netem("eth0", latency_ms=10)
        except PermissionError as exc:
            assert "allow-list" in str(exc)
        else:
            raise AssertionError("unlisted interface should be rejected")

    asyncio.run(run())


def test_linux_control_readiness_checks_real_targets(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000\n", encoding="utf-8")
    ready = LinuxPhysicalControlBackend(
        interfaces=("lo",),
        cpu_max_path=cpu_max,
        tc_path="tc",
        ip_path="ip",
    ).readiness()
    assert ready["interfaces"] == {"lo": True}
    assert ready["netem_ready"] is True
    assert ready["route_ready"] is True
    assert ready["cpu_capacity_ready"] is True

    missing = LinuxPhysicalControlBackend(
        interfaces=("darpan-interface-that-does-not-exist",),
        tc_path="tc",
        ip_path="ip",
    ).readiness()
    assert missing["netem_ready"] is False
    assert missing["route_ready"] is False
