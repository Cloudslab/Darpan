"""Opt-in Linux control plane used by Darpan physical experiments.

The implementation intentionally exposes a small, validated command surface.
It never accepts arbitrary shell commands from the controller.  Mutations are
tracked and restored so a failed experiment does not silently leave tc, route,
or cgroup state behind.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

CommandRunner = Callable[[tuple[str, ...]], Awaitable[tuple[int, str, str]]]


async def _default_runner(command: tuple[str, ...]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (
        int(process.returncode or 0),
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


@dataclass(frozen=True, slots=True)
class LinuxControlCapabilities:
    netem: bool
    route: bool
    cpu_capacity: bool
    interfaces: tuple[str, ...]
    cpu_max_path: str | None


class LinuxPhysicalControlBackend:
    """Restricted Linux tc/ip/cgroup mutations with deterministic restoration."""

    def __init__(
        self,
        *,
        interfaces: Iterable[str] = (),
        cpu_max_path: str | Path | None = None,
        runner: CommandRunner | None = None,
        tc_path: str | None = None,
        ip_path: str | None = None,
        allow_replace_existing_qdisc: bool = False,
    ) -> None:
        self.interfaces = frozenset(str(item) for item in interfaces if str(item))
        self.cpu_max_path = (
            None if cpu_max_path is None else Path(cpu_max_path).expanduser().resolve()
        )
        self._runner = runner or _default_runner
        self.tc_path = tc_path or shutil.which("tc")
        self.ip_path = ip_path or shutil.which("ip")
        self.allow_replace_existing_qdisc = allow_replace_existing_qdisc
        self._qdisc_original: dict[str, tuple[dict[str, object], ...]] = {}
        self._route_original: dict[str, tuple[dict[str, object], ...]] = {}
        self._cpu_original: str | None = None
        self._actions: list[dict[str, object]] = []

    def capabilities(self) -> LinuxControlCapabilities:
        return LinuxControlCapabilities(
            netem=self.tc_path is not None and bool(self.interfaces),
            route=self.ip_path is not None and bool(self.interfaces),
            cpu_capacity=self.cpu_max_path is not None,
            interfaces=tuple(sorted(self.interfaces)),
            cpu_max_path=(
                None if self.cpu_max_path is None else str(self.cpu_max_path)
            ),
        )

    def readiness(self) -> dict[str, object]:
        """Report whether configured Linux control targets exist and are writable."""

        interface_status = {
            interface: Path("/sys/class/net", interface).exists()
            for interface in sorted(self.interfaces)
        }
        cpu_exists = self.cpu_max_path is not None and self.cpu_max_path.exists()
        cpu_writable = bool(
            cpu_exists
            and self.cpu_max_path is not None
            and os.access(self.cpu_max_path, os.W_OK)
        )
        return {
            "tc_available": self.tc_path is not None,
            "ip_available": self.ip_path is not None,
            "interfaces": interface_status,
            "cpu_max_exists": bool(cpu_exists),
            "cpu_max_writable": cpu_writable,
            "netem_ready": bool(self.tc_path)
            and bool(interface_status)
            and all(interface_status.values()),
            "route_ready": bool(self.ip_path)
            and bool(interface_status)
            and all(interface_status.values()),
            "cpu_capacity_ready": bool(cpu_exists and cpu_writable),
        }

    async def _command(self, *parts: str, allow_missing: bool = False) -> str:
        code, stdout, stderr = await self._runner(tuple(parts))
        if code != 0 and not allow_missing:
            rendered = " ".join(parts)
            raise RuntimeError(
                f"Linux physical-control command failed ({code}): {rendered}: "
                f"{stderr.strip()}"
            )
        return stdout

    def _require_interface(self, interface: str) -> str:
        interface = str(interface).strip()
        if not interface or interface not in self.interfaces:
            raise PermissionError(
                f"interface {interface!r} is not in the Agent physical-control allow-list"
            )
        return interface

    async def _qdisc_snapshot(self, interface: str) -> tuple[dict[str, object], ...]:
        if self.tc_path is None:
            raise RuntimeError("Linux tc is unavailable")
        raw = await self._command(self.tc_path, "-j", "qdisc", "show", "dev", interface)
        payload = json.loads(raw or "[]")
        if not isinstance(payload, list):
            raise RuntimeError("tc qdisc snapshot did not return a JSON list")
        return tuple(dict(item) for item in payload if isinstance(item, dict))

    @staticmethod
    def _has_nontrivial_root_qdisc(snapshot: tuple[dict[str, object], ...]) -> bool:
        for item in snapshot:
            if str(item.get("parent", "")) == "root" or bool(item.get("root", False)):
                kind = str(item.get("kind", ""))
                if kind not in {"", "noqueue", "pfifo_fast", "fq_codel"}:
                    return True
        return False

    async def apply_netem(
        self,
        interface: str,
        *,
        latency_ms: float | None = None,
        bandwidth_mbps: float | None = None,
        loss_pct: float | None = None,
    ) -> dict[str, object]:
        interface = self._require_interface(interface)
        if self.tc_path is None:
            raise RuntimeError("Linux tc is unavailable")
        if latency_ms is not None and latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if bandwidth_mbps is not None and bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be positive")
        if loss_pct is not None and not 0 <= loss_pct <= 100:
            raise ValueError("loss_pct must be in [0, 100]")
        if interface not in self._qdisc_original:
            snapshot = await self._qdisc_snapshot(interface)
            if self._has_nontrivial_root_qdisc(snapshot) and not self.allow_replace_existing_qdisc:
                raise RuntimeError(
                    f"refusing to replace existing non-trivial qdisc on {interface}; "
                    "use an isolated experiment interface or opt in explicitly"
                )
            self._qdisc_original[interface] = snapshot
        command = [self.tc_path, "qdisc", "replace", "dev", interface, "root", "netem"]
        if latency_ms is not None:
            command += ["delay", f"{float(latency_ms):.6f}ms"]
        if bandwidth_mbps is not None:
            command += ["rate", f"{float(bandwidth_mbps):.6f}mbit"]
        if loss_pct is not None:
            command += ["loss", f"{float(loss_pct):.6f}%"]
        if len(command) == 7:
            raise ValueError("netem mutation must set latency, bandwidth, or loss")
        await self._command(*command)
        observed = await self._qdisc_snapshot(interface)
        if not any(str(item.get("kind", "")) == "netem" for item in observed):
            raise RuntimeError(
                f"Linux tc did not report the requested netem qdisc on {interface}"
            )
        action = {
            "kind": "netem.apply",
            "interface": interface,
            "latency_ms": latency_ms,
            "bandwidth_mbps": bandwidth_mbps,
            "loss_pct": loss_pct,
            "observed_qdisc": [dict(item) for item in observed],
        }
        self._actions.append(action)
        return action

    async def clear_netem(self, interface: str) -> dict[str, object]:
        interface = self._require_interface(interface)
        if self.tc_path is None:
            raise RuntimeError("Linux tc is unavailable")
        if interface not in self._qdisc_original:
            self._qdisc_original[interface] = await self._qdisc_snapshot(interface)
        await self._command(
            self.tc_path,
            "qdisc",
            "del",
            "dev",
            interface,
            "root",
            allow_missing=True,
        )
        action = {"kind": "netem.clear", "interface": interface}
        self._actions.append(action)
        return action

    async def _route_snapshot(self, destination: str) -> tuple[dict[str, object], ...]:
        if self.ip_path is None:
            raise RuntimeError("Linux ip is unavailable")
        raw = await self._command(self.ip_path, "-j", "route", "show", "exact", destination)
        payload = json.loads(raw or "[]")
        if not isinstance(payload, list):
            raise RuntimeError("ip route snapshot did not return a JSON list")
        return tuple(dict(item) for item in payload if isinstance(item, dict))

    async def bind_route(
        self,
        *,
        destination: str,
        via: str,
        interface: str,
    ) -> dict[str, object]:
        interface = self._require_interface(interface)
        if self.ip_path is None:
            raise RuntimeError("Linux ip is unavailable")
        if destination not in self._route_original:
            self._route_original[destination] = await self._route_snapshot(destination)
        await self._command(
            self.ip_path,
            "route",
            "replace",
            destination,
            "via",
            via,
            "dev",
            interface,
        )
        action = {
            "kind": "route.bind",
            "destination": destination,
            "via": via,
            "interface": interface,
        }
        self._actions.append(action)
        return action

    async def clear_route(self, *, destination: str) -> dict[str, object]:
        if self.ip_path is None:
            raise RuntimeError("Linux ip is unavailable")
        if destination not in self._route_original:
            self._route_original[destination] = await self._route_snapshot(destination)
        await self._command(
            self.ip_path,
            "route",
            "del",
            destination,
            allow_missing=True,
        )
        action = {"kind": "route.clear", "destination": destination}
        self._actions.append(action)
        return action

    async def set_cpu_capacity(self, cpus: float) -> dict[str, object]:
        if cpus <= 0:
            raise ValueError("CPU capacity must be positive")
        if self.cpu_max_path is None:
            raise RuntimeError("Agent has no physical-control cpu.max path")
        if self._cpu_original is None:
            self._cpu_original = self.cpu_max_path.read_text(encoding="utf-8").strip()
        period = 100000
        quota = max(1, int(round(float(cpus) * period)))
        rendered = f"{quota} {period}"
        self.cpu_max_path.write_text(rendered + "\n", encoding="utf-8")
        observed = self.cpu_max_path.read_text(encoding="utf-8").strip()
        if observed != rendered:
            raise RuntimeError(
                "cpu.max readback differs from the requested CPU-capacity control"
            )
        action = {
            "kind": "cpu.capacity",
            "cpus": float(cpus),
            "cpu_max": rendered,
            "observed_cpu_max": observed,
        }
        self._actions.append(action)
        return action

    async def restore(self) -> dict[str, object]:
        errors: list[str] = []
        restored: list[str] = []
        if self.tc_path is not None:
            for interface, original in reversed(tuple(self._qdisc_original.items())):
                try:
                    await self._command(
                        self.tc_path,
                        "qdisc",
                        "del",
                        "dev",
                        interface,
                        "root",
                        allow_missing=True,
                    )
                    # Arbitrary qdisc trees cannot be reconstructed safely from
                    # tc JSON. We only mutate non-trivial roots when explicitly
                    # opted in; in that exceptional mode restoration cannot be
                    # claimed as verified.
                    if self._has_nontrivial_root_qdisc(original):
                        raise RuntimeError(
                            "original non-trivial qdisc cannot be reconstructed automatically"
                        )
                    restored.append(f"qdisc:{interface}")
                except Exception as exc:
                    errors.append(f"qdisc:{interface}: {type(exc).__name__}: {exc}")
        if self.ip_path is not None:
            for destination, original in reversed(tuple(self._route_original.items())):
                try:
                    await self._command(
                        self.ip_path,
                        "route",
                        "del",
                        destination,
                        allow_missing=True,
                    )
                    if original:
                        first = original[0]
                        command = [self.ip_path, "route", "replace", destination]
                        gateway = first.get("gateway")
                        dev = first.get("dev")
                        metric = first.get("metric")
                        if gateway:
                            command += ["via", str(gateway)]
                        if dev:
                            command += ["dev", str(dev)]
                        if metric is not None:
                            command += ["metric", str(metric)]
                        await self._command(*command)
                    restored.append(f"route:{destination}")
                except Exception as exc:
                    errors.append(f"route:{destination}: {type(exc).__name__}: {exc}")
        if self._cpu_original is not None and self.cpu_max_path is not None:
            try:
                self.cpu_max_path.write_text(self._cpu_original + "\n", encoding="utf-8")
                restored.append("cpu.max")
            except Exception as exc:
                errors.append(f"cpu.max: {type(exc).__name__}: {exc}")
        return {"ok": not errors, "restored": restored, "errors": errors}

    async def verify_restored(self) -> dict[str, object]:
        mismatches: list[str] = []
        for interface, original in self._qdisc_original.items():
            if self._has_nontrivial_root_qdisc(original):
                mismatches.append(f"qdisc:{interface}: restoration not verifiable")
                continue
            current = await self._qdisc_snapshot(interface)
            if any(str(item.get("kind", "")) == "netem" for item in current):
                mismatches.append(f"qdisc:{interface}: netem still active")
        for destination, original in self._route_original.items():
            current = await self._route_snapshot(destination)
            if current != original:
                mismatches.append(f"route:{destination}: route differs from snapshot")
        if self._cpu_original is not None and self.cpu_max_path is not None:
            current = self.cpu_max_path.read_text(encoding="utf-8").strip()
            if current != self._cpu_original:
                mismatches.append("cpu.max differs from snapshot")
        return {"ok": not mismatches, "mismatches": mismatches}

    def report(self) -> dict[str, object]:
        return {
            "schema": "darpan.linux-physical-control/v1",
            "capabilities": {
                "netem": self.capabilities().netem,
                "route": self.capabilities().route,
                "cpu_capacity": self.capabilities().cpu_capacity,
                "interfaces": list(self.capabilities().interfaces),
                "cpu_max_path": self.capabilities().cpu_max_path,
            },
            "actions": list(self._actions),
            "captured": {
                "qdisc_interfaces": sorted(self._qdisc_original),
                "route_destinations": sorted(self._route_original),
                "cpu_max": self._cpu_original is not None,
            },
        }
