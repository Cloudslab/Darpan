"""Telemetry collection helpers for physical nodes."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from darpan.core.event import Event, EventKind
from darpan.core.measurement import Measurement
from darpan.core.serialization import to_primitive


def _current_cgroup_root(
    *,
    cgroup_fs_root: Path = Path("/sys/fs/cgroup"),
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
) -> Path:
    """Resolve this process' cgroup-v2 directory, falling back to the mount root."""

    try:
        lines = proc_self_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return cgroup_fs_root
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0" or parts[1] != "":
            continue
        relative = parts[2].strip().lstrip("/")
        candidate = cgroup_fs_root / relative
        if candidate.exists():
            return candidate
    return cgroup_fs_root


def measurement_event(measurement: Measurement, *, source: str) -> Event:
    return Event(
        kind=EventKind.MEASUREMENT_OBSERVED,
        event_time=measurement.timestamp,
        source=source,
        subject=measurement.target,
        payload={"measurement": to_primitive(measurement)},
    )


def detect_cpu_capacity(
    *,
    cgroup_root: Path | None = None,
    cpu_count: int | None = None,
) -> float:
    """Return CPU capacity visible to an Agent, respecting common cgroup quotas.

    The value is a capacity signal, not instantaneous CPU availability. Darpan
    later clamps it to the node capacity declared by the experiment, so host
    telemetry can reduce but never silently expand the user's resource model.
    """

    cgroup_root = _current_cgroup_root() if cgroup_root is None else cgroup_root
    host_cpus = float(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    host_cpus = max(host_cpus, 1e-6)

    cpu_max = cgroup_root / "cpu.max"
    try:
        quota_text, period_text = cpu_max.read_text().strip().split()[:2]
        if quota_text != "max":
            quota = float(quota_text)
            period = float(period_text)
            if quota > 0 and period > 0:
                return min(host_cpus, max(1e-6, quota / period))
    except (OSError, ValueError, IndexError):
        pass

    quota_path = cgroup_root / "cpu" / "cpu.cfs_quota_us"
    period_path = cgroup_root / "cpu" / "cpu.cfs_period_us"
    try:
        quota = float(quota_path.read_text().strip())
        period = float(period_path.read_text().strip())
        if quota > 0 and period > 0:
            return min(host_cpus, max(1e-6, quota / period))
    except (OSError, ValueError):
        pass
    return host_cpus


def detect_memory_capacity_bytes(
    *,
    cgroup_root: Path | None = None,
    host_bytes: int | None = None,
) -> int | None:
    """Return host-visible memory capacity, clamped by common cgroup limits."""

    cgroup_root = _current_cgroup_root() if cgroup_root is None else cgroup_root
    if host_bytes is None:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            host_bytes = pages * page_size if pages > 0 and page_size > 0 else None
        except (AttributeError, OSError, TypeError, ValueError):
            host_bytes = None

    limits: list[int] = []
    memory_max = cgroup_root / "memory.max"
    try:
        text = memory_max.read_text().strip()
        if text and text != "max":
            value = int(text)
            if value > 0:
                limits.append(value)
    except (OSError, ValueError):
        pass

    memory_limit = cgroup_root / "memory" / "memory.limit_in_bytes"
    try:
        value = int(memory_limit.read_text().strip())
        if value > 0:
            limits.append(value)
    except (OSError, ValueError):
        pass

    if host_bytes is not None and host_bytes > 0:
        limits.append(int(host_bytes))
    return min(limits) if limits else None


def detect_cgroup_version(*, cgroup_root: Path | None = None) -> str:
    cgroup_root = _current_cgroup_root() if cgroup_root is None else cgroup_root
    if (cgroup_root / "cgroup.controllers").is_file():
        return "v2"
    if (cgroup_root / "cpu").exists() or (cgroup_root / "memory").exists():
        return "v1"
    return "none"


def environment_snapshot(
    *,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    """Stable, non-secret physical environment metadata for paper provenance."""

    cgroup_root = _current_cgroup_root() if cgroup_root is None else cgroup_root
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "cpu_capacity": detect_cpu_capacity(cgroup_root=cgroup_root),
        "memory_capacity_bytes": detect_memory_capacity_bytes(cgroup_root=cgroup_root),
        "cgroup_version": detect_cgroup_version(cgroup_root=cgroup_root),
        "container_hint": bool(
            Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
        ),
    }
