from __future__ import annotations

from darpan.runtime.real.telemetry import (
    detect_cgroup_version,
    detect_memory_capacity_bytes,
    environment_snapshot,
)


def test_memory_capacity_respects_cgroup_v2_limit(tmp_path):
    (tmp_path / "memory.max").write_text("536870912\n", encoding="utf-8")
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n", encoding="utf-8")

    assert detect_memory_capacity_bytes(
        cgroup_root=tmp_path,
        host_bytes=2 * 1024**3,
    ) == 512 * 1024**2
    assert detect_cgroup_version(cgroup_root=tmp_path) == "v2"


def test_memory_capacity_ignores_unlimited_cgroup_v2(tmp_path):
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    assert detect_memory_capacity_bytes(
        cgroup_root=tmp_path,
        host_bytes=1024,
    ) == 1024


def test_environment_snapshot_has_reproducibility_fields():
    payload = environment_snapshot()

    assert payload["system"]
    assert payload["machine"]
    assert payload["python_version"]
    assert float(payload["cpu_capacity"]) > 0
    assert payload["cgroup_version"] in {"none", "v1", "v2"}
