from pathlib import Path

from darpan.runtime.real.telemetry import _current_cgroup_root, detect_cpu_capacity


def test_detect_cpu_capacity_respects_cgroup_v2_quota(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    assert detect_cpu_capacity(cgroup_root=tmp_path, cpu_count=8) == 1.5


def test_detect_cpu_capacity_falls_back_to_host_cpu_count(tmp_path: Path) -> None:
    assert detect_cpu_capacity(cgroup_root=tmp_path, cpu_count=4) == 4.0


def test_current_cgroup_root_resolves_unified_process_path(tmp_path: Path) -> None:
    cgroup_fs = tmp_path / "cgroup"
    service = cgroup_fs / "system.slice" / "darpan-agent.service"
    service.mkdir(parents=True)
    proc_self = tmp_path / "self.cgroup"
    proc_self.write_text(
        "0::/system.slice/darpan-agent.service\n",
        encoding="utf-8",
    )

    assert (
        _current_cgroup_root(
            cgroup_fs_root=cgroup_fs,
            proc_self_cgroup=proc_self,
        )
        == service
    )
