from __future__ import annotations

import pytest

from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.codec import load_workload
from darpan.core.workload import ArrivalSpec, WorkloadSpec


def _app(app_id: str) -> ApplicationSpec:
    return ApplicationSpec(app_id, (ComponentSpec("task"),))


def test_workload_fails_fast_for_empty_duplicate_and_unknown_arrivals():
    with pytest.raises(ValueError, match="at least one application"):
        WorkloadSpec((), (ArrivalSpec("missing"),))

    with pytest.raises(ValueError, match="duplicate application"):
        WorkloadSpec((_app("a"), _app("a")), (ArrivalSpec("a"),))

    with pytest.raises(ValueError, match="at least one arrival"):
        WorkloadSpec((_app("a"),), ())

    with pytest.raises(ValueError, match="unknown applications: missing"):
        WorkloadSpec((_app("a"),), (ArrivalSpec("missing"),))


def test_workload_codec_resolves_application_files_relative_to_workload(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("id: demo\ncomponents:\n  task: {}\n", encoding="utf-8")
    workload = tmp_path / "workload.yaml"
    workload.write_text(
        "name: demo-load\n"
        "applications:\n"
        "  - app.yaml\n"
        "arrivals:\n"
        "  - application: demo\n"
        "    at: 1.5\n"
        "    count: 2\n",
        encoding="utf-8",
    )

    loaded = load_workload(workload)
    assert loaded.name == "demo-load"
    assert loaded.application("demo").component("task").id == "task"
    assert loaded.arrivals[0] == ArrivalSpec("demo", at_s=1.5, count=2)
