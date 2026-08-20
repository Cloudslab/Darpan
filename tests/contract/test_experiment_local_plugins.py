from __future__ import annotations

import json

from darpan.cli.main import main
from darpan.core.action import ActionKind
from darpan.core.loading import load_symbol
from darpan.experiment.spec import ExperimentSpec


def test_local_plugin_file_is_loaded_without_packaging(tmp_path):
    plugin = tmp_path / "my_policy.py"
    plugin.write_text(
        "class Marker:\n"
        "    value = 42\n",
        encoding="utf-8",
    )
    marker = load_symbol("my_policy.py:Marker", search_path=tmp_path)
    assert marker.value == 42


def test_experiment_paths_are_relative_to_experiment_file(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task:\n    work_units: 0.1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n",
        encoding="utf-8",
    )
    spec = ExperimentSpec.load(experiment)
    assert spec.resolve(spec.system) == (tmp_path / "system.yaml").resolve()
    assert spec.resolve(spec.application or "") == (tmp_path / "app.yaml").resolve()


def test_cli_runs_policy_and_metric_beside_experiment(tmp_path, monkeypatch, capsys):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task:\n    work_units: 0.1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "my_policy.py").write_text(
        "from darpan import Action\n"
        "from darpan.core.event import EventKind\n"
        "class MyPolicy:\n"
        "    def decide(self, state, trigger):\n"
        "        if trigger.kind != EventKind.COMPONENT_READY:\n"
        "            return None\n"
        "        return Action.place(trigger.subject, 'n1', source='mine')\n",
        encoding="utf-8",
    )
    (tmp_path / "my_metric.py").write_text(
        "class Seen:\n"
        "    name = 'seen_events'\n"
        "    def reset(self): self.n = 0\n"
        "    def observe(self, event, state): self.n += 1\n"
        "    def result(self): return self.n\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n"
        "policy: my_policy.py:MyPolicy\n"
        "metrics:\n  - my_metric.py:Seen\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["darpan", "run", str(experiment)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["seen_events"] > 0


def test_real_experiment_materializes_local_network_driver(tmp_path):
    from darpan.cli.run import _session
    from darpan.core.codec import load_system

    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task:\n    work_units: 0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "network_driver.py").write_text(
        "class Driver:\n"
        "    async def bind_route(self, binding, state): pass\n"
        "    async def clear_route(self, app, source, target, state): pass\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: real\n"
        "network_driver: network_driver.py:Driver\n",
        encoding="utf-8",
    )
    spec = ExperimentSpec.load(experiment)
    session = _session(spec, system=load_system(spec.resolve(spec.system)))
    assert session.supports_action(ActionKind.ROUTE)


def test_network_driver_rejected_for_twin_experiment(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task: {work_units: 0.1}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n"
        "network_driver: driver.py:Driver\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="network_driver requires runtime: real"):
        ExperimentSpec.load(experiment)
