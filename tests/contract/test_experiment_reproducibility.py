from __future__ import annotations

import json

from darpan.cli.main import main


def _write_experiment(tmp_path, output: str):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: app\ncomponents:\n  task:\n    resources: {cpu: 1}\n    work_units: 0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "metric.py").write_text(
        "import random\n"
        "class RandomMetric:\n"
        "    name = 'random_value'\n"
        "    def reset(self): self.value = random.random()\n"
        "    def observe(self, event, state): pass\n"
        "    def result(self): return self.value\n",
        encoding="utf-8",
    )
    experiment = tmp_path / f"experiment-{output}.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "application: app.yaml\n"
        "runtime: twin\n"
        "repeat: 2\n"
        "seed: 41\n"
        "metrics: [metric.py:RandomMetric]\n"
        f"output: {output}\n",
        encoding="utf-8",
    )
    return experiment


def test_experiment_output_is_reproducible_and_self_describing(tmp_path, monkeypatch, capsys):
    first = _write_experiment(tmp_path, "runs-a")
    monkeypatch.setattr("sys.argv", ["darpan", "run", str(first)])
    main()
    first_stdout = json.loads(capsys.readouterr().out)

    second = _write_experiment(tmp_path, "runs-b")
    monkeypatch.setattr("sys.argv", ["darpan", "run", str(second)])
    main()
    second_stdout = json.loads(capsys.readouterr().out)

    assert first_stdout == second_stdout
    root = tmp_path / "runs-a"
    assert json.loads((root / "run-0001" / "metadata.json").read_text())["seed"] == 41
    assert json.loads((root / "run-0002" / "metadata.json").read_text())["seed"] == 42
    assert (root / "run-0001" / "events.jsonl").stat().st_size > 0
    assert (root / "run-0001" / "state.json").is_file()
    assert (root / "run-0001" / "models.json").is_file()
    assert (root / "inputs-manifest.json").is_file()
    assert json.loads((root / "summary.json").read_text()) == first_stdout
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["python"]
    assert provenance["platform"]
