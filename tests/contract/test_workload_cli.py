from __future__ import annotations

import json

from darpan.cli.main import main


def test_experiment_cli_runs_workload_file(tmp_path, monkeypatch, capsys):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "app.yaml").write_text(
        "id: job\ncomponents:\n  task:\n    resources: {cpu: 1}\n    work_units: 0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "workload.yaml").write_text(
        "applications:\n  - app.yaml\n"
        "arrivals:\n  - application: job\n    at: 0\n    count: 2\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        "workload: workload.yaml\n"
        "runtime: twin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["darpan", "run", str(experiment)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["application_latency_s"] > 0
