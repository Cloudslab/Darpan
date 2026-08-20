from __future__ import annotations

import json

from darpan.cli.main import main


def _experiment(tmp_path, name: str, work_units: float):
    app = tmp_path / f"{name}-app.yaml"
    app.write_text(
        "id: job\ncomponents:\n  task:\n"
        f"    work_units: {work_units}\n"
        "    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / f"{name}.yaml"
    experiment.write_text(
        "system: system.yaml\n"
        f"application: {app.name}\n"
        "runtime: twin\n"
        "policy: first-fit\n",
        encoding="utf-8",
    )
    return experiment


def test_paired_benchmark_cli_runs_matched_seeds_and_records_artifacts(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "system.yaml").write_text(
        "nodes:\n  - id: n1\n    resources: {cpu: 1}\n",
        encoding="utf-8",
    )
    baseline = _experiment(tmp_path, "baseline", 2.0)
    candidate = _experiment(tmp_path, "candidate", 1.0)
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        f"baseline: {baseline.name}\n"
        f"candidate: {candidate.name}\n"
        "metric: application_latency_s\n"
        "higher_is_better: false\n"
        "seeds: [7, 8, 9]\n"
        "bootstrap_resamples: 100\n"
        "output: benchmark-results\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["darpan", "benchmark", str(benchmark)])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["samples"] == 3
    assert payload["summary"]["wins"] == 3
    assert [item["seed"] for item in payload["samples"]] == [7, 8, 9]
    assert (tmp_path / "benchmark-results" / "benchmark.json").is_file()
    assert (tmp_path / "benchmark-results" / "samples.jsonl").is_file()
    assert (tmp_path / "benchmark-results" / "inputs" / "checksums.json").is_file()
    for seed in (7, 8, 9):
        for label in ("baseline", "candidate"):
            run = tmp_path / "benchmark-results" / "runs" / f"seed-{seed}" / label
            assert (run / "run-0001" / "events.jsonl").is_file()
            assert (run / "run-0001" / "state.json").is_file()
            assert (run / "run-0001" / "models.json").is_file()
