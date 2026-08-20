from __future__ import annotations

import json
from types import SimpleNamespace

from darpan.cli.suite import _lock, _verify


def test_suite_lock_and_verify_cli_write_durable_artifacts(tmp_path, capsys):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: paper\nversion: '1'\nitems:\n"
        "  - id: trace\n    kind: evidence\n    path: evidence.json\n",
        encoding="utf-8",
    )
    lock = tmp_path / "paper.lock.json"
    _lock(SimpleNamespace(suite=str(suite), output=str(lock)))
    assert json.loads(lock.read_text())["name"] == "paper"

    report = tmp_path / "verification"
    _verify(
        SimpleNamespace(
            suite=str(suite),
            lock=str(lock),
            output=str(report),
        )
    )
    payload = json.loads((report / "suite-verification.json").read_text())
    assert payload["verified"] is True
    assert (report / "inputs" / "suite.yaml").is_file()
    assert (report / "inputs" / "suite.lock.json").is_file()
    capsys.readouterr()
