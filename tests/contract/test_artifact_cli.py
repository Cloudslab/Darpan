from __future__ import annotations

import json
from types import SimpleNamespace

from darpan.cli.artifact import _verify
from darpan.experiment.artifact import seal_artifact


def test_artifact_verify_cli_prints_verified_report(tmp_path, capsys):
    (tmp_path / "value.json").write_text("{}\n", encoding="utf-8")
    seal_artifact(tmp_path, schema="test/v1", identity={"id": "x"})

    _verify(SimpleNamespace(directory=str(tmp_path)))

    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["identity"] == {"id": "x"}
