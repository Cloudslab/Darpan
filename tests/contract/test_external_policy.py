from __future__ import annotations

import sys

from darpan.adapters.external import ExternalProcessPolicy
from darpan.core.event import Event
from darpan.core.state import ContinuumState


def test_non_python_policy_can_return_canonical_action(tmp_path):
    script = tmp_path / "policy.py"
    script.write_text(
        "import json,sys\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'action': {'kind':'custom.test',"
        "'target':'x','payload':{'value':1}}}))\n",
        encoding="utf-8",
    )
    policy = ExternalProcessPolicy([sys.executable, str(script)])
    action = policy.decide(ContinuumState(), Event("test", 0.0, "test"))
    assert action is not None
    assert action.kind == "custom.test"
    assert action.payload["value"] == 1
