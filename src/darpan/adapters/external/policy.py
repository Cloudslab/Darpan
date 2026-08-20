"""Cross-language policy adapter using newline-delimited JSON over stdio."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Any

from darpan.core.action import Action
from darpan.core.event import Event
from darpan.core.serialization import to_primitive
from darpan.core.state import ContinuumState


class ExternalProcessPolicy:
    def __init__(self, command: Sequence[str], *, timeout: float = 10.0) -> None:
        self.command = tuple(command)
        self.timeout = timeout

    def decide(self, state: ContinuumState, trigger: Event) -> Action | None:
        request = {
            "state": to_primitive(state),
            "event": trigger.to_dict(),
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=True,
        )
        line = completed.stdout.strip().splitlines()[-1]
        response: dict[str, Any] = json.loads(line)
        if response.get("action") is None:
            return None
        action = response["action"]
        return Action(
            kind=str(action["kind"]),
            source="external",
            target=action.get("target"),
            payload=dict(action.get("payload", {})),
            priority=int(action.get("priority", 0)),
        )
