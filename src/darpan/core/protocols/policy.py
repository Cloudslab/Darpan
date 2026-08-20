from __future__ import annotations

from typing import Protocol

from ..action import Action
from ..event import Event
from ..state import ContinuumState


class Policy(Protocol):
    def decide(
        self, state: ContinuumState, trigger: Event
    ) -> Action | list[Action] | None: ...
