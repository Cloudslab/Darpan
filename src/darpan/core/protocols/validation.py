from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..action import Action
from ..state import ContinuumState


class ActionValidator(Protocol):
    def validate(self, action: Action, state: ContinuumState) -> tuple[bool, str | None]: ...


class ActionArbiter(Protocol):
    def select(self, actions: Sequence[Action], state: ContinuumState) -> list[Action]: ...
