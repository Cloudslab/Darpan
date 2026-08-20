from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..application import ComponentSpec


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    return_code: int
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    output_bytes: int = 0
    measurements: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.return_code == 0


class Executor(Protocol):
    async def execute(self, component: ComponentSpec) -> ExecutionResult: ...
