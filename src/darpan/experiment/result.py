from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    metrics: Mapping[str, Any]
    objectives: tuple[float, ...] = ()
    feasible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
