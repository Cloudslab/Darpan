"""Extensible measurements shared by physical and digital Continuums."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Measurement:
    name: str
    value: Any
    unit: str = "1"
    target: str | None = None
    timestamp: float = 0.0
    source: str = "unknown"
    uncertainty: float | None = None
    quality: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("measurement name cannot be empty")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("measurement uncertainty cannot be negative")
        if self.quality is not None and not 0 <= self.quality <= 1:
            raise ValueError("measurement quality must be in [0, 1]")
