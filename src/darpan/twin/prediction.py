"""Prediction value used by high-level Twin queries and assurance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Prediction:
    estimate: Any
    uncertainty: float | None = None
    interval: tuple[float, float] | None = None
    samples: tuple[float, ...] = ()
    horizon_s: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
