from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..event import Event
from ..state import ContinuumState


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    estimate: Any
    uncertainty: float | None = None
    lower: float | None = None
    upper: float | None = None
    samples: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TwinModel(Protocol):
    name: str

    def observe(self, event: Event, state: ContinuumState) -> None: ...

    def advance(
        self, start_time: float, end_time: float, state: ContinuumState
    ) -> None: ...

    def predict(self, query: Mapping[str, Any], state: ContinuumState) -> ModelPrediction: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def restore(self, snapshot: Mapping[str, Any]) -> None: ...
