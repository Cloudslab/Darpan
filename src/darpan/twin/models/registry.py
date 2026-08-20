"""Named Twin model registry with snapshot/restore semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from darpan.core.event import Event
from darpan.core.protocols.model import TwinModel
from darpan.core.state import ContinuumState


class ModelRegistry:
    def __init__(self, models: list[TwinModel] | None = None) -> None:
        self._models: dict[str, TwinModel] = {}
        for model in models or []:
            self.register(model)

    def register(self, model: TwinModel, *, replace: bool = False) -> None:
        if model.name in self._models and not replace:
            raise ValueError(f"duplicate Twin model: {model.name}")
        self._models[model.name] = model

    def get(self, name: str) -> TwinModel:
        return self._models[name]

    def observe(self, event: Event, state: ContinuumState) -> None:
        for model in self._models.values():
            model.observe(event, state)

    def advance(self, start: float, end: float, state: ContinuumState) -> None:
        for model in self._models.values():
            model.advance(start, end, state)

    def snapshot(self) -> dict[str, Any]:
        return {name: dict(model.snapshot()) for name, model in self._models.items()}

    def restore(
        self,
        snapshot: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> None:
        unknown = set(snapshot).difference(self._models)
        if strict and unknown:
            raise ValueError(
                "model snapshot contains unavailable models: "
                + ", ".join(sorted(unknown))
            )
        for name, model_state in snapshot.items():
            if name in self._models:
                self._models[name].restore(model_state)

    def names(self) -> tuple[str, ...]:
        return tuple(self._models)
