"""Immutable, portable Digital Twin snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darpan.core.codec import continuum_state_from_dict
from darpan.core.serialization import to_primitive
from darpan.core.state import ContinuumState


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_tuple(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class TwinSnapshot:
    state: ContinuumState
    virtual_time: float
    model_states: Mapping[str, Any] = field(default_factory=dict)
    rng_state: Any | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = 1

    def clone_model_states(self) -> dict[str, Any]:
        return deepcopy(dict(self.model_states))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "virtual_time": self.virtual_time,
            "state": to_primitive(self.state),
            "model_states": to_primitive(self.model_states),
            "rng_state": to_primitive(self.rng_state),
            "model_versions": dict(self.model_versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TwinSnapshot:
        return cls(
            state=continuum_state_from_dict(data["state"]),
            virtual_time=float(data["virtual_time"]),
            model_states=deepcopy(dict(data.get("model_states", {}))),
            rng_state=_as_tuple(data.get("rng_state")),
            model_versions={
                str(key): str(value)
                for key, value in data.get("model_versions", {}).items()
            },
            schema_version=int(data.get("schema_version", 1)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TwinSnapshot:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Twin snapshot must contain a JSON object")
        return cls.from_dict(data)
