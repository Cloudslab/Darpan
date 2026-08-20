"""Data and storage model for data-aware Continuum research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DataObjectSpec:
    id: str
    size_bytes: int
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or self.size_bytes < 0:
            raise ValueError("invalid data object")


@dataclass(frozen=True, slots=True)
class StorageSpec:
    id: str
    node_id: str
    capacity_bytes: int
    kind: str = "local"
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataLocation:
    data_id: str
    storage_id: str
    node_id: str
