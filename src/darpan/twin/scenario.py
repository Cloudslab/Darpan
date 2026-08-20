"""Scenario = immutable snapshot + ordered modifications + seed/horizon."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from darpan.core.codec import measurement_from_dict
from darpan.core.event import Event
from darpan.core.measurement import Measurement
from darpan.core.serialization import to_primitive
from darpan.core.state import ContinuumState

from .modification import ChangeLink, RemoveNode, ScenarioModification, SetMeasurement
from .snapshot import TwinSnapshot


def _modification_to_dict(modification: ScenarioModification) -> dict[str, Any]:
    if isinstance(modification, RemoveNode):
        return {"kind": "remove_node", "node_id": modification.node_id}
    if isinstance(modification, ChangeLink):
        return {
            "kind": "change_link",
            "link_id": modification.link_id,
            "latency_ms": modification.latency_ms,
            "bandwidth_mbps": modification.bandwidth_mbps,
        }
    if isinstance(modification, SetMeasurement):
        return {
            "kind": "set_measurement",
            "measurement": to_primitive(modification.measurement),
        }
    raise TypeError(f"unsupported scenario modification: {type(modification).__name__}")


def _modification_from_dict(data: Mapping[str, Any]) -> ScenarioModification:
    kind = data.get("kind")
    if kind == "remove_node":
        return RemoveNode(str(data["node_id"]))
    if kind == "change_link":
        return ChangeLink(
            str(data["link_id"]),
            None if data.get("latency_ms") is None else float(data["latency_ms"]),
            (
                None
                if data.get("bandwidth_mbps") is None
                else float(data["bandwidth_mbps"])
            ),
        )
    if kind == "set_measurement":
        return SetMeasurement(measurement_from_dict(data["measurement"]))
    raise ValueError(f"unknown scenario modification kind: {kind!r}")


@dataclass(slots=True)
class Scenario:
    snapshot: TwinSnapshot
    seed: int = 0
    horizon_s: float | None = None
    modifications: list[ScenarioModification] = field(default_factory=list)
    injected_events: list[Event] = field(default_factory=list)

    def remove_node(self, node_id: str) -> Scenario:
        self.modifications.append(RemoveNode(node_id))
        return self

    def change_link(
        self,
        link_id: str,
        *,
        latency_ms: float | None = None,
        bandwidth_mbps: float | None = None,
    ) -> Scenario:
        self.modifications.append(ChangeLink(link_id, latency_ms, bandwidth_mbps))
        return self

    def set_measurement(self, measurement: Measurement) -> Scenario:
        self.modifications.append(SetMeasurement(measurement))
        return self

    def inject_event(self, event: Event) -> Scenario:
        self.injected_events.append(event)
        return self

    def materialize(self) -> ContinuumState:
        state = self.snapshot.state
        for modification in self.modifications:
            state = modification.apply(state)
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "horizon_s": self.horizon_s,
            "snapshot": self.snapshot.to_dict(),
            "modifications": [
                _modification_to_dict(item) for item in self.modifications
            ],
            "injected_events": [event.to_dict() for event in self.injected_events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        return cls(
            snapshot=TwinSnapshot.from_dict(data["snapshot"]),
            seed=int(data.get("seed", 0)),
            horizon_s=(
                None if data.get("horizon_s") is None else float(data["horizon_s"])
            ),
            modifications=[
                _modification_from_dict(item)
                for item in data.get("modifications", [])
            ],
            injected_events=[
                Event.from_dict(item) for item in data.get("injected_events", [])
            ],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Twin scenario must contain a JSON object")
        return cls.from_dict(data)
