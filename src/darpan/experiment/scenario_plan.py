"""Portable event-driven scenario plans for Twin experiments.

A scenario plan describes controlled changes to a registered continuum using
canonical events.  It deliberately does not mutate Twin internals: the same
Reducer/EventLog path used by normal runtime telemetry remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from darpan.core.event import Event, EventKind
from darpan.core.serialization import to_primitive
from darpan.core.topology import LinkSpec
from darpan.runtime.session import Session

_SUPPORTED_KINDS = frozenset(
    {
        EventKind.NODE_OFFLINE,
        EventKind.NODE_RECOVERED,
        "link.down",
        EventKind.LINK_CHANGED,
        EventKind.MEASUREMENT_OBSERVED,
    }
)


@dataclass(frozen=True, slots=True)
class ScenarioPlanEvent:
    at_s: float
    kind: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.at_s < 0:
            raise ValueError("scenario event at_s cannot be negative")
        if self.kind not in _SUPPORTED_KINDS:
            raise ValueError(f"unsupported scenario event kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    events: tuple[ScenarioPlanEvent, ...]
    name: str = "scenario"
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name cannot be empty")
        if not self.events:
            raise ValueError("scenario must contain at least one event")

    @classmethod
    def load(cls, path: str | Path) -> ScenarioPlan:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("scenario configuration must be a mapping")
        events_raw = raw.get("events", ())
        if not isinstance(events_raw, list):
            raise ValueError("scenario events must be a list")
        events: list[ScenarioPlanEvent] = []
        for item in events_raw:
            if not isinstance(item, dict):
                raise ValueError("scenario event must be a mapping")
            if "kind" not in item:
                raise ValueError("scenario event kind is required")
            config = dict(item)
            kind = str(config.pop("kind"))
            at_s = float(config.pop("at_s", config.pop("at", 0.0)))
            events.append(ScenarioPlanEvent(at_s=at_s, kind=kind, config=config))
        return cls(
            events=tuple(events),
            name=str(raw.get("name", source.stem)),
            source=source,
        )

    async def apply(self, session: Session) -> None:
        """Apply/schedule this plan using Twin time or controlled Physical time."""

        deterministic = getattr(session.backend, "schedule_event", None) is not None
        physical_control = getattr(session.backend, "physical_control_driver", None)
        if not deterministic and physical_control is None:
            raise RuntimeError(
                "Real scenario plans require a PhysicalControlDriver; canonical "
                "state is never mutated as a substitute for physical control"
            )

        base_time = session.clock.now()
        link_specs: dict[str, LinkSpec] = {
            link_id: link.spec for link_id, link in session.state.links.items()
        }
        for item in sorted(self.events, key=lambda event: event.at_s):
            event = self._event(item, base_time=base_time, link_specs=link_specs)
            if item.at_s == 0:
                await session.emit(event)
            else:
                await session.schedule_event(event, at=event.event_time)

    def _event(
        self,
        item: ScenarioPlanEvent,
        *,
        base_time: float,
        link_specs: dict[str, LinkSpec],
    ) -> Event:
        at = base_time + item.at_s
        config = item.config
        metadata = {"scenario": self.name, "scenario_at_s": item.at_s}

        if item.kind in {EventKind.NODE_OFFLINE, EventKind.NODE_RECOVERED}:
            node_id = _required_text(config, "node_id", item.kind)
            return Event(
                kind=item.kind,
                event_time=at,
                source="experiment.scenario",
                subject=node_id,
                payload={"node_id": node_id},
                metadata=metadata,
            )

        if item.kind == "link.down":
            link_id = _required_text(config, "link_id", item.kind)
            if link_id not in link_specs:
                raise ValueError(f"scenario references unknown link: {link_id}")
            return Event(
                kind=EventKind.LINK_REMOVED,
                event_time=at,
                source="experiment.scenario",
                subject=link_id,
                payload={"link_id": link_id},
                metadata=metadata,
            )

        if item.kind == EventKind.LINK_CHANGED:
            link_id = _required_text(config, "link_id", item.kind)
            try:
                current = link_specs[link_id]
            except KeyError as exc:
                raise ValueError(f"scenario references unknown link: {link_id}") from exc
            updates: dict[str, Any] = {}
            for key in ("latency_ms", "bandwidth_mbps", "bidirectional", "labels"):
                if key in config:
                    updates[key] = config[key]
            if not updates:
                raise ValueError("link.changed scenario event must change at least one field")
            if "latency_ms" in updates:
                updates["latency_ms"] = float(updates["latency_ms"])
            if "bandwidth_mbps" in updates:
                updates["bandwidth_mbps"] = float(updates["bandwidth_mbps"])
            if "bidirectional" in updates:
                updates["bidirectional"] = bool(updates["bidirectional"])
            if "labels" in updates:
                if not isinstance(updates["labels"], dict):
                    raise ValueError("link.changed labels must be a mapping")
                updates["labels"] = dict(updates["labels"])
            changed = replace(current, **updates)
            link_specs[link_id] = changed
            return Event(
                kind=EventKind.LINK_CHANGED,
                event_time=at,
                source="experiment.scenario",
                subject=link_id,
                payload={"link": to_primitive(changed)},
                metadata=metadata,
            )

        if item.kind == EventKind.MEASUREMENT_OBSERVED:
            name = _required_text(config, "name", item.kind)
            target = _required_text(config, "target", item.kind)
            if "value" not in config:
                raise ValueError("measurement.observed scenario event requires value")
            measurement: dict[str, Any] = {
                "name": name,
                "value": config["value"],
                "unit": str(config.get("unit", "1")),
                "target": target,
                "timestamp": at,
                "source": "experiment.scenario",
                "metadata": {"scenario": self.name, **dict(config.get("metadata", {}))},
            }
            for key in ("uncertainty", "quality"):
                if config.get(key) is not None:
                    measurement[key] = float(config[key])
            return Event(
                kind=EventKind.MEASUREMENT_OBSERVED,
                event_time=at,
                source="experiment.scenario",
                subject=target,
                payload={"measurement": measurement},
                metadata=metadata,
            )

        raise AssertionError(f"unhandled scenario event kind: {item.kind}")


def _required_text(config: dict[str, Any], key: str, kind: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"{kind} scenario event requires {key}")
    return value
