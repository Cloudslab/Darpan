"""Canonical event envelope with distributed-system ordering metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


class EventKind:
    NODE_REGISTERED = "node.registered"
    NODE_OFFLINE = "node.offline"
    NODE_RECOVERED = "node.recovered"
    NODE_REMOVED = "node.removed"
    LINK_REGISTERED = "link.registered"
    LINK_CHANGED = "link.changed"
    LINK_REMOVED = "link.removed"
    APPLICATION_REGISTERED = "application.registered"
    APPLICATION_SUBMITTED = "application.submitted"
    APPLICATION_COMPLETED = "application.completed"
    COMPONENT_CREATED = "component.created"
    COMPONENT_READY = "component.ready"
    COMPONENT_SCHEDULED = "component.scheduled"
    COMPONENT_STARTED = "component.started"
    COMPONENT_SCALING = "component.scaling"
    COMPONENT_SCALED = "component.scaled"
    COMPONENT_MIGRATING = "component.migrating"
    COMPONENT_RESTARTING = "component.restarting"
    COMPONENT_RETRYING = "component.retrying"
    COMPONENT_COMPLETED = "component.completed"
    COMPONENT_FAILED = "component.failed"
    RESOURCE_ALLOCATED = "resource.allocated"
    RESOURCE_RELEASED = "resource.released"
    MEASUREMENT_OBSERVED = "measurement.observed"
    ACTION_REQUESTED = "action.requested"
    ACTION_ACCEPTED = "action.accepted"
    ACTION_REJECTED = "action.rejected"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    DATA_TRANSFER_STARTED = "data.transfer_started"
    DATA_TRANSFER_COMPLETED = "data.transfer_completed"
    DATA_TRANSFER_FAILED = "data.transfer_failed"
    DATA_LOCATION_CHANGED = "data.location_changed"
    FLOW_ROUTING = "flow.routing"
    FLOW_ROUTED = "flow.routed"
    TIME_ADVANCED = "runtime.time_advanced"


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    event_time: float
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    subject: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1
    ingest_time: float = field(default_factory=time)
    source_sequence: int | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind or not self.source:
            raise ValueError("event kind/source cannot be empty")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "event_time": self.event_time,
            "ingest_time": self.ingest_time,
            "source": self.source,
            "source_sequence": self.source_sequence,
            "subject": self.subject,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Event:
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            schema_version=int(data.get("schema_version", 1)),
            event_time=float(data["event_time"]),
            ingest_time=float(data.get("ingest_time", time())),
            source=str(data["source"]),
            source_sequence=data.get("source_sequence"),
            subject=data.get("subject"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
        )
