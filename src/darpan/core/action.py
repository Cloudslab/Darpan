"""Canonical action proposal envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


class ActionKind:
    PLACE = "component.place"
    MIGRATE = "component.migrate"
    SCALE = "service.scale"
    ROUTE = "network.route"
    RESTART = "component.restart"
    STOP = "component.stop"


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    requested_at: float = field(default_factory=time)
    priority: int = 0
    schema_version: int = 1
    correlation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def place(
        cls,
        instance_id: str,
        node_id: str,
        *,
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        return cls(
            kind=ActionKind.PLACE,
            source=source,
            target=instance_id,
            priority=priority,
            payload={"instance_id": instance_id, "node_id": node_id},
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def migrate(
        cls,
        instance_id: str,
        node_id: str,
        *,
        mode: str = "restart",
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        """Move a running long-lived component to another node."""

        return cls(
            kind=ActionKind.MIGRATE,
            source=source,
            target=instance_id,
            priority=priority,
            payload={
                "instance_id": instance_id,
                "node_id": node_id,
                "mode": mode,
            },
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def restart(
        cls,
        instance_id: str,
        *,
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        """Restart one currently running long-lived component in place."""

        return cls(
            kind=ActionKind.RESTART,
            source=source,
            target=instance_id,
            priority=priority,
            payload={"instance_id": instance_id},
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def scale(
        cls,
        instance_id: str,
        replicas: int,
        *,
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        """Set the desired replica count for one long-running component.

        ``instance_id`` may name any member of the logical replica set. The
        runtime resolves it to the shared application/component identity.
        Scaling changes process replicas only; request/load balancing remains
        a separate data-plane concern.
        """

        return cls(
            kind=ActionKind.SCALE,
            source=source,
            target=instance_id,
            priority=priority,
            payload={"instance_id": instance_id, "replicas": int(replicas)},
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def route(
        cls,
        application_instance_id: str,
        source_component_id: str,
        target_component_id: str,
        path: tuple[str, ...] | list[str],
        *,
        links: tuple[str, ...] | list[str] | None = None,
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        """Bind one logical application flow to an explicit node path.

        An empty path clears the explicit binding and restores automatic
        topology routing.  New bindings affect future transfers; in-flight
        transfers keep the path selected when they started.
        """

        path_items = tuple(str(item) for item in path)
        link_items = () if links is None else tuple(str(item) for item in links)
        return cls(
            kind=ActionKind.ROUTE,
            source=source,
            target=(
                f"{application_instance_id}:{source_component_id}"
                f"->{target_component_id}"
            ),
            priority=priority,
            payload={
                "application_instance_id": application_instance_id,
                "source_component_id": source_component_id,
                "target_component_id": target_component_id,
                "path": list(path_items),
                "links": list(link_items),
            },
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def stop(
        cls,
        instance_id: str,
        *,
        source: str = "policy",
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        """Stop a running long-lived service or stream component."""

        return cls(
            kind=ActionKind.STOP,
            source=source,
            target=instance_id,
            priority=priority,
            payload={"instance_id": instance_id},
            metadata={} if metadata is None else dict(metadata),
        )
