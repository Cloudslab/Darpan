"""Stable runtime identities for horizontally scaled component replicas."""

from __future__ import annotations

from darpan.core.state import ComponentInstanceState, ContinuumState


def logical_instance_id(application_instance_id: str, component_id: str) -> str:
    """Return the backward-compatible replica-zero instance id."""

    return f"{application_instance_id}:{component_id}"


def replica_instance_id(
    application_instance_id: str,
    component_id: str,
    replica_index: int,
) -> str:
    """Return a deterministic runtime id for one replica.

    Replica zero intentionally keeps the v0.1 identity so existing traces,
    policies, and workload artifacts remain compatible.
    """

    if replica_index < 0:
        raise ValueError("replica_index cannot be negative")
    base = logical_instance_id(application_instance_id, component_id)
    return base if replica_index == 0 else f"{base}#replica-{replica_index}"


def active_replicas(
    state: ContinuumState,
    instance_id: str,
) -> tuple[ComponentInstanceState, ...]:
    return state.component_replicas(instance_id, include_terminal=False)


def next_replica_indices(
    state: ContinuumState,
    instance_id: str,
    count: int,
) -> tuple[int, ...]:
    """Choose deterministic never-colliding replica indexes."""

    if count < 0:
        raise ValueError("count cannot be negative")
    used = {item.replica_index for item in state.component_replicas(instance_id)}
    selected: list[int] = []
    candidate = 1
    while len(selected) < count:
        if candidate not in used:
            selected.append(candidate)
        candidate += 1
    return tuple(selected)
