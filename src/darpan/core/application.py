"""Application model supporting tasks, services, functions, and streams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .resource import ResourceRequest


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded automatic retry policy for finite components.

    ``max_retries`` counts retries after the initial attempt. An empty ``on``
    set means any canonical failure kind is eligible.
    """

    max_retries: int = 0
    on: frozenset[str] = frozenset()
    backoff_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("retry max_retries cannot be negative")
        if self.backoff_s < 0:
            raise ValueError("retry backoff_s cannot be negative")
        if any(not str(kind).strip() for kind in self.on):
            raise ValueError("retry failure kinds cannot be empty")

    def allows(self, failure_kind: str) -> bool:
        return self.max_retries > 0 and (not self.on or failure_kind in self.on)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    id: str
    kind: str = "task"
    image: str | None = None
    command: tuple[str, ...] = ()
    resources: tuple[ResourceRequest, ...] = ()
    work_units: float = 1.0
    labels: Mapping[str, str] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("component id cannot be empty")
        if self.work_units < 0:
            raise ValueError("work_units cannot be negative")
        if self.is_long_running and self.retry.max_retries:
            raise ValueError(
                "automatic retry is only supported for finite components; "
                "use RESTART/MIGRATE for running services and streams"
            )

    @property
    def is_long_running(self) -> bool:
        return self.kind in {"service", "stream"}


@dataclass(frozen=True, slots=True)
class FlowSpec:
    source: str
    target: str
    data_size_bytes: int = 0
    kind: str = "data"
    artifact: str | None = None
    target_path: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_size_bytes < 0:
            raise ValueError("flow data_size_bytes cannot be negative")
        if self.target_path is not None and self.artifact is None:
            raise ValueError("flow target_path requires artifact")
        for name, value in (
            ("artifact", self.artifact),
            ("target_path", self.target_path),
        ):
            if value is None:
                continue
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
                raise ValueError(f"flow {name} must be a safe relative path")

    @property
    def artifact_target(self) -> str | None:
        return self.target_path or self.artifact


@dataclass(frozen=True, slots=True)
class ApplicationSpec:
    id: str
    components: tuple[ComponentSpec, ...]
    flows: tuple[FlowSpec, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("application id cannot be empty")
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError(f"duplicate component in application {self.id}")
        known = set(component_ids)
        for flow in self.flows:
            if flow.source not in known or flow.target not in known:
                raise ValueError("flow references unknown component")
        self._assert_acyclic()

    def component(self, component_id: str) -> ComponentSpec:
        for component in self.components:
            if component.id == component_id:
                return component
        raise KeyError(component_id)

    def predecessors(self, component_id: str) -> tuple[str, ...]:
        return tuple(flow.source for flow in self.flows if flow.target == component_id)

    def successors(self, component_id: str) -> tuple[str, ...]:
        return tuple(flow.target for flow in self.flows if flow.source == component_id)

    def flow(self, source: str, target: str) -> FlowSpec | None:
        for flow in self.flows:
            if flow.source == source and flow.target == target:
                return flow
        return None

    def roots(self) -> tuple[str, ...]:
        return tuple(
            component.id
            for component in self.components
            if not self.predecessors(component.id)
        )

    def _assert_acyclic(self) -> None:
        indegree = {component.id: 0 for component in self.components}
        successors: dict[str, list[str]] = {key: [] for key in indegree}
        for flow in self.flows:
            indegree[flow.target] += 1
            successors[flow.source].append(flow.target)
        ready = [key for key, value in indegree.items() if value == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for successor in successors[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(indegree):
            raise ValueError("application flows must be acyclic")
