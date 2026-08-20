"""Continuum topology specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .resource import ResourceSpec


@dataclass(frozen=True, slots=True)
class NodeSpec:
    id: str
    tier: str = "edge"
    resources: tuple[ResourceSpec, ...] = ()
    capabilities: frozenset[str] = frozenset()
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id cannot be empty")
        names = [resource.name for resource in self.resources]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate resource on node {self.id}")


@dataclass(frozen=True, slots=True)
class LinkSpec:
    id: str
    source: str
    target: str
    latency_ms: float = 0.0
    bandwidth_mbps: float = float("inf")
    bidirectional: bool = True
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.source or not self.target:
            raise ValueError("link id/source/target cannot be empty")
        if self.latency_ms < 0 or self.bandwidth_mbps <= 0:
            raise ValueError("invalid link latency/bandwidth")


@dataclass(frozen=True, slots=True)
class SystemSpec:
    nodes: tuple[NodeSpec, ...]
    links: tuple[LinkSpec, ...] = ()
    name: str = "continuum"

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node id")
        known = set(node_ids)
        for link in self.links:
            if link.source not in known or link.target not in known:
                raise ValueError(f"link {link.id} references unknown node")
