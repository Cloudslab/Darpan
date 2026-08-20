"""Simple node discovery registry; distributed discovery can replace this adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentEndpoint:
    node_id: str
    host: str
    port: int = 8765


class DiscoveryRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentEndpoint] = {}

    def register(self, endpoint: AgentEndpoint) -> None:
        self._agents[endpoint.node_id] = endpoint

    def get(self, node_id: str) -> AgentEndpoint:
        return self._agents[node_id]

    def all(self) -> tuple[AgentEndpoint, ...]:
        return tuple(self._agents.values())
