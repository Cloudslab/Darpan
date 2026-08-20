"""RL action adapters decode model outputs to canonical Darpan Actions."""

from __future__ import annotations

from typing import Any, Protocol

from darpan.core.action import Action
from darpan.core.state import ComponentInstanceState, ContinuumState
from darpan.runtime.action_plan import placement_feasibility


class RLActionAdapter(Protocol):
    def decode(
        self,
        action: Any,
        state: ContinuumState,
        decision: ComponentInstanceState,
    ) -> Action: ...

    def action_mask(
        self, state: ContinuumState, decision: ComponentInstanceState
    ) -> list[bool]: ...


class PlacementActionAdapter:
    def _nodes(self, state: ContinuumState) -> list[str]:
        return sorted(state.nodes)

    def decode(
        self,
        action: Any,
        state: ContinuumState,
        decision: ComponentInstanceState,
    ) -> Action:
        nodes = self._nodes(state)
        index = int(action)
        if index < 0 or index >= len(nodes):
            raise ValueError(f"placement action {index} outside [0, {len(nodes)})")
        return Action.place(decision.id, nodes[index], source="rl")

    def action_mask(
        self, state: ContinuumState, decision: ComponentInstanceState
    ) -> list[bool]:
        return [
            placement_feasibility(state, decision.id, node_id)[0]
            for node_id in self._nodes(state)
        ]
