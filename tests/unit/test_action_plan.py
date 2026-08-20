from __future__ import annotations

from darpan.core.action import Action
from darpan.core.state import ContinuumState
from darpan.runtime.action_plan import PriorityArbiter


def test_priority_arbiter_resolves_same_target():
    low = Action("custom", "a", target="node", priority=1)
    high = Action("custom", "b", target="node", priority=10)
    selected = PriorityArbiter().select([low, high], ContinuumState())
    assert selected == [high]
