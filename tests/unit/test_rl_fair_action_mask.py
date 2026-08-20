from __future__ import annotations

from darpan.adapters.rl.action import PlacementActionAdapter
from darpan.core.application import ApplicationSpec, ComponentSpec
from darpan.core.resource import ResourceRequest, ResourceState
from darpan.core.state import ComponentInstanceState, ContinuumState, NodeState


def test_rl_action_mask_uses_fair_admission_not_strict_available_capacity():
    app = ApplicationSpec(
        "app",
        components=(
            ComponentSpec("task", resources=(ResourceRequest("cpu", 1.0),)),
        ),
    )
    decision = ComponentInstanceState(
        id="run:task",
        application_id="app",
        application_instance_id="run",
        component_id="task",
        status="ready",
    )
    state = ContinuumState(
        nodes={
            "edge": NodeState(
                id="edge",
                tier="edge",
                resources={
                    "cpu": ResourceState(
                        "cpu",
                        capacity=1.0,
                        allocated=1.0,
                        attributes={"scheduling": "fair"},
                    )
                },
            )
        },
        applications={"app": app},
        components={decision.id: decision},
    )
    assert PlacementActionAdapter().action_mask(state, decision) == [True]
