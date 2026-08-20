"""Physical network-control extension boundary for explicit flow routing."""

from __future__ import annotations

from typing import Protocol

from darpan.core.state import ContinuumState, FlowRouteBinding


class NetworkControlDriver(Protocol):
    """Enforce Darpan flow bindings in a real network control plane.

    Implementations may target Linux routing/policy rules, an SDN controller,
    an overlay, Kubernetes networking, or another real mechanism.  Darpan does
    not provide a no-op implementation because that would falsely claim that a
    Physical Continuum route changed when only controller state changed.
    """

    async def bind_route(
        self,
        binding: FlowRouteBinding,
        state: ContinuumState,
    ) -> None: ...

    async def clear_route(
        self,
        application_instance_id: str,
        source_component_id: str,
        target_component_id: str,
        state: ContinuumState,
    ) -> None: ...
