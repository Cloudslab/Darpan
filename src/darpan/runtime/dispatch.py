"""Dispatch helpers for policies, metrics, analyzers, and observers.

The policy dispatcher deliberately serializes policy decisions.  A policy may
cause actions which themselves emit events synchronously (especially in the
Twin runtime).  Calling the policy recursively from those events makes the
control loop re-entrant and can lead to duplicate decisions or unbounded
recursion.  The dispatcher therefore queues events and drains them from one
structured coroutine instead of spawning detached tasks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable

from darpan.core.event import Event, EventKind
from darpan.core.protocols.policy import Policy
from darpan.core.state import ContinuumState

Listener = Callable[[Event, ContinuumState], Awaitable[None] | None]


class PolicyDispatcher:
    """Serialize event-driven policy decisions for one :class:`Session`.

    Action lifecycle events are excluded by default.  They are feedback about
    a decision already made, and feeding them directly back into a generic
    event-driven policy is a common source of accidental control-loop
    recursion.  Research code that explicitly needs those events can opt in.
    """

    ACTION_FEEDBACK_KINDS = frozenset(
        {
            EventKind.ACTION_REQUESTED,
            EventKind.ACTION_ACCEPTED,
            EventKind.ACTION_REJECTED,
            EventKind.ACTION_STARTED,
            EventKind.ACTION_COMPLETED,
            EventKind.ACTION_FAILED,
        }
    )

    def __init__(
        self,
        session,
        policies: list[Policy],
        *,
        include_action_feedback: bool = False,
    ) -> None:
        self.session = session
        self.policies = policies
        self.include_action_feedback = include_action_feedback
        self._pending: deque[tuple[Event, ContinuumState]] = deque()
        self._processing = False

    async def __call__(self, event: Event, state: ContinuumState) -> None:
        if not self.include_action_feedback and event.kind in self.ACTION_FEEDBACK_KINDS:
            return

        self._pending.append((event, state))
        if self._processing:
            return

        self._processing = True
        try:
            while self._pending:
                pending_event, pending_state = self._pending.popleft()
                actions = []
                for policy in self.policies:
                    result = policy.decide(pending_state, pending_event)
                    if result is None:
                        continue
                    if isinstance(result, list):
                        actions.extend(result)
                    else:
                        actions.append(result)
                if actions:
                    await self.session.apply_many(actions)
        except Exception:
            # Do not replay stale state snapshots if the policy or backend
            # raises.  A subsequent event starts a fresh drain cycle.
            self._pending.clear()
            raise
        finally:
            self._processing = False
