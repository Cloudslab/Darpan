from __future__ import annotations

import asyncio

from darpan import Action, Darpan
from darpan.core.event import EventKind
from darpan.runtime.dispatch import PolicyDispatcher


class FeedbackHungryPolicy:
    """A deliberately unsafe policy used to prove dispatcher isolation."""

    def __init__(self) -> None:
        self.action_feedback_seen = 0
        self.decisions = 0

    def decide(self, state, trigger):
        if trigger.kind.startswith("action."):
            self.action_feedback_seen += 1
            # Old create_task based dispatch could recursively re-enter here.
            ready = state.ready_components()
            if ready:
                return Action.place(ready[0].id, sorted(state.nodes)[0], source="feedback")
        if trigger.kind == EventKind.COMPONENT_READY and trigger.subject is not None:
            self.decisions += 1
            return Action.place(trigger.subject, sorted(state.nodes)[0], source="test")
        return None


def test_policy_dispatcher_is_structured_and_filters_action_feedback(small_system, small_app):
    async def run():
        session = Darpan.twin()
        policy = FeedbackHungryPolicy()
        dispatcher = PolicyDispatcher(session, [policy])
        session.subscribe(dispatcher)
        await session.start()
        await session.register_system(small_system)
        await session.submit_application(small_app)
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=2,
        )
        assert policy.decisions == len(small_app.components)
        assert policy.action_feedback_seen == 0
        await session.close()

    asyncio.run(run())
