"""Deterministic lifecycle policy used only by the bundled reference study."""

from darpan.core.action import Action
from darpan.core.event import EventKind


class ReferenceLifecyclePolicy:
    """Start a two-stage long-running pipeline, then stop it cleanly."""

    def decide(self, state, trigger):
        if trigger.kind == EventKind.COMPONENT_READY and trigger.subject:
            instance = state.components.get(trigger.subject)
            if instance is None or instance.status != "ready":
                return None
            node = "edge" if instance.component_id == "ingest" else "fog"
            return Action.place(instance.id, node, source="reference-lifecycle")

        if trigger.kind == EventKind.COMPONENT_STARTED and trigger.subject:
            instance = state.components.get(trigger.subject)
            if instance is None or instance.component_id != "processor":
                return None
            app_instance = instance.application_instance_id
            ingest_id = f"{app_instance}:ingest"
            processor_id = f"{app_instance}:processor"
            if state.components[ingest_id].status != "running":
                return None
            if state.components[processor_id].status != "running":
                return None
            return [
                Action.stop(ingest_id, source="reference-lifecycle"),
                Action.stop(processor_id, source="reference-lifecycle"),
            ]
        return None
