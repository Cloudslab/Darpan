"""Reference Physical placement policies with a fixed edge ingress.

The source component is pinned to edge-1 so downstream policies are compared
against the same physical data origin.  The classes intentionally reuse the
built-in heuristics for every other placement decision.
"""

from darpan.core.action import Action
from darpan.experiment.baselines import FirstFitPolicy, LatencyAwarePolicy, RoundRobinPolicy


class _PinnedIngressPolicy:
    delegate_type = FirstFitPolicy

    def __init__(self) -> None:
        self.delegate = self.delegate_type()

    def decide(self, state, trigger):
        for instance in sorted(state.ready_components(), key=lambda item: item.id):
            if instance.component_id == "source":
                return Action.place(instance.id, "edge-1", source="reference-ingress")
        return self.delegate.decide(state, trigger)


class PinnedIngressFirstFitPolicy(_PinnedIngressPolicy):
    delegate_type = FirstFitPolicy


class PinnedIngressRoundRobinPolicy(_PinnedIngressPolicy):
    delegate_type = RoundRobinPolicy


class PinnedIngressLatencyAwarePolicy(_PinnedIngressPolicy):
    delegate_type = LatencyAwarePolicy
