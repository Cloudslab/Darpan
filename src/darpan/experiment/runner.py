"""Policy-driven experiment runner shared by real and Twin sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from darpan.core.application import ApplicationSpec
from darpan.core.event import EventKind
from darpan.core.protocols.metric import Metric
from darpan.core.protocols.policy import Policy
from darpan.core.topology import SystemSpec
from darpan.core.workload import WorkloadSpec
from darpan.runtime.dispatch import PolicyDispatcher
from darpan.runtime.session import Session

from .constraint import Constraint
from .metric import MetricSet
from .objective import Objective
from .result import ExperimentResult


def _retry_evidence(session: Session, instance_ids: set[str]) -> dict[str, object]:
    retried: set[str] = set()
    retry_attempts = 0
    pending: dict[str, float] = {}
    downtimes: list[float] = []
    for event in session.event_log:
        component_id = event.payload.get("instance_id")
        if component_id is None:
            continue
        component_id = str(component_id)
        component = session.state.components.get(component_id)
        if component is None or component.application_instance_id not in instance_ids:
            continue
        if event.kind == EventKind.COMPONENT_RETRYING:
            retry_attempts += 1
            retried.add(component_id)
            pending[component_id] = event.event_time
        elif event.kind == EventKind.COMPONENT_STARTED and component_id in pending:
            downtimes.append(max(0.0, event.event_time - pending.pop(component_id)))
    recovered = sorted(
        component_id
        for component_id in retried
        if session.state.components[component_id].status == "completed"
    )
    exhausted = sorted(
        component_id
        for component_id in retried
        if session.state.components[component_id].status == "failed"
    )
    recovery_rate = None if not retried else len(recovered) / len(retried)
    retry_downtime_s = None if not downtimes else sum(downtimes) / len(downtimes)
    return {
        "retry_attempts": retry_attempts,
        "retried_components": sorted(retried),
        "recovered_components": recovered,
        "retry_exhausted_components": exhausted,
        "retry_recovery_rate": recovery_rate,
        "retry_downtime_s": retry_downtime_s,
    }


SetupHook = Callable[[Session], Awaitable[None]]


class ExperimentRunner:
    def __init__(
        self,
        session: Session,
        *,
        metrics: Iterable[Metric],
        objectives: Iterable[Objective] = (),
        constraints: Iterable[Constraint] = (),
    ) -> None:
        self.session = session
        self.metric_set = MetricSet(metrics)
        self.objectives = tuple(objectives)
        self.constraints = tuple(constraints)
        self.session.subscribe(self.metric_set.observe)

    async def run(
        self,
        system: SystemSpec,
        application: ApplicationSpec,
        policy: Policy,
        *,
        timeout: float = 60.0,
        setup: SetupHook | None = None,
    ) -> ExperimentResult:
        self.metric_set.reset()
        await self.session.start()
        await self.session.register_system(system)
        if setup is not None:
            await setup(self.session)
        dispatcher = PolicyDispatcher(self.session, [policy])
        self.session.subscribe(dispatcher)
        try:
            instance_id = await self.session.submit_application(application)
            completion = await self.session.wait_for(
                lambda event, state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance_id
                ),
                timeout=timeout,
            )
            application_success = bool(completion.event.payload.get("success", True))
            metrics = self.metric_set.results()
            numeric = {
                key: float(value) for key, value in metrics.items() if value is not None
            }
            objective_values = tuple(
                objective.value(numeric) for objective in self.objectives
            )
            constraints_satisfied = all(
                constraint.satisfied(numeric) for constraint in self.constraints
            )
            feasible = application_success and constraints_satisfied
            failed_components = sorted(
                component.id
                for component in self.session.state.components.values()
                if component.application_instance_id == instance_id
                and component.status == "failed"
            )
            return ExperimentResult(
                metrics,
                objective_values,
                feasible,
                {
                    "successful": application_success,
                    "success_rate": 1.0 if application_success else 0.0,
                    "failed_instances": [] if application_success else [instance_id],
                    "failed_components": failed_components,
                    **_retry_evidence(self.session, {instance_id}),
                },
            )
        finally:
            self.session.unsubscribe(dispatcher)

    async def run_workload(
        self,
        system: SystemSpec,
        workload: WorkloadSpec,
        policy: Policy,
        *,
        timeout: float = 60.0,
        setup: SetupHook | None = None,
    ) -> ExperimentResult:
        """Run a timed workload with equivalent Real/Twin arrival semantics."""

        self.metric_set.reset()
        await self.session.start()
        await self.session.register_system(system)
        for app in workload.applications:
            await self.session.register_application(app)
        if setup is not None:
            await setup(self.session)

        dispatcher = PolicyDispatcher(self.session, [policy])
        self.session.subscribe(dispatcher)
        base_time = self.session.clock.now()
        instance_ids: list[str] = []
        sequence = 0
        try:
            for arrival in sorted(workload.arrivals, key=lambda item: item.at_s):
                app = workload.application(arrival.application_id)
                for _ in range(arrival.count):
                    sequence += 1
                    instance_id = f"{app.id}-arrival-{sequence:06d}"
                    instance_ids.append(
                        await self.session.schedule_application(
                            app,
                            at=base_time + arrival.at_s,
                            instance_id=instance_id,
                        )
                    )

            if hasattr(self.session.backend, "advance"):
                await self.session.advance()
            else:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            self.session.wait_for(
                                lambda event, state, expected=instance_id: (
                                    event.kind == EventKind.APPLICATION_COMPLETED
                                    and event.payload.get("instance_id") == expected
                                )
                            )
                            for instance_id in instance_ids
                        )
                    ),
                    timeout=timeout,
                )

            completion_events = {
                str(event.payload.get("instance_id")): event
                for event in self.session.event_log
                if event.kind == EventKind.APPLICATION_COMPLETED
            }
            completed = set(completion_events)
            missing = [item for item in instance_ids if item not in completed]
            if missing:
                raise RuntimeError(
                    "workload did not complete all application instances: "
                    + ", ".join(missing)
                )

            metrics = self.metric_set.results()
            numeric = {
                key: float(value) for key, value in metrics.items() if value is not None
            }
            objective_values = tuple(
                objective.value(numeric) for objective in self.objectives
            )
            failed_instances = sorted(
                instance_id
                for instance_id, event in completion_events.items()
                if not bool(event.payload.get("success", True))
            )
            success_rate = (
                (len(instance_ids) - len(failed_instances)) / len(instance_ids)
                if instance_ids
                else 1.0
            )
            constraints_satisfied = all(
                constraint.satisfied(numeric) for constraint in self.constraints
            )
            successful = not failed_instances
            feasible = successful and constraints_satisfied
            failed_components = sorted(
                component.id
                for component in self.session.state.components.values()
                if component.status == "failed"
            )
            return ExperimentResult(
                metrics,
                objective_values,
                feasible,
                {
                    "successful": successful,
                    "success_rate": success_rate,
                    "failed_instances": failed_instances,
                    "failed_components": failed_components,
                    **_retry_evidence(self.session, set(instance_ids)),
                },
            )
        finally:
            self.session.unsubscribe(dispatcher)

    def run_sync(self, *args, **kwargs) -> ExperimentResult:
        return asyncio.run(self.run(*args, **kwargs))
