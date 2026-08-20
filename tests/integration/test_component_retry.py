from __future__ import annotations

import asyncio
from pathlib import Path

from darpan import (
    ApplicationSpec,
    ComponentSpec,
    Darpan,
    FlowSpec,
    RetryPolicy,
)
from darpan.core.event import Event, EventKind
from darpan.core.protocols.executor import ExecutionResult
from darpan.core.resource import ResourceRequest, ResourceSpec
from darpan.core.topology import NodeSpec, SystemSpec
from darpan.experiment.baselines import FirstFitPolicy
from darpan.runtime.dispatch import PolicyDispatcher
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.real.executors.local import LocalExecutor


class FlakyExecutor:
    def __init__(self, *, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def execute(self, component) -> ExecutionResult:
        del component
        self.calls += 1
        return ExecutionResult(
            return_code=1 if self.calls <= self.failures else 0,
            duration_s=0.001,
        )


class ArtifactRetryExecutor(LocalExecutor):
    def __init__(self, root: Path) -> None:
        super().__init__(workspace_root=root)
        self.producer_calls = 0
        self.consumer_payload: bytes | None = None
        self.stale_seen_on_retry = False

    async def execute_in_workspace(self, component, workspace_id: str) -> ExecutionResult:
        workspace = self.workspace_path(workspace_id)
        if component.id == "producer":
            self.producer_calls += 1
            output = workspace / "out.bin"
            if self.producer_calls == 1:
                output.write_bytes(b"stale-partial")
                return ExecutionResult(return_code=1, duration_s=0.001)
            self.stale_seen_on_retry = output.exists()
            output.write_bytes(b"fresh")
            return ExecutionResult(return_code=0, duration_s=0.001)
        self.consumer_payload = (workspace / "in.bin").read_bytes()
        return ExecutionResult(return_code=0, duration_s=0.001)


async def _attach_first_fit(session) -> PolicyDispatcher:
    dispatcher = PolicyDispatcher(session, [FirstFitPolicy()])
    session.subscribe(dispatcher)
    return dispatcher


def test_real_finite_component_retries_then_succeeds() -> None:
    async def run() -> None:
        executor = FlakyExecutor(failures=1)
        session = Darpan.real(backend=RealBackend(default_executor=executor))
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-real",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    retry=RetryPolicy(max_retries=1, on=frozenset({"execution_failed"})),
                ),
            ),
        )
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            completion = await session.wait_for(
                lambda event, _state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance
                ),
                timeout=2,
            )
            component_id = f"{instance}:task"
            assert completion.event.payload["success"] is True
            assert executor.calls == 2
            assert session.state.components[component_id].status == "completed"
            assert session.state.components[component_id].attempt == 1
            retry_events = [
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_RETRYING
            ]
            assert len(retry_events) == 1
            assert retry_events[0].payload["failure_kind"] == "execution_failed"
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_twin_node_failure_retries_on_another_node() -> None:
    async def run() -> None:
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
                NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
            )
        )
        app = ApplicationSpec(
            "retry-twin",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=10,
                    retry=RetryPolicy(max_retries=1, on=frozenset({"node_offline"})),
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=1.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                1.0,
            )
            instance = await session.submit_application(app)
            await session.advance(until=20.0)
            component_id = f"{instance}:task"
            component = session.state.components[component_id]
            assert component.status == "completed"
            assert component.node_id == "fog"
            assert component.attempt == 1
            starts = [
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_STARTED
                and event.payload.get("instance_id") == component_id
            ]
            assert [event.payload["node_id"] for event in starts] == ["edge", "fog"]
            completion = [
                event
                for event in session.event_log
                if event.kind == EventKind.APPLICATION_COMPLETED
                and event.payload.get("instance_id") == instance
            ]
            assert len(completion) == 1
            assert completion[0].payload["success"] is True
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_retry_budget_exhaustion_propagates_dependency_failure() -> None:
    async def run() -> None:
        executor = FlakyExecutor(failures=99)
        session = Darpan.real(backend=RealBackend(default_executor=executor))
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-exhausted",
            components=(
                ComponentSpec("source", retry=RetryPolicy(max_retries=1)),
                ComponentSpec("sink"),
            ),
            flows=(FlowSpec("source", "sink"),),
        )
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            completion = await session.wait_for(
                lambda event, _state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance
                ),
                timeout=2,
            )
            source = session.state.components[f"{instance}:source"]
            sink = session.state.components[f"{instance}:sink"]
            assert completion.event.payload["success"] is False
            assert executor.calls == 2
            assert source.status == "failed"
            assert source.attempt == 1
            assert sink.status == "failed"
            assert len(
                [
                    event
                    for event in session.event_log
                    if event.kind == EventKind.COMPONENT_RETRYING
                ]
            ) == 1
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_failed_attempt_output_artifact_is_cleaned_before_retry(tmp_path: Path) -> None:
    async def run() -> None:
        executor = ArtifactRetryExecutor(tmp_path / "workspaces")
        session = Darpan.real(backend=RealBackend(default_executor=executor))
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-artifact",
            components=(
                ComponentSpec("producer", retry=RetryPolicy(max_retries=1)),
                ComponentSpec("consumer"),
            ),
            flows=(
                FlowSpec(
                    "producer",
                    "consumer",
                    artifact="out.bin",
                    target_path="in.bin",
                ),
            ),
        )
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            completion = await session.wait_for(
                lambda event, _state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance
                ),
                timeout=2,
            )
            assert completion.event.payload["success"] is True
            assert executor.producer_calls == 2
            assert executor.stale_seen_on_retry is False
            assert executor.consumer_payload == b"fresh"
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_experiment_result_records_retry_evidence_and_metrics() -> None:
    async def run() -> None:
        from darpan.experiment.metric import (
            ComponentRetryCount,
            ComponentRetryExhaustionCount,
            ComponentRetryRecoveryRate,
        )
        from darpan.experiment.runner import ExperimentRunner

        executor = FlakyExecutor(failures=1)
        session = Darpan.real(backend=RealBackend(default_executor=executor))
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-evidence",
            components=(ComponentSpec("task", retry=RetryPolicy(max_retries=2)),),
        )
        runner = ExperimentRunner(
            session,
            metrics=[
                ComponentRetryCount(),
                ComponentRetryRecoveryRate(),
                ComponentRetryExhaustionCount(),
            ],
        )
        try:
            result = await runner.run(system, app, FirstFitPolicy(), timeout=2)
            assert result.metrics["component_retry_count"] == 1
            assert result.metrics["component_retry_recovery_rate"] == 1.0
            assert result.metrics["component_retry_exhaustion_count"] == 0
            assert result.metadata["retry_attempts"] == 1
            assert len(result.metadata["retried_components"]) == 1
            assert result.metadata["recovered_components"] == result.metadata["retried_components"]
            assert result.metadata["retry_exhausted_components"] == []
        finally:
            await session.close()

    asyncio.run(run())


def test_twin_retry_backoff_uses_virtual_time() -> None:
    async def run() -> None:
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
                NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
            )
        )
        app = ApplicationSpec(
            "retry-backoff",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=1,
                    retry=RetryPolicy(
                        max_retries=1,
                        on=frozenset({"node_offline"}),
                        backoff_s=2.5,
                    ),
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=0.5,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                0.5,
            )
            instance = await session.submit_application(app)
            await session.advance(until=10)
            component_id = f"{instance}:task"
            retrying = next(
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_RETRYING
                and event.payload.get("instance_id") == component_id
            )
            ready = next(
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_READY
                and event.payload.get("instance_id") == component_id
                and event.payload.get("retry") is True
            )
            starts = [
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_STARTED
                and event.payload.get("instance_id") == component_id
            ]
            assert ready.event_time == retrying.event_time + 2.5
            assert starts[-1].event_time >= ready.event_time
            assert session.state.components[component_id].status == "completed"
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_real_node_loss_retry_moves_finite_task_to_surviving_node() -> None:
    class Blocking:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def execute(self, component) -> ExecutionResult:
            del component
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError("unreachable")

    class Success:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, component) -> ExecutionResult:
            del component
            self.calls += 1
            return ExecutionResult(return_code=0, duration_s=0.001)

    async def run() -> None:
        edge = Blocking()
        fog = Success()
        backend = RealBackend(executors={"edge": edge, "fog": fog})
        session = Darpan.real(backend=backend)
        system = SystemSpec(
            nodes=(
                NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),
                NodeSpec("fog", resources=(ResourceSpec("cpu", 1),)),
            )
        )
        app = ApplicationSpec(
            "retry-real-node-loss",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    retry=RetryPolicy(max_retries=1, on=frozenset({"node_offline"})),
                ),
            ),
        )
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            await asyncio.wait_for(edge.started.wait(), timeout=1)
            await session.emit(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=session.clock.now(),
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                )
            )
            await backend.node_unavailable("edge", reason="test fault")
            completion = await session.wait_for(
                lambda event, _state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance
                ),
                timeout=2,
            )
            component_id = f"{instance}:task"
            assert completion.event.payload["success"] is True
            assert edge.cancelled is True
            assert fog.calls == 1
            assert session.state.components[component_id].node_id == "fog"
            assert session.state.components[component_id].attempt == 1
            assert session.state.nodes["edge"].resources["cpu"].allocated == 0
            assert session.state.nodes["fog"].resources["cpu"].allocated == 0
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_retry_waits_ready_until_only_node_recovers() -> None:
    async def run() -> None:
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-recovery-liveness",
            components=(
                ComponentSpec(
                    "task",
                    resources=(ResourceRequest("cpu", 1),),
                    work_units=10,
                    retry=RetryPolicy(max_retries=1, on=frozenset({"node_offline"})),
                ),
            ),
        )
        session = Darpan.twin()
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_OFFLINE,
                    event_time=0.5,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                0.5,
            )
            session.backend.schedule_event(
                Event(
                    kind=EventKind.NODE_RECOVERED,
                    event_time=2.0,
                    source="test",
                    subject="edge",
                    payload={"node_id": "edge"},
                ),
                2.0,
            )
            instance = await session.submit_application(app)
            await session.advance(until=20)
            component_id = f"{instance}:task"
            component = session.state.components[component_id]
            assert component.status == "completed"
            assert component.attempt == 1
            retry_ready = next(
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_READY
                and event.payload.get("instance_id") == component_id
                and event.payload.get("retry") is True
            )
            restarted = [
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_STARTED
                and event.payload.get("instance_id") == component_id
            ][-1]
            assert retry_ready.event_time == 0.5
            assert restarted.event_time >= 2.0
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())


def test_real_retry_backoff_uses_managed_wall_clock_task() -> None:
    async def run() -> None:
        executor = FlakyExecutor(failures=1)
        session = Darpan.real(backend=RealBackend(default_executor=executor))
        system = SystemSpec(nodes=(NodeSpec("edge", resources=(ResourceSpec("cpu", 1),)),))
        app = ApplicationSpec(
            "retry-real-backoff",
            components=(
                ComponentSpec(
                    "task",
                    retry=RetryPolicy(max_retries=1, backoff_s=0.03),
                ),
            ),
        )
        await session.start()
        dispatcher = await _attach_first_fit(session)
        try:
            await session.register_system(system)
            instance = await session.submit_application(app)
            await session.wait_for(
                lambda event, _state: (
                    event.kind == EventKind.APPLICATION_COMPLETED
                    and event.payload.get("instance_id") == instance
                ),
                timeout=2,
            )
            component_id = f"{instance}:task"
            retrying = next(
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_RETRYING
                and event.payload.get("instance_id") == component_id
            )
            starts = [
                event
                for event in session.event_log
                if event.kind == EventKind.COMPONENT_STARTED
                and event.payload.get("instance_id") == component_id
            ]
            assert len(starts) == 2
            assert starts[-1].event_time - retrying.event_time >= 0.02
        finally:
            session.unsubscribe(dispatcher)
            await session.close()

    asyncio.run(run())
