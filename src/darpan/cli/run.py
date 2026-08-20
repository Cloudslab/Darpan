"""Run experiments from one portable configuration or legacy CLI flags."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from darpan.api import Darpan
from darpan.core.codec import load_application, load_system, load_workload
from darpan.core.loading import materialize_plugin
from darpan.experiment.baselines import (
    FirstFitPolicy,
    LatencyAwarePolicy,
    RoundRobinPolicy,
)
from darpan.experiment.fidelity import FidelitySuiteTracker
from darpan.experiment.metric import (
    ApplicationLatency,
    ApplicationSuccessRate,
    ComponentRetryCount,
    ComponentRetryExhaustionCount,
    ComponentRetryRecoveryRate,
    MigrationDowntime,
    RestartDowntime,
    RetryDowntime,
    RouteChangeLatency,
    ScaleConvergence,
)
from darpan.experiment.reproducibility import (
    ExperimentArtifactRecorder,
    seed_everything,
)
from darpan.experiment.runner import ExperimentRunner
from darpan.experiment.scenario_plan import ScenarioPlan
from darpan.experiment.spec import ExperimentSpec
from darpan.runtime.real.cluster.inventory import ClusterInventory
from darpan.runtime.real.cluster.session import session_from_inventory
from darpan.runtime.session import Session
from darpan.twin.models.artifact import ArtifactSizeModel
from darpan.twin.models.execution import ExecutionTimeModel
from darpan.twin.models.network import NetworkDelayModel
from darpan.twin.models.queue import QueueDelayModel
from darpan.twin.models.registry import ModelRegistry


def _policy(name: str, *, search_path: Path | None = None):
    if name == "first-fit":
        return FirstFitPolicy()
    if name == "round-robin":
        return RoundRobinPolicy()
    if name == "latency-aware":
        return LatencyAwarePolicy()
    return materialize_plugin(name, search_path=search_path)


def _metric(name: str, *, search_path: Path | None = None):
    if name == "application_latency_s":
        return ApplicationLatency()
    if name == "application_success_rate":
        return ApplicationSuccessRate()
    if name == "component_retry_count":
        return ComponentRetryCount()
    if name == "component_retry_recovery_rate":
        return ComponentRetryRecoveryRate()
    if name == "component_retry_exhaustion_count":
        return ComponentRetryExhaustionCount()
    if name == "migration_downtime_s":
        return MigrationDowntime()
    if name == "restart_downtime_s":
        return RestartDowntime()
    if name == "retry_downtime_s":
        return RetryDowntime()
    if name == "scale_convergence_s":
        return ScaleConvergence()
    if name == "route_change_s":
        return RouteChangeLatency()
    return materialize_plugin(name, search_path=search_path)


def _default_models() -> ModelRegistry:
    return ModelRegistry(
        [
            ExecutionTimeModel(),
            ArtifactSizeModel(),
            NetworkDelayModel(),
            QueueDelayModel(),
        ]
    )


def _load_model_states(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("model snapshot must contain a JSON object")
    if "models" in payload:
        payload = payload["models"]
    elif "model_states" in payload:
        payload = payload["model_states"]
    if not isinstance(payload, dict):
        raise ValueError("model snapshot models/model_states must be a mapping")
    return dict(payload)


def _models(
    names: tuple[str, ...],
    *,
    search_path: Path,
    snapshot: Path | None = None,
) -> ModelRegistry | None:
    if not names and snapshot is None:
        return None
    registry = _default_models()
    for name in names:
        registry.register(materialize_plugin(name, search_path=search_path), replace=True)
    if snapshot is not None:
        registry.restore(_load_model_states(snapshot), strict=True)
    return registry


def _real_session(
    cluster: str | Path | None,
    *,
    system=None,
    network_driver=None,
    physical_control: bool = False,
) -> Session:
    if cluster is None:
        from darpan.runtime.real.backend import RealBackend

        return Darpan.real(backend=RealBackend(network_driver=network_driver))
    inventory = ClusterInventory.load(cluster)
    return session_from_inventory(
        inventory,
        system=system,
        network_driver=network_driver,
        physical_control=physical_control,
    )


def _session(spec: ExperimentSpec, *, system=None) -> Session:
    snapshot = (
        None if spec.model_snapshot is None else spec.resolve(spec.model_snapshot)
    )
    if spec.runtime == "twin":
        return Darpan.twin(
            models=_models(
                spec.models,
                search_path=spec.directory,
                snapshot=snapshot,
            )
        )
    cluster = None if spec.cluster is None else spec.resolve(spec.cluster)
    network_driver = (
        None
        if spec.network_driver is None
        else "linux-agent"
        if spec.network_driver == "linux-agent"
        else materialize_plugin(spec.network_driver, search_path=spec.directory)
    )
    return _real_session(
        cluster,
        system=system,
        network_driver=network_driver,
        physical_control=spec.physical_control,
    )


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key].append(float(value))
    return {
        key: {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
        for key, values in numeric.items()
        if values
    }


async def _restore_physical_control(session: Session) -> dict[str, Any] | None:
    restore = getattr(session.backend, "restore_physical_control", None)
    if restore is None:
        return None
    result = restore()
    if hasattr(result, "__await__"):
        await result
    report = getattr(session.backend, "physical_control_report", None)
    return None if report is None else dict(report)


async def run_experiment_spec(
    spec: ExperimentSpec,
    *,
    emit_output: bool = True,
) -> dict[str, Any]:
    system = load_system(spec.resolve(spec.system))
    application = (
        None
        if spec.application is None
        else load_application(spec.resolve(spec.application))
    )
    workload = (
        None if spec.workload is None else load_workload(spec.resolve(spec.workload))
    )
    results: list[dict[str, Any]] = []
    artifact_recorder = (
        None
        if spec.output is None
        else ExperimentArtifactRecorder(spec.resolve(spec.output), spec)
    )

    for run_index in range(spec.repeat):
        run_seed = spec.seed + run_index
        seed_everything(run_seed)
        session = _session(spec, system=system)
        fidelity_models = None
        fidelity_tracker = None
        if spec.fidelity_tracking:
            fidelity_models = _models(
                spec.models,
                search_path=spec.directory,
                snapshot=(
                    None
                    if spec.model_snapshot is None
                    else spec.resolve(spec.model_snapshot)
                ),
            )
            if fidelity_models is None:
                fidelity_models = _default_models()
            fidelity_tracker = FidelitySuiteTracker(fidelity_models)
            session.subscribe(fidelity_tracker.observe)
        runner = ExperimentRunner(
            session,
            metrics=[_metric(item, search_path=spec.directory) for item in spec.metrics],
        )
        scenario = (
            None
            if spec.scenario is None
            else ScenarioPlan.load(spec.resolve(spec.scenario))
        )
        setup = None if scenario is None else scenario.apply
        run_payload: dict[str, Any] | None = None
        try:
            policy = _policy(spec.policy, search_path=spec.directory)
            if workload is not None:
                result = await runner.run_workload(
                    system,
                    workload,
                    policy,
                    timeout=spec.timeout,
                    setup=setup,
                )
            else:
                assert application is not None
                result = await runner.run(
                    system,
                    application,
                    policy,
                    timeout=spec.timeout,
                    setup=setup,
                )
            run_payload = dict(result.metrics)
            run_payload["_experiment"] = {
                "feasible": result.feasible,
                "objectives": list(result.objectives),
                **dict(result.metadata),
            }
            fidelity_payload = None
            if fidelity_tracker is not None and fidelity_models is not None:
                fidelity_payload = {
                    **fidelity_tracker.payload(),
                    "models": fidelity_models.snapshot(),
                }
                run_payload["fidelity"] = fidelity_payload
            results.append(run_payload)
            physical_control_payload = await _restore_physical_control(session)
            if physical_control_payload is not None:
                run_payload["_physical_control"] = physical_control_payload
            if artifact_recorder is not None:
                extra_json = (
                    {}
                    if fidelity_payload is None
                    else {"fidelity.json": fidelity_payload}
                )
                if physical_control_payload is not None:
                    extra_json["physical-control.json"] = physical_control_payload
                artifact_recorder.record_run(
                    index=run_index + 1,
                    seed=run_seed,
                    session=session,
                    result=run_payload,
                    extra_json=extra_json or None,
                )
        except Exception as exc:
            failure: BaseException = exc
            physical_control_payload = None
            try:
                physical_control_payload = await _restore_physical_control(session)
            except BaseException as restore_exc:
                failure = ExceptionGroup(
                    "experiment and Physical restoration both failed",
                    [exc, restore_exc],
                )
                report = getattr(session.backend, "physical_control_report", None)
                physical_control_payload = None if report is None else dict(report)
            if artifact_recorder is not None:
                artifact_recorder.record_failed_run(
                    index=run_index + 1,
                    seed=run_seed,
                    session=session,
                    error=failure,
                    extra_json=(
                        None
                        if physical_control_payload is None
                        else {"physical-control.json": physical_control_payload}
                    ),
                )
            if failure is exc:
                raise
            raise failure from exc
        finally:
            await session.close()

    output: dict[str, Any]
    if len(results) == 1:
        output = results[0]
    else:
        output = {"runs": results, "aggregate": _aggregate(results)}
    if artifact_recorder is not None:
        artifact_recorder.record_summary(output)
    if emit_output:
        print(json.dumps(output, indent=2))
    return output


async def _run_legacy(args: argparse.Namespace) -> None:
    system = load_system(args.system)
    application = load_application(args.app)
    session = (
        Darpan.twin()
        if args.runtime == "twin"
        else _real_session(args.cluster, system=system)
    )
    runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
    try:
        result = await runner.run(
            system,
            application,
            _policy(args.policy),
            timeout=args.timeout,
        )
        print(json.dumps(dict(result.metrics), indent=2))
    finally:
        await session.close()


async def _run(args: argparse.Namespace) -> None:
    if args.experiment is not None:
        if args.system is not None or args.app is not None:
            raise ValueError("do not mix experiment.yaml with --system/--app")
        await run_experiment_spec(ExperimentSpec.load(args.experiment))
        return
    if args.system is None or args.app is None:
        raise ValueError("provide experiment.yaml or both --system and --app")
    await _run_legacy(args)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("run", help="run an application on real or Twin runtime")
    parser.add_argument("experiment", nargs="?", help="portable experiment YAML")
    parser.add_argument("--system")
    parser.add_argument("--app")
    parser.add_argument("--runtime", choices=["real", "twin"], default="twin")
    parser.add_argument("--cluster", help="cluster inventory for real runtime")
    parser.add_argument("--policy", default="round-robin")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.set_defaults(func=lambda args: asyncio.run(_run(args)))
