from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import yaml

from darpan._version import __version__
from darpan.core.codec import load_system
from darpan.experiment.artifact import require_fresh_artifact_directory, seal_artifact
from darpan.experiment.provenance import collect_provenance
from darpan.experiment.recorder import ResultRecorder
from darpan.runtime.real.agent import AgentServer
from darpan.runtime.real.cluster.acceptance import (
    accept_cluster,
    build_deployment_receipt,
)
from darpan.runtime.real.cluster.bootstrap import start_agent_over_ssh
from darpan.runtime.real.cluster.deployment import (
    BootstrapOptions,
    apply_bootstrap_plan,
    build_bootstrap_plan,
)
from darpan.runtime.real.cluster.discovery import (
    discover_cluster,
    discovered_inventory_dict,
    discovered_system_dict,
)
from darpan.runtime.real.cluster.exercise import exercise_cluster_runtime
from darpan.runtime.real.cluster.first_run import run_cluster_first_run
from darpan.runtime.real.cluster.health import check_cluster
from darpan.runtime.real.cluster.inventory import ClusterInventory
from darpan.runtime.real.cluster.readiness import check_deployment_readiness
from darpan.runtime.real.cluster.validation import validate_cluster
from darpan.runtime.real.transport import server_tls_context


async def _agent(args) -> None:
    token = None
    if args.token_env is not None:
        if args.token_env not in os.environ:
            raise RuntimeError(f"token environment variable is not set: {args.token_env}")
        token = os.environ[args.token_env]
    tls = None
    if (args.tls_cert is None) != (args.tls_key is None):
        raise ValueError("--tls-cert and --tls-key must be supplied together")
    if args.tls_cert is not None:
        tls = server_tls_context(args.tls_cert, args.tls_key)
    physical_control = None
    if args.enable_physical_control:
        from darpan.runtime.real.linux_control import LinuxPhysicalControlBackend

        physical_control = LinuxPhysicalControlBackend(
            interfaces=args.control_interface,
            cpu_max_path=args.control_cpu_max,
            allow_replace_existing_qdisc=args.allow_replace_existing_qdisc,
        )
    server = AgentServer(
        args.node_id,
        host=args.host,
        port=args.port,
        workspace_root=args.workspace_root,
        token=token,
        ssl_context=tls,
        allow_network_probe=args.enable_network_probe,
        allow_artifact_forward=args.enable_artifact_forward,
        physical_control=physical_control,
    )
    print(f"Darpan agent {args.node_id} listening on {args.host}:{args.port}")
    await server.serve_forever()


async def _bootstrap(args) -> None:
    prepared_output = (
        None
        if args.output is None
        else require_fresh_artifact_directory(args.output)
    )
    inventory = ClusterInventory.load(args.cluster)
    options = BootstrapOptions(
        service_user=args.service_user,
        install_root=args.install_root,
        config_root=args.config_root,
        data_root=args.data_root,
        package_spec=args.package or f"darpan-continuum=={__version__}",
        wheel=None if args.wheel is None else Path(args.wheel).expanduser().resolve(),
        python_command=args.python_command,
        grant_docker_group=args.grant_docker_group,
        tls_source_dir=(
            None
            if args.tls_source_dir is None
            else Path(args.tls_source_dir).expanduser().resolve()
        ),
    )
    plan = build_bootstrap_plan(inventory, options=options)
    payload: dict[str, object] = {"plan": plan.to_dict(), "applied": False}
    if args.apply:
        payload["result"] = await apply_bootstrap_plan(
            inventory,
            plan,
            options=options,
        )
        payload["applied"] = True
        await asyncio.sleep(args.wait)
        payload["agent_status"] = await check_cluster(inventory)
    print(json.dumps(payload, indent=2))
    if prepared_output is not None:
        output = prepared_output
        recorder = ResultRecorder(output)
        recorder.write_json("bootstrap.json", payload)
        cluster_source = Path(args.cluster).expanduser().resolve()
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        checksums = {"cluster.yaml": recorder.sha256(cluster_source)}
        if options.wheel is not None:
            checksums["wheel"] = recorder.sha256(options.wheel)
        if options.tls_source_dir is not None:
            checksums["tls_certificates"] = {
                node.id: recorder.sha256(options.tls_source_dir / f"{node.id}.crt")
                for node in inventory.nodes
            }
        recorder.write_json("inputs/checksums.json", checksums)
        for node_plan in plan.nodes:
            recorder.write_text(
                f"rendered/{node_plan.service_name}",
                node_plan.systemd_unit,
            )
        seal_artifact(
            output,
            schema="darpan.cluster-bootstrap-artifact/v1",
            identity={"applied": bool(args.apply)},
        )


async def _check(args) -> None:
    prepared_output = (
        None
        if args.output is None
        else require_fresh_artifact_directory(args.output)
    )
    inventory = ClusterInventory.load(args.cluster)
    system = None if args.system is None else load_system(args.system)
    report = await check_deployment_readiness(
        inventory,
        require_agent=args.require_agent,
        system=system,
        max_clock_offset_s=args.max_clock_offset,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if prepared_output is not None:
        output = prepared_output
        recorder = ResultRecorder(output)
        recorder.write_json("cluster-check.json", payload)
        cluster_source = Path(args.cluster).expanduser().resolve()
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        if args.system is not None:
            recorder.copy(Path(args.system).expanduser().resolve(), "inputs/system.yaml")
        seal_artifact(
            output,
            schema="darpan.cluster-check/v1",
            identity={"require_agent": bool(args.require_agent)},
        )
    if not report.ready:
        raise SystemExit(2)


async def _discover(args) -> None:
    output = require_fresh_artifact_directory(args.output)
    inventory = ClusterInventory.load(args.cluster)
    discovery = await discover_cluster(inventory)
    cluster_payload = discovered_inventory_dict(inventory, discovery)
    system_payload = discovered_system_dict(inventory, discovery)
    payload = discovery.to_dict()
    print(json.dumps(payload, indent=2))
    if output is not None:
        recorder = ResultRecorder(output)
        recorder.write_json("discovery.json", payload)
        recorder.write_text(
            "cluster.discovered.yaml",
            yaml.safe_dump(cluster_payload, sort_keys=False),
        )
        recorder.write_text(
            "system.discovered.yaml",
            yaml.safe_dump(system_payload, sort_keys=False),
        )
        cluster_source = Path(args.cluster).expanduser().resolve()
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        seal_artifact(
            output,
            schema="darpan.cluster-discovery/v1",
            identity={"environment_fingerprint": discovery.environment_fingerprint},
        )


async def _accept(args) -> None:
    prepared_output = (
        None
        if args.output is None
        else require_fresh_artifact_directory(args.output)
    )
    inventory = ClusterInventory.load(args.cluster)
    system = load_system(args.system)
    command = (
        args.python_command,
        "-c",
        "import time; time.sleep(3600)",
    )
    report = await accept_cluster(
        inventory,
        system,
        source_node_id=args.source,
        target_node_id=args.target,
        command=command,
        python_command=args.python_command,
        cpu_request=args.cpu,
        timeout_s=args.timeout,
        exercise_physical_control=args.exercise_physical_control,
        exercise_link_control=args.exercise_link_control,
    )
    cluster_source = Path(args.cluster).expanduser().resolve()
    system_source = Path(args.system).expanduser().resolve()
    receipt = build_deployment_receipt(
        report,
        cluster=cluster_source,
        system=system_source,
    )
    payload = report.to_dict()
    payload["deployment_receipt"] = receipt.to_dict()
    print(json.dumps(payload, indent=2))
    if prepared_output is not None:
        output = prepared_output
        recorder = ResultRecorder(output)
        recorder.write_json("cluster-acceptance.json", report.to_dict())
        recorder.write_json("deployment-receipt.json", receipt.to_dict())
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        recorder.copy(system_source, "inputs/system.yaml")
        recorder.write_json("provenance.json", asdict(collect_provenance()))
        seal_artifact(
            output,
            schema="darpan.cluster-acceptance/v1",
            identity={
                "ready_for_study": report.ready_for_study,
                "source": args.source,
                "target": args.target,
                "deployment_fingerprint": receipt.deployment_fingerprint,
            },
        )
    if not report.ready_for_study:
        raise SystemExit(2)


async def _first_run(args) -> None:
    output = require_fresh_artifact_directory(args.output)
    inventory = ClusterInventory.load(args.cluster)
    system = load_system(args.system)
    command = (
        args.python_command,
        "-c",
        "import time; time.sleep(3600)",
    )
    report = await run_cluster_first_run(
        inventory,
        system,
        source_node_id=args.source,
        target_node_id=args.target,
        command=command,
        python_command=args.python_command,
        cpu_request=args.cpu,
        timeout_s=args.timeout,
        max_clock_offset_s=args.max_clock_offset,
        exercise_physical_control=args.exercise_physical_control,
        exercise_link_control=args.exercise_link_control,
        skip_ssh_check=args.skip_ssh_check,
    )
    cluster_source = Path(args.cluster).expanduser().resolve()
    system_source = Path(args.system).expanduser().resolve()
    receipt = None
    if report.acceptance is not None:
        receipt = build_deployment_receipt(
            report.acceptance,
            cluster=cluster_source,
            system=system_source,
            additional_capabilities=(
                ("discovery",)
                if report.ssh_check_skipped
                else ("host_readiness", "discovery")
            ),
        )
    payload = report.to_dict()
    if receipt is not None:
        payload["deployment_receipt"] = receipt.to_dict()
    print(json.dumps(payload, indent=2))
    if output is not None:
        recorder = ResultRecorder(output)
        recorder.write_json("first-run.json", report.to_dict())
        if receipt is not None:
            recorder.write_json("deployment-receipt.json", receipt.to_dict())
        if report.discovery is not None:
            recorder.write_json("discovery.json", report.discovery.to_dict())
            recorder.write_text(
                "cluster.discovered.yaml",
                yaml.safe_dump(
                    discovered_inventory_dict(inventory, report.discovery),
                    sort_keys=False,
                ),
            )
            recorder.write_text(
                "system.discovered.yaml",
                yaml.safe_dump(
                    discovered_system_dict(inventory, report.discovery),
                    sort_keys=False,
                ),
            )
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        recorder.copy(system_source, "inputs/system.yaml")
        recorder.write_json("provenance.json", asdict(collect_provenance()))
        seal_artifact(
            output,
            schema="darpan.cluster-first-run/v1",
            identity={
                "ready_for_study": report.ready_for_study,
                "ssh_check_skipped": report.ssh_check_skipped,
                "deployment_fingerprint": (
                    None if receipt is None else receipt.deployment_fingerprint
                ),
            },
        )
    if not report.ready_for_study:
        raise SystemExit(2)


async def _status(args) -> None:
    inventory = ClusterInventory.load(args.cluster)
    print(json.dumps(await check_cluster(inventory), indent=2))


async def _deploy(args) -> None:
    inventory = ClusterInventory.load(args.cluster)
    for node in inventory.nodes:
        await start_agent_over_ssh(node)
    await asyncio.sleep(args.wait)
    print(json.dumps(await check_cluster(inventory), indent=2))


async def _validate(args) -> None:
    inventory = ClusterInventory.load(args.cluster)
    system = None if args.system is None else load_system(args.system)
    report = await validate_cluster(
        inventory,
        system=system,
        exercise_data_plane=args.exercise_data_plane,
        payload_bytes=args.payload_bytes,
        timeout_s=args.timeout,
        max_clock_offset_s=getattr(args, "max_clock_offset", 1.0),
    )
    payload = report.to_dict()
    payload["cluster"] = str(Path(args.cluster).expanduser().resolve())
    payload["system"] = (
        None if args.system is None else str(Path(args.system).expanduser().resolve())
    )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        recorder = ResultRecorder(Path(args.output).expanduser().resolve())
        recorder.write_json("cluster-validation.json", payload)
        cluster_source = Path(args.cluster).expanduser().resolve()
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        checksums = {"cluster.yaml": recorder.sha256(cluster_source)}
        if args.system is not None:
            system_source = Path(args.system).expanduser().resolve()
            recorder.copy(system_source, "inputs/system.yaml")
            checksums["system.yaml"] = recorder.sha256(system_source)
        recorder.write_json("inputs/checksums.json", checksums)
    if not report.ready:
        raise SystemExit(2)




async def _exercise(args) -> None:
    inventory = ClusterInventory.load(args.cluster)
    system = load_system(args.system)
    command = (
        args.python_command,
        "-c",
        "import time; time.sleep(3600)",
    )
    report = await exercise_cluster_runtime(
        inventory,
        system,
        source_node_id=args.source,
        target_node_id=args.target,
        command=command,
        cpu_request=args.cpu,
        timeout_s=args.timeout,
    )
    payload = report.to_dict()
    payload["cluster"] = str(Path(args.cluster).expanduser().resolve())
    payload["system"] = str(Path(args.system).expanduser().resolve())
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        recorder = ResultRecorder(Path(args.output).expanduser().resolve())
        recorder.write_json("cluster-exercise.json", payload)
        cluster_source = Path(args.cluster).expanduser().resolve()
        system_source = Path(args.system).expanduser().resolve()
        recorder.copy(cluster_source, "inputs/cluster.yaml")
        recorder.copy(system_source, "inputs/system.yaml")
        recorder.write_json(
            "inputs/checksums.json",
            {
                "cluster.yaml": recorder.sha256(cluster_source),
                "system.yaml": recorder.sha256(system_source),
            },
        )
        recorder.write_json(
            "provenance.json",
            asdict(collect_provenance()),
        )
    if not report.ready:
        raise SystemExit(2)


def _add_status_parser(subparsers, name: str = "status"):
    parser = subparsers.add_parser(name, help="check a physical Darpan cluster")
    parser.add_argument("--cluster", required=True)
    parser.set_defaults(func=lambda args: asyncio.run(_status(args)))
    return parser


def _add_deploy_parser(subparsers, name: str = "deploy"):
    parser = subparsers.add_parser(name, help="start installed agents over SSH")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--wait", type=float, default=1.0)
    parser.set_defaults(func=lambda args: asyncio.run(_deploy(args)))
    return parser


def _add_validate_parser(subparsers, name: str = "validate"):
    parser = subparsers.add_parser(
        name,
        help="validate physical cluster control/data-plane readiness",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument(
        "--system",
        help="system YAML whose declared links should be exercised",
    )
    parser.add_argument(
        "--exercise-data-plane",
        action="store_true",
        help="run opt-in Agent-to-Agent probe and direct artifact checks",
    )
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--max-clock-offset",
        type=float,
        default=1.0,
        help="maximum clock offset in seconds after RTT uncertainty",
    )
    parser.add_argument("--output", help="optional durable report directory")
    parser.set_defaults(func=lambda args: asyncio.run(_validate(args)))
    return parser




def _add_exercise_parser(subparsers, name: str = "exercise"):
    parser = subparsers.add_parser(
        name,
        help="actively place, migrate, and stop a service across two Agents",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--source", required=True, help="source node id")
    parser.add_argument("--target", required=True, help="target node id")
    parser.add_argument(
        "--python-command",
        default="python3",
        help="Python executable available on both remote Agents",
    )
    parser.add_argument(
        "--cpu",
        type=float,
        help="optional CPU request used to exercise resource migration",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", help="optional durable report directory")
    parser.set_defaults(func=lambda args: asyncio.run(_exercise(args)))
    return parser



def _add_bootstrap_parser(subparsers):
    parser = subparsers.add_parser(
        "bootstrap",
        help="plan or apply an auditable systemd Agent deployment over SSH",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wheel", help="local darpan-continuum wheel to upload/install")
    parser.add_argument(
        "--package",
        help="package spec to install when --wheel is not supplied (defaults to this version)",
    )
    parser.add_argument("--service-user", default="darpan")
    parser.add_argument("--install-root", default="/opt/darpan")
    parser.add_argument("--config-root", default="/etc/darpan")
    parser.add_argument("--data-root", default="/var/lib/darpan")
    parser.add_argument("--python-command", default="python3")
    parser.add_argument("--grant-docker-group", action="store_true")
    parser.add_argument(
        "--tls-source-dir",
        help="directory containing <node-id>.crt and <node-id>.key for upload",
    )
    parser.add_argument("--wait", type=float, default=1.0)
    parser.add_argument("--output")
    parser.set_defaults(func=lambda args: asyncio.run(_bootstrap(args)))
    return parser


def _add_check_parser(subparsers):
    parser = subparsers.add_parser(
        "check",
        help="check SSH host prerequisites and optionally deployed Agent readiness",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--system")
    parser.add_argument("--require-agent", action="store_true")
    parser.add_argument("--max-clock-offset", type=float, default=1.0)
    parser.add_argument("--output")
    parser.set_defaults(func=lambda args: asyncio.run(_check(args)))
    return parser


def _add_discover_parser(subparsers):
    parser = subparsers.add_parser(
        "discover",
        help="collect Agent hardware/environment data and emit inventory/system candidates",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=lambda args: asyncio.run(_discover(args)))
    return parser


def _add_accept_parser(subparsers):
    parser = subparsers.add_parser(
        "accept",
        help="run the active first-run acceptance suite before a Physical Study",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--python-command", default="python3")
    parser.add_argument("--cpu", type=float)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--exercise-physical-control", action="store_true")
    parser.add_argument("--exercise-link-control", action="store_true")
    parser.add_argument("--output")
    parser.set_defaults(func=lambda args: asyncio.run(_accept(args)))
    return parser



def _add_first_run_parser(subparsers):
    parser = subparsers.add_parser(
        "first-run",
        help="check, discover and actively accept a newly deployed Physical cluster",
    )
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--python-command", default="python3")
    parser.add_argument("--cpu", type=float)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-clock-offset", type=float, default=1.0)
    parser.add_argument(
        "--skip-ssh-check",
        action="store_true",
        help="explicitly skip SSH prerequisite checks when provisioning is externally managed",
    )
    parser.add_argument("--exercise-physical-control", action="store_true")
    parser.add_argument("--exercise-link-control", action="store_true")
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=lambda args: asyncio.run(_first_run(args)))
    return parser

def add_parsers(subparsers) -> None:
    agent = subparsers.add_parser("agent", help="start a Darpan node agent")
    agent.add_argument("--node-id", required=True)
    agent.add_argument("--host", default="0.0.0.0")
    agent.add_argument("--port", type=int, default=8765)
    agent.add_argument(
        "--workspace-root",
        help="persistent root for component workspaces (defaults to a temporary directory)",
    )
    agent.add_argument(
        "--token-env",
        help="environment variable containing the shared agent token",
    )
    agent.add_argument("--tls-cert", help="PEM server certificate")
    agent.add_argument("--tls-key", help="PEM server private key")
    agent.add_argument(
        "--enable-network-probe",
        action="store_true",
        help="allow this agent to originate probes to other Darpan agents",
    )
    agent.add_argument(
        "--enable-artifact-forward",
        action="store_true",
        help="allow ticket-authorized direct artifact forwarding to peer agents",
    )
    agent.add_argument(
        "--enable-physical-control",
        action="store_true",
        help="enable restricted Linux tc/ip/cgroup control RPCs",
    )
    agent.add_argument(
        "--control-interface",
        action="append",
        default=[],
        help="network interface allowed for tc/ip physical control (repeatable)",
    )
    agent.add_argument(
        "--control-cpu-max",
        help="explicit cgroup cpu.max file allowed for CPU-capacity fault injection",
    )
    agent.add_argument(
        "--allow-replace-existing-qdisc",
        action="store_true",
        help="allow replacing a non-trivial root qdisc (restoration may be unverifiable)",
    )
    agent.set_defaults(func=lambda args: asyncio.run(_agent(args)))

    # Keep the original v0.1/v0.2 top-level commands as compatibility aliases.
    _add_status_parser(subparsers)
    _add_deploy_parser(subparsers)
    _add_validate_parser(subparsers, "validate-cluster")
    _add_exercise_parser(subparsers, "exercise-cluster")

    cluster = subparsers.add_parser(
        "cluster",
        help="physical cluster deployment and validation",
    )
    cluster_subparsers = cluster.add_subparsers(dest="cluster_command", required=True)
    _add_bootstrap_parser(cluster_subparsers)
    _add_check_parser(cluster_subparsers)
    _add_discover_parser(cluster_subparsers)
    _add_accept_parser(cluster_subparsers)
    _add_first_run_parser(cluster_subparsers)
    _add_status_parser(cluster_subparsers)
    _add_deploy_parser(cluster_subparsers)
    _add_validate_parser(cluster_subparsers)
    _add_exercise_parser(cluster_subparsers)
