"""Explicit SSH bootstrap plan for agent processes.

Darpan does not silently install software on remote machines. This helper starts
an already-installed Darpan agent over SSH, which keeps deployment predictable
and auditable.
"""

from __future__ import annotations

import asyncio
import os
import shlex

from .inventory import ClusterNode


async def start_agent_over_ssh(node: ClusterNode) -> None:
    destination = f"{node.ssh_user}@{node.host}" if node.ssh_user else node.host
    remote = (
        f"nohup darpan agent --node-id {shlex.quote(node.id)} "
        f"--host 0.0.0.0 --port {node.port} "
    )
    if node.token_env:
        remote += f"--token-env {shlex.quote(node.token_env)} "
    if node.network_probe:
        remote += "--enable-network-probe "
    if node.artifact_forward:
        remote += "--enable-artifact-forward "
    if node.physical_control:
        remote += "--enable-physical-control "
        for interface in node.control_interfaces:
            remote += f"--control-interface {shlex.quote(interface)} "
        if node.control_cpu_max is not None:
            remote += f"--control-cpu-max {shlex.quote(node.control_cpu_max)} "
        if node.allow_replace_existing_qdisc:
            remote += "--allow-replace-existing-qdisc "
    if node.tls_cert or node.tls_key:
        if not node.tls_cert or not node.tls_key:
            raise ValueError(f"node {node.id} must define both tls_cert and tls_key")
        remote += (
            f"--tls-cert {shlex.quote(node.tls_cert)} "
            f"--tls-key {shlex.quote(node.tls_key)} "
        )
    remote += ">darpan-agent.log 2>&1 &"
    connection_args = (
        (
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={os.devnull}",
        )
        if node.ssh_identity_file is None
        else (
            "-i",
            node.ssh_identity_file,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={os.devnull}",
        )
    )
    process = await asyncio.create_subprocess_exec(
        "ssh", *connection_args, "-p", str(node.ssh_port), destination, remote,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace"))
