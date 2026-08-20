"""Cluster inventory loaded from a small YAML file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ClusterNode:
    id: str
    host: str
    port: int = 8765
    ssh_user: str | None = None
    ssh_port: int = 22
    ssh_identity_file: str | None = None
    token_env: str | None = None
    tls_ca: str | None = None
    tls_server_name: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None
    network_probe: bool = False
    artifact_forward: bool = False
    physical_control: bool = False
    control_interfaces: tuple[str, ...] = ()
    control_cpu_max: str | None = None
    allow_replace_existing_qdisc: bool = False
    labels: Mapping[str, str] = field(default_factory=dict)

    def token(self) -> str | None:
        if self.token_env is None:
            return None
        try:
            return os.environ[self.token_env]
        except KeyError as exc:
            raise RuntimeError(
                f"cluster node {self.id} requires unset token environment variable "
                f"{self.token_env}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ClusterInventory:
    nodes: tuple[ClusterNode, ...]
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = tuple(node.id for node in self.nodes)
        if len(set(ids)) != len(ids):
            raise ValueError("cluster inventory node ids must be unique")
        for node in self.nodes:
            if not node.id or not node.host:
                raise ValueError("cluster inventory node id/host cannot be empty")
            if not 1 <= node.port <= 65535:
                raise ValueError(
                    f"cluster node {node.id!r} port must be in [1, 65535]"
                )
            if not 1 <= node.ssh_port <= 65535:
                raise ValueError(
                    f"cluster node {node.id!r} ssh_port must be in [1, 65535]"
                )
            if (node.tls_cert is None) != (node.tls_key is None):
                raise ValueError(
                    f"cluster node {node.id!r} tls_cert/tls_key must be paired"
                )

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def client(self, node: ClusterNode):
        from ..transport import AgentClient, client_tls_context_from_pem

        ca_pem = self.tls_ca_pem(node)
        context = None if ca_pem is None else client_tls_context_from_pem(ca_pem)
        return AgentClient(
            node.host,
            node.port,
            token=node.token(),
            ssl_context=context,
            server_hostname=(node.tls_server_name or node.host) if context else None,
            tls_ca_pem=ca_pem,
        )

    def tls_ca_pem(self, node: ClusterNode) -> str | None:
        if node.tls_ca is None:
            return None
        ca = Path(node.tls_ca).expanduser()
        if not ca.is_absolute():
            ca = self.directory / ca
        return ca.resolve().read_text(encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ClusterInventory:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        nodes = tuple(
            ClusterNode(
                id=str(item["id"]),
                host=str(item["host"]),
                port=int(item.get("port", 8765)),
                ssh_user=item.get("ssh_user"),
                ssh_port=int(item.get("ssh_port", 22)),
                ssh_identity_file=(
                    None
                    if item.get("ssh_identity_file") is None
                    else str(item["ssh_identity_file"])
                ),
                token_env=item.get("token_env"),
                tls_ca=item.get("tls_ca"),
                tls_server_name=item.get("tls_server_name"),
                tls_cert=item.get("tls_cert"),
                tls_key=item.get("tls_key"),
                network_probe=bool(item.get("network_probe", False)),
                artifact_forward=bool(item.get("artifact_forward", False)),
                physical_control=bool(item.get("physical_control", False)),
                control_interfaces=tuple(
                    str(value) for value in item.get("control_interfaces", [])
                ),
                control_cpu_max=(
                    None
                    if item.get("control_cpu_max") is None
                    else str(item["control_cpu_max"])
                ),
                allow_replace_existing_qdisc=bool(
                    item.get("allow_replace_existing_qdisc", False)
                ),
                labels=dict(item.get("labels", {})),
            )
            for item in data.get("nodes", [])
        )
        return cls(nodes, source=source)
