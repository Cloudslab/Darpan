"""Portable experiment specification and preflight validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    system: str
    application: str | None = None
    workload: str | None = None
    runtime: str = "twin"
    cluster: str | None = None
    network_driver: str | None = None
    physical_control: bool = False
    policy: str = "round-robin"
    repeat: int = 1
    seed: int = 0
    timeout: float = 120.0
    metrics: tuple[str, ...] = ("application_latency_s",)
    models: tuple[str, ...] = ()
    model_snapshot: str | None = None
    scenario: str | None = None
    fidelity_tracking: bool = False
    output: str | None = None
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.runtime not in {"real", "twin"}:
            raise ValueError("experiment runtime must be 'real' or 'twin'")
        if (self.application is None) == (self.workload is None):
            raise ValueError("experiment must define exactly one of application or workload")
        if self.repeat <= 0:
            raise ValueError("experiment repeat must be positive")
        if self.timeout <= 0:
            raise ValueError("experiment timeout must be positive")
        if not self.system:
            raise ValueError("experiment system cannot be empty")
        if not self.policy:
            raise ValueError("experiment policy cannot be empty")
        if not self.metrics:
            raise ValueError("experiment must define at least one metric")
        if self.network_driver is not None and self.runtime != "real":
            raise ValueError("network_driver requires runtime: real")
        if self.physical_control and self.runtime != "real":
            raise ValueError("physical_control requires runtime: real")
        if self.physical_control and self.cluster is None:
            raise ValueError("physical_control requires a real cluster inventory")
        if self.scenario is not None and self.runtime == "real" and not self.physical_control:
            raise ValueError(
                "real scenario requires physical_control: true; canonical events are "
                "never used as a substitute for Physical Continuum control"
            )
        if self.fidelity_tracking and self.runtime != "real":
            raise ValueError("fidelity_tracking requires runtime: real")
        if (
            self.model_snapshot is not None
            and self.runtime == "real"
            and not self.fidelity_tracking
        ):
            raise ValueError(
                "model_snapshot on runtime: real requires fidelity_tracking: true"
            )

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def resolve(self, value: str) -> Path:
        """Resolve portable paths while keeping v0.1 repo-root configs working."""

        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw
        local = self.directory / raw
        if local.exists():
            return local.resolve()
        cwd_relative = Path.cwd() / raw
        if cwd_relative.exists():
            return cwd_relative.resolve()
        # Prefer the experiment-local interpretation in error messages.
        return local.resolve()

    def preflight_paths(self) -> None:
        required = [self.system]
        if self.application is not None:
            required.append(self.application)
        if self.workload is not None:
            required.append(self.workload)
        if self.runtime == "real" and self.cluster is not None:
            required.append(self.cluster)
        if self.model_snapshot is not None:
            required.append(self.model_snapshot)
        if self.scenario is not None:
            required.append(self.scenario)
        missing = [str(self.resolve(item)) for item in required if not self.resolve(item).is_file()]
        if missing:
            raise FileNotFoundError("experiment references missing files: " + ", ".join(missing))

    def to_dict(self, *, resolved: bool = False) -> dict[str, Any]:
        def path(value: str | None) -> str | None:
            if value is None:
                return None
            return str(self.resolve(value)) if resolved else value

        return {
            "system": path(self.system),
            "application": path(self.application),
            "workload": path(self.workload),
            "runtime": self.runtime,
            "cluster": path(self.cluster),
            "network_driver": self.network_driver,
            "physical_control": self.physical_control,
            "policy": self.policy,
            "repeat": self.repeat,
            "seed": self.seed,
            "timeout": self.timeout,
            "metrics": list(self.metrics),
            "models": list(self.models),
            "model_snapshot": path(self.model_snapshot),
            "scenario": path(self.scenario),
            "fidelity_tracking": self.fidelity_tracking,
            "output": path(self.output),
        }

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("experiment configuration must be a mapping")
        spec = cls(
            system=str(data["system"]),
            application=(
                None if data.get("application") is None else str(data["application"])
            ),
            workload=None if data.get("workload") is None else str(data["workload"]),
            runtime=str(data.get("runtime", "twin")),
            cluster=None if data.get("cluster") is None else str(data["cluster"]),
            network_driver=(
                None
                if data.get("network_driver") is None
                else str(data["network_driver"])
            ),
            physical_control=bool(data.get("physical_control", False)),
            policy=str(data.get("policy", "round-robin")),
            repeat=int(data.get("repeat", 1)),
            seed=int(data.get("seed", 0)),
            timeout=float(data.get("timeout", 120.0)),
            metrics=tuple(str(item) for item in data.get("metrics", ["application_latency_s"])),
            models=tuple(str(item) for item in data.get("models", [])),
            model_snapshot=(
                None
                if data.get("model_snapshot") is None
                else str(data["model_snapshot"])
            ),
            scenario=None if data.get("scenario") is None else str(data["scenario"]),
            fidelity_tracking=bool(data.get("fidelity_tracking", False)),
            output=None if data.get("output") is None else str(data["output"]),
            source=source,
        )
        spec.preflight_paths()
        return spec
