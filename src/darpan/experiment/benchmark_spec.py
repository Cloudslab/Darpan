"""Portable specification for seed-matched paired experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PairedExperimentBenchmarkSpec:
    baseline: str
    candidate: str
    metric: str
    higher_is_better: bool = True
    require_success: bool = True
    seed: int = 0
    repeat: int = 10
    seeds: tuple[int, ...] = ()
    bootstrap_resamples: int = 2000
    output: str | None = None
    source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.baseline or not self.candidate:
            raise ValueError("benchmark baseline/candidate cannot be empty")
        if not self.metric:
            raise ValueError("benchmark metric cannot be empty")
        if self.repeat <= 0:
            raise ValueError("benchmark repeat must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("benchmark bootstrap_resamples must be positive")
        if self.seeds and len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be unique")

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def resolve(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.directory / raw).resolve()

    @property
    def run_seeds(self) -> tuple[int, ...]:
        if self.seeds:
            return self.seeds
        return tuple(self.seed + index for index in range(self.repeat))

    @classmethod
    def load(cls, path: str | Path) -> PairedExperimentBenchmarkSpec:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("benchmark configuration must be a mapping")
        spec = cls(
            baseline=str(data["baseline"]),
            candidate=str(data["candidate"]),
            metric=str(data["metric"]),
            higher_is_better=bool(data.get("higher_is_better", True)),
            require_success=bool(data.get("require_success", True)),
            seed=int(data.get("seed", 0)),
            repeat=int(data.get("repeat", 10)),
            seeds=tuple(int(item) for item in data.get("seeds", ())),
            bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
            output=None if data.get("output") is None else str(data["output"]),
            source=source,
        )
        for item in (spec.baseline, spec.candidate):
            if not spec.resolve(item).is_file():
                resolved = spec.resolve(item)
                raise FileNotFoundError(
                    f"benchmark experiment does not exist: {resolved}"
                )
        return spec
