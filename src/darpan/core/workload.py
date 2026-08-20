"""Workload and arrival specifications."""

from __future__ import annotations

from dataclasses import dataclass

from .application import ApplicationSpec


@dataclass(frozen=True, slots=True)
class ArrivalSpec:
    application_id: str
    at_s: float = 0.0
    count: int = 1

    def __post_init__(self) -> None:
        if self.at_s < 0 or self.count <= 0:
            raise ValueError("invalid workload arrival")


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    applications: tuple[ApplicationSpec, ...]
    arrivals: tuple[ArrivalSpec, ...] = ()
    name: str = "workload"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("workload name cannot be empty")
        if not self.applications:
            raise ValueError("workload must define at least one application")

        app_ids = [app.id for app in self.applications]
        if len(app_ids) != len(set(app_ids)):
            raise ValueError("workload contains duplicate application ids")

        if not self.arrivals:
            raise ValueError("workload must define at least one arrival")
        known = set(app_ids)
        unknown = sorted(
            {
                arrival.application_id
                for arrival in self.arrivals
                if arrival.application_id not in known
            }
        )
        if unknown:
            raise ValueError(
                "workload arrivals reference unknown applications: " + ", ".join(unknown)
            )

    def application(self, app_id: str) -> ApplicationSpec:
        for app in self.applications:
            if app.id == app_id:
                return app
        raise KeyError(app_id)
