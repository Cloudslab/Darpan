"""Event-driven max-min CPU sharing for opt-in Twin resources."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from darpan.core.state import ContinuumState

_EPSILON = 1e-12


@dataclass(slots=True)
class TwinComputeJob:
    """A unit of CPU work whose progress depends on its current fair share."""

    id: str
    node_id: str
    max_cpu: float
    work_cpu_seconds: float
    started_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    remaining_work: float | None = None

    def __post_init__(self) -> None:
        if self.max_cpu <= 0:
            raise ValueError("compute job max_cpu must be positive")
        if self.work_cpu_seconds < 0:
            raise ValueError("compute job work cannot be negative")
        if self.remaining_work is None:
            self.remaining_work = float(self.work_cpu_seconds)


class MaxMinComputeScheduler:
    """Fluid CPU scheduler with per-job caps and node capacity constraints.

    The scheduler only models resources that explicitly opt into
    ``scheduling: fair``. Other resources retain Darpan's strict reservation
    semantics in :class:`QueueDelayModel`.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, TwinComputeJob] = {}
        self._rates: dict[str, float] = {}
        self._last_time = 0.0
        self.generation = 0

    @property
    def jobs(self) -> tuple[TwinComputeJob, ...]:
        return tuple(self._jobs.values())

    @staticmethod
    def _capacity(state: ContinuumState, node_id: str) -> float:
        node = state.nodes.get(node_id)
        if node is None or node.status != "online":
            return 0.0
        return node.effective_resource_capacity("cpu")

    @classmethod
    def max_min_rates(
        cls,
        jobs: tuple[TwinComputeJob, ...],
        *,
        state: ContinuumState,
    ) -> dict[str, float]:
        """Return max-min fair CPU rates respecting each job's CPU cap."""

        rates: dict[str, float] = {job.id: 0.0 for job in jobs}
        by_node: dict[str, list[TwinComputeJob]] = {}
        for job in jobs:
            by_node.setdefault(job.node_id, []).append(job)

        for node_id, node_jobs in by_node.items():
            capacity = cls._capacity(state, node_id)
            active = {job.id: job for job in node_jobs}
            remaining = capacity
            while active and remaining > _EPSILON:
                fair_share = remaining / len(active)
                capped = [
                    job
                    for job in active.values()
                    if job.max_cpu <= fair_share + _EPSILON
                ]
                if not capped:
                    for job in active.values():
                        rates[job.id] = fair_share
                    break
                for job in capped:
                    granted = min(job.max_cpu, remaining)
                    rates[job.id] = granted
                    remaining -= granted
                    active.pop(job.id, None)
        return rates

    def _advance_existing(self, now: float) -> None:
        if now < self._last_time - _EPSILON:
            raise ValueError("cannot move compute scheduler backwards")
        elapsed = max(0.0, now - self._last_time)
        if elapsed > 0:
            for job_id, job in self._jobs.items():
                remaining = float(job.remaining_work or 0.0)
                if math.isfinite(remaining):
                    job.remaining_work = max(
                        0.0,
                        remaining - self._rates.get(job_id, 0.0) * elapsed,
                    )
        self._last_time = now

    def _reschedule(self, *, state: ContinuumState) -> None:
        self._rates = self.max_min_rates(self.jobs, state=state)
        self.generation += 1

    def add(
        self,
        job: TwinComputeJob,
        *,
        now: float,
        state: ContinuumState,
    ) -> None:
        self._advance_existing(now)
        if job.id in self._jobs:
            raise ValueError(f"duplicate compute job: {job.id}")
        self._jobs[job.id] = job
        self._reschedule(state=state)

    def capacity_changed(self, *, now: float, state: ContinuumState) -> None:
        self._advance_existing(now)
        self._reschedule(state=state)

    def advance(
        self,
        *,
        now: float,
        state: ContinuumState,
    ) -> tuple[TwinComputeJob, ...]:
        self._advance_existing(now)
        completed = tuple(
            job
            for job in self._jobs.values()
            if math.isfinite(float(job.remaining_work or 0.0))
            and float(job.remaining_work or 0.0) <= _EPSILON
        )
        for job in completed:
            self._jobs.pop(job.id, None)
            self._rates.pop(job.id, None)
        self._reschedule(state=state)
        return completed

    def next_event_at(self, *, now: float) -> float | None:
        finish_times = []
        for job_id, job in self._jobs.items():
            remaining = float(job.remaining_work or 0.0)
            rate = self._rates.get(job_id, 0.0)
            if not math.isfinite(remaining) or rate <= _EPSILON:
                continue
            finish_times.append(now + remaining / rate)
        return min(finish_times, default=None)

    def cancel(
        self,
        job_ids: set[str],
        *,
        now: float,
        state: ContinuumState,
    ) -> tuple[TwinComputeJob, ...]:
        self._advance_existing(now)
        cancelled = tuple(
            self._jobs.pop(job_id)
            for job_id in tuple(job_ids)
            if job_id in self._jobs
        )
        for job in cancelled:
            self._rates.pop(job.id, None)
        self._reschedule(state=state)
        return cancelled

    def blocked(self, state: ContinuumState) -> tuple[TwinComputeJob, ...]:
        return tuple(
            job for job in self._jobs.values() if self._capacity(state, job.node_id) <= 0
        )

    def clear(self) -> None:
        self._jobs.clear()
        self._rates.clear()
        self._last_time = 0.0
        self.generation += 1
