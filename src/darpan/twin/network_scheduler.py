"""Fluid max-min network sharing for the executable Digital Twin."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from darpan.core.state import ContinuumState

_EPSILON = 1e-9


def _remaining_tolerance_bits(transfer: TwinTransfer) -> float:
    """Return a scale-aware completion tolerance for floating-point bit work."""

    return max(_EPSILON, abs(float(transfer.work_bits)) * 1e-12)


@dataclass(slots=True)
class TwinTransfer:
    id: str
    links: tuple[str, ...]
    size_bytes: int
    started_at: float
    ready_at: float
    work_bits: float
    baseline_duration_s: float
    payload: dict[str, Any] = field(default_factory=dict)
    remaining_bits: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_bits = max(0.0, float(self.work_bits))


class MaxMinNetworkScheduler:
    """Share link capacity between active end-to-end transfers.

    The scheduler is event driven. Between topology/flow changes every active
    transfer has a constant max-min fair rate. The next wake-up is therefore
    either a propagation-latency activation or a transfer completion.
    """

    def __init__(self) -> None:
        self.transfers: dict[str, TwinTransfer] = {}
        self._rates_mbps: dict[str, float] = {}
        self._last_time = 0.0
        self.generation = 0

    @staticmethod
    def _link_capacity(state: ContinuumState, link_id: str) -> float:
        link = state.links[link_id]
        if link.status != "up":
            return 0.0
        measurement = link.measurements.get("network.bandwidth_mbps") or link.measurements.get(
            "bandwidth_mbps"
        )
        if measurement is not None:
            return max(0.0, float(measurement.value))
        return max(0.0, float(link.spec.bandwidth_mbps))

    @classmethod
    def max_min_rates(
        cls,
        transfers: tuple[TwinTransfer, ...],
        state: ContinuumState,
    ) -> dict[str, float]:
        """Return progressive-filling max-min rates in Mbit/s."""

        if not transfers:
            return {}
        capacities = {
            link_id: cls._link_capacity(state, link_id)
            for transfer in transfers
            for link_id in transfer.links
        }
        rates = {transfer.id: 0.0 for transfer in transfers}
        active = {transfer.id for transfer in transfers}
        by_id = {transfer.id: transfer for transfer in transfers}
        remaining = dict(capacities)

        while active:
            counts = {
                link_id: sum(link_id in by_id[item].links for item in active)
                for link_id in remaining
            }
            increments = [
                remaining[link_id] / count
                for link_id, count in counts.items()
                if count > 0
            ]
            if not increments:
                break
            delta = max(0.0, min(increments))
            if not math.isfinite(delta):
                # Every relevant link has infinite capacity. Keep rates finite
                # enough for deterministic arithmetic while effectively making
                # transfer service immediate.
                for transfer_id in active:
                    rates[transfer_id] = float("inf")
                break
            for transfer_id in active:
                rates[transfer_id] += delta
            for link_id, count in counts.items():
                if count > 0:
                    remaining[link_id] = max(0.0, remaining[link_id] - delta * count)
            saturated = {
                link_id
                for link_id, count in counts.items()
                if count > 0 and remaining[link_id] <= _EPSILON
            }
            frozen = {
                transfer_id
                for transfer_id in active
                if saturated.intersection(by_id[transfer_id].links)
            }
            if not frozen:
                # Numerical guard: if no bottleneck was recognized, stop rather
                # than spinning forever on an infinitesimal residual.
                break
            active.difference_update(frozen)
        return rates

    def _advance_existing(self, now: float) -> None:
        if now < self._last_time - _EPSILON:
            raise ValueError("network scheduler cannot move backwards")
        elapsed = max(0.0, now - self._last_time)
        if elapsed > 0:
            for transfer_id, rate_mbps in self._rates_mbps.items():
                transfer = self.transfers.get(transfer_id)
                if transfer is None or transfer.ready_at > self._last_time + _EPSILON:
                    continue
                if math.isinf(rate_mbps):
                    transfer.remaining_bits = 0.0
                else:
                    transfer.remaining_bits = max(
                        0.0,
                        transfer.remaining_bits - rate_mbps * 1_000_000 * elapsed,
                    )
                    # At a large virtual clock, adding the sub-microsecond
                    # serialization time of a small flow can lose enough
                    # precision to leave a sub-bit residual.  Scheduling that
                    # residual may then round back to ``now`` forever.  Bound
                    # the completion tolerance by the clock's representable
                    # service quantum as well as by the transfer's work scale.
                    clock_quantum_bits = (
                        rate_mbps
                        * 1_000_000
                        * max(math.ulp(now), math.ulp(self._last_time))
                    )
                    if transfer.remaining_bits <= max(
                        _remaining_tolerance_bits(transfer),
                        2.0 * clock_quantum_bits,
                    ):
                        transfer.remaining_bits = 0.0
        self._last_time = now

    def add(self, transfer: TwinTransfer, *, now: float, state: ContinuumState) -> None:
        if transfer.id in self.transfers:
            raise ValueError(f"duplicate Twin transfer id: {transfer.id}")
        self._advance_existing(now)
        self.transfers[transfer.id] = transfer
        self._reschedule(now, state)

    def topology_changed(self, *, now: float, state: ContinuumState) -> None:
        self._advance_existing(now)
        self._reschedule(now, state)

    def advance(
        self,
        *,
        now: float,
        state: ContinuumState,
    ) -> tuple[TwinTransfer, ...]:
        self._advance_existing(now)
        completed = tuple(
            transfer
            for transfer in self.transfers.values()
            if transfer.ready_at <= now + _EPSILON
            and transfer.remaining_bits <= _remaining_tolerance_bits(transfer)
        )
        for transfer in completed:
            self.transfers.pop(transfer.id, None)
            self._rates_mbps.pop(transfer.id, None)
        self._reschedule(now, state)
        return completed

    def _reschedule(self, now: float, state: ContinuumState) -> None:
        ready = tuple(
            transfer
            for transfer in self.transfers.values()
            if transfer.ready_at <= now + _EPSILON
            and transfer.remaining_bits > _remaining_tolerance_bits(transfer)
        )
        self._rates_mbps = self.max_min_rates(ready, state)
        self.generation += 1

    def next_event_at(self, *, now: float) -> float | None:
        candidates = [
            transfer.ready_at
            for transfer in self.transfers.values()
            if transfer.ready_at > now + _EPSILON
        ]
        for transfer_id, rate_mbps in self._rates_mbps.items():
            transfer = self.transfers[transfer_id]
            if math.isinf(rate_mbps):
                candidates.append(now)
            elif rate_mbps > _EPSILON:
                candidates.append(now + transfer.remaining_bits / (rate_mbps * 1_000_000))
        if not candidates:
            return None
        return max(now, min(candidates))


    def blocked(self, state: ContinuumState) -> tuple[TwinTransfer, ...]:
        """Return transfers whose selected link/node path became unavailable."""

        blocked = []
        for transfer in self.transfers.values():
            links_blocked = any(
                link_id not in state.links or state.links[link_id].status != "up"
                for link_id in transfer.links
            )
            event_payload = transfer.payload.get("event_payload", {})
            path = event_payload.get("path", ()) if isinstance(event_payload, dict) else ()
            nodes_blocked = any(
                node_id not in state.nodes or state.nodes[node_id].status != "online"
                for node_id in path
            )
            if links_blocked or nodes_blocked:
                blocked.append(transfer)
        return tuple(blocked)

    def cancel(self, transfer_ids: set[str]) -> tuple[TwinTransfer, ...]:
        cancelled = tuple(
            transfer
            for transfer_id in transfer_ids
            if (transfer := self.transfers.pop(transfer_id, None)) is not None
        )
        for transfer in cancelled:
            self._rates_mbps.pop(transfer.id, None)
        if cancelled:
            self.generation += 1
        return cancelled

    def clear(self) -> None:
        self.transfers.clear()
        self._rates_mbps.clear()
        self.generation += 1
