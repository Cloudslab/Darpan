"""Controller-side physical scenario control and restoration evidence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from darpan.core.event import Event, EventKind
from darpan.core.state import ContinuumState


class PhysicalControlDriver(Protocol):
    """Apply canonical scenario changes to the Physical Continuum before commit."""

    async def prepare_event(self, event: Event, state: ContinuumState) -> None: ...

    async def restore(self) -> None: ...

    def report(self) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class AgentPhysicalControlDriver:
    """Physical scenario driver backed by the restricted Darpan Agent RPC.

    ``node.offline`` is implemented as *workload-plane unavailability*: the
    Agent management endpoint remains reachable so the controller can restore
    the experiment deterministically. This is not claimed to be host power-off.
    A true power/IPMI fault can be supplied as another PhysicalControlDriver.
    """

    clients: Mapping[str, Any]
    actions: list[dict[str, Any]] = field(default_factory=list)
    restore_results: dict[str, Any] = field(default_factory=dict)
    restore_verified: bool | None = None
    lease_s: float = 60.0
    _heartbeat: asyncio.Task | None = field(default=None, init=False, repr=False)
    _heartbeat_cycles: int = field(default=0, init=False, repr=False)
    _heartbeat_successes: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _heartbeat_failures: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _continuity_violations: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _auto_restore_baseline: dict[str, str | None] = field(
        default_factory=dict, init=False, repr=False
    )

    def _record_heartbeat_failure(
        self, *, node_id: str, operation: str, error: BaseException
    ) -> None:
        self._heartbeat_failures.append(
            {
                "node_id": node_id,
                "operation": operation,
                "cycle": self._heartbeat_cycles,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        # Keep the receipt bounded during long experiments while retaining the
        # most recent evidence needed to diagnose a control-plane disruption.
        del self._heartbeat_failures[:-100]

    @staticmethod
    def _auto_restore_signature(inspect: Mapping[str, Any]) -> str | None:
        payload = inspect.get("last_auto_restore")
        if payload is None:
            return None
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    async def _capture_auto_restore_baseline(self) -> None:
        for node_id, client in self.clients.items():
            inspect_fn = getattr(client, "control_inspect", None)
            if inspect_fn is None:
                self._auto_restore_baseline[node_id] = None
                continue
            try:
                inspect = await inspect_fn()
            except Exception as exc:
                self._record_heartbeat_failure(
                    node_id=node_id, operation="initial_inspect", error=exc
                )
                self._auto_restore_baseline[node_id] = None
                continue
            self._auto_restore_baseline[node_id] = self._auto_restore_signature(inspect)

    async def _renew_all_leases(self, *, required: bool) -> bool:
        items = tuple(self.clients.items())
        results = await asyncio.gather(
            *(client.control_lease(lease_s=self.lease_s) for _, client in items),
            return_exceptions=True,
        )
        failures = []
        for (node_id, _), result in zip(items, results, strict=True):
            if isinstance(result, BaseException):
                failures.append((node_id, result))
                self._record_heartbeat_failure(
                    node_id=node_id, operation="lease_renewal", error=result
                )
            else:
                self._heartbeat_successes[node_id] = self._heartbeat_successes.get(node_id, 0) + 1
        if required and failures:
            rendered = "; ".join(
                f"{node_id}: {type(error).__name__}: {error}" for node_id, error in failures
            )
            raise RuntimeError(f"initial physical-control lease failed: {rendered}")
        return not failures

    async def _inspect_lease_continuity(self) -> None:
        items = tuple(self.clients.items())
        available = tuple(
            (node_id, client, getattr(client, "control_inspect", None))
            for node_id, client in items
            if getattr(client, "control_inspect", None) is not None
        )
        if not available:
            return
        results = await asyncio.gather(
            *(inspect_fn() for _, _, inspect_fn in available),
            return_exceptions=True,
        )
        existing = {json.dumps(item, sort_keys=True) for item in self._continuity_violations}
        for (node_id, _, _), result in zip(available, results, strict=True):
            if isinstance(result, BaseException):
                self._record_heartbeat_failure(
                    node_id=node_id, operation="continuity_inspect", error=result
                )
                continue
            reasons = []
            if not bool(result.get("lease_active", False)):
                reasons.append("lease_inactive_after_renewal")
            baseline = self._auto_restore_baseline.get(node_id)
            current = self._auto_restore_signature(result)
            if current != baseline:
                reasons.append("new_agent_auto_restore")
            for reason in reasons:
                violation = {
                    "node_id": node_id,
                    "cycle": self._heartbeat_cycles,
                    "reason": reason,
                }
                signature = json.dumps(violation, sort_keys=True)
                if signature not in existing:
                    self._continuity_violations.append(violation)
                    existing.add(signature)

    async def start(self) -> None:
        if self._heartbeat is not None:
            return
        await self._capture_auto_restore_baseline()
        await self._renew_all_leases(required=True)

        async def heartbeat() -> None:
            try:
                delay_s = self.lease_s / 4.0
                while True:
                    await asyncio.sleep(delay_s)
                    self._heartbeat_cycles += 1
                    renewed = await self._renew_all_leases(required=False)
                    if renewed:
                        await self._inspect_lease_continuity()
                        delay_s = self.lease_s / 4.0
                    else:
                        # A transient RPC failure must not terminate the entire
                        # heartbeat. Retry promptly while the previous lease is
                        # still live; the Agent remains fail-safe if connectivity
                        # is genuinely lost for the full lease duration.
                        delay_s = min(1.0, self.lease_s / 10.0)
            except asyncio.CancelledError:
                raise

        self._heartbeat = asyncio.create_task(heartbeat())

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat is None:
            return
        self._heartbeat.cancel()
        await asyncio.gather(self._heartbeat, return_exceptions=True)
        self._heartbeat = None

    async def _record(self, event: Event, node_id: str, operation) -> None:
        started = perf_counter()
        try:
            details = await operation()
        except Exception as exc:
            self.actions.append(
                {
                    "event_id": event.id,
                    "event_kind": event.kind,
                    "node_id": node_id,
                    "ok": False,
                    "duration_s": max(0.0, perf_counter() - started),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self.actions.append(
            {
                "event_id": event.id,
                "event_kind": event.kind,
                "node_id": node_id,
                "ok": True,
                "duration_s": max(0.0, perf_counter() - started),
                "details": dict(details or {}),
            }
        )

    def _client(self, node_id: str):
        try:
            return self.clients[node_id]
        except KeyError as exc:
            raise RuntimeError(f"no physical-control Agent for node {node_id!r}") from exc

    @staticmethod
    def _link_interfaces(link) -> tuple[tuple[str, str], ...]:
        labels = dict(link.spec.labels)
        if labels.get("physical_control_scope") != "interface":
            raise RuntimeError(
                f"link {link.spec.id!r} must declare physical_control_scope: interface; "
                "Linux tc changes the whole interface, not one abstract flow"
            )
        pairs: list[tuple[str, str]] = []
        source_interface = labels.get("physical_source_interface")
        target_interface = labels.get("physical_target_interface")
        if source_interface:
            pairs.append((link.spec.source, str(source_interface)))
        if link.spec.bidirectional and target_interface:
            pairs.append((link.spec.target, str(target_interface)))
        if not pairs:
            raise RuntimeError(
                f"link {link.spec.id!r} requires physical_source_interface and/or "
                "physical_target_interface labels for Real scenario control"
            )
        return tuple(pairs)

    async def prepare_event(self, event: Event, state: ContinuumState) -> None:
        if event.source != "experiment.scenario":
            return
        if event.kind in {EventKind.NODE_OFFLINE, EventKind.NODE_RECOVERED}:
            node_id = str(event.payload["node_id"])
            enabled = event.kind == EventKind.NODE_RECOVERED
            await self._record(
                event,
                node_id,
                lambda: self._client(node_id).control_workload(
                    enabled=enabled, lease_s=self.lease_s
                ),
            )
            return
        if event.kind in {EventKind.LINK_CHANGED, EventKind.LINK_REMOVED}:
            link_id = (
                str(event.payload["link"]["id"])
                if event.kind == EventKind.LINK_CHANGED
                else str(event.payload["link_id"])
            )
            try:
                current = state.links[link_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"physical scenario references unknown link {link_id!r}"
                ) from exc
            if event.kind == EventKind.LINK_CHANGED:
                link_payload = dict(event.payload["link"])
                latency_ms = float(link_payload["latency_ms"])
                bandwidth_mbps = float(link_payload["bandwidth_mbps"])
                loss_pct = None
            else:
                latency_ms = None
                bandwidth_mbps = None
                loss_pct = 100.0
            for node_id, interface in self._link_interfaces(current):
                await self._record(
                    event,
                    node_id,
                    lambda node_id=node_id, interface=interface: self._client(
                        node_id
                    ).control_netem(
                        interface=interface,
                        latency_ms=latency_ms,
                        bandwidth_mbps=bandwidth_mbps,
                        loss_pct=loss_pct,
                        lease_s=self.lease_s,
                    ),
                )
            return
        if event.kind == EventKind.MEASUREMENT_OBSERVED:
            measurement = dict(event.payload.get("measurement", {}))
            if str(measurement.get("name", "")) != "compute.cpu_capacity":
                return
            node_id = str(measurement.get("target", ""))
            value = float(measurement["value"])
            await self._record(
                event,
                node_id,
                lambda: self._client(node_id).control_cpu_capacity(value, lease_s=self.lease_s),
            )

    async def restore(self) -> None:
        await self._stop_heartbeat()
        results: dict[str, Any] = {}
        verified = not self._continuity_violations
        for node_id, client in self.clients.items():
            try:
                result = await client.control_restore()
            except Exception as exc:
                results[node_id] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                verified = False
                continue
            results[node_id] = result
            verified = verified and bool(result.get("ok", False))
        self.restore_results = results
        self.restore_verified = verified

    def report(self) -> Mapping[str, Any]:
        return {
            "schema": "darpan.physical-control/v1",
            "node_offline_mode": "agent-workload-plane-unavailable",
            "actions": list(self.actions),
            "restore_results": dict(self.restore_results),
            "restore_verified": self.restore_verified,
            "heartbeat": {
                "running": self._heartbeat is not None,
                "cycles": self._heartbeat_cycles,
                "renewal_successes": dict(self._heartbeat_successes),
                "failures": list(self._heartbeat_failures),
                "continuity_violations": list(self._continuity_violations),
                "continuity_verified": not self._continuity_violations,
            },
        }
