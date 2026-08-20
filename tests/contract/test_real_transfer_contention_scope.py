from __future__ import annotations

import asyncio

from darpan import LinkSpec, NodeSpec, SystemSpec
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.session import Session


def test_real_transfer_contention_only_marks_overlapping_logical_paths():
    async def run():
        backend = RealBackend()
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "a",
            logical_links=frozenset({"l1"}),
        )
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "b",
            logical_links=frozenset({"l2"}),
        )
        assert await backend._leave_artifact_transfer("a") is False  # noqa: SLF001
        assert await backend._leave_artifact_transfer("b") is False  # noqa: SLF001

        await backend._enter_artifact_transfer(  # noqa: SLF001
            "c",
            logical_links=frozenset({"l1", "l2"}),
        )
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "d",
            logical_links=frozenset({"l2", "l3"}),
        )
        assert await backend._leave_artifact_transfer("c") is True  # noqa: SLF001
        assert await backend._leave_artifact_transfer("d") is True  # noqa: SLF001

    asyncio.run(run())


def test_unknown_real_transfer_path_remains_conservative():
    async def run():
        backend = RealBackend()
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "known", logical_links=frozenset({"l1"})
        )
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "unknown", logical_links=None
        )
        assert await backend._leave_artifact_transfer("known") is True  # noqa: SLF001
        assert await backend._leave_artifact_transfer("unknown") is True  # noqa: SLF001

        await backend._enter_artifact_transfer(  # noqa: SLF001
            "local", logical_links=frozenset()
        )
        await backend._enter_artifact_transfer(  # noqa: SLF001
            "remote", logical_links=None
        )
        assert await backend._leave_artifact_transfer("local") is False  # noqa: SLF001
        assert await backend._leave_artifact_transfer("remote") is False  # noqa: SLF001

    asyncio.run(run())


def test_real_transfer_audit_path_uses_declared_continuum_links():
    async def run():
        backend = RealBackend()
        session = Session(backend)
        await session.start()
        await session.register_system(
            SystemSpec(
                nodes=tuple(NodeSpec(node_id) for node_id in ("a", "b", "c", "d")),
                links=(
                    LinkSpec("ab", "a", "b", latency_ms=1.0, bandwidth_mbps=10.0),
                    LinkSpec("bc", "b", "c", latency_ms=1.0, bandwidth_mbps=10.0),
                    LinkSpec("ad", "a", "d", latency_ms=50.0, bandwidth_mbps=10.0),
                    LinkSpec("dc", "d", "c", latency_ms=50.0, bandwidth_mbps=10.0),
                ),
            )
        )
        assert backend._logical_transfer_links(  # noqa: SLF001
            "a", "c", size_bytes=1_000
        ) == frozenset({"ab", "bc"})
        assert backend._logical_transfer_links(  # noqa: SLF001
            "a", "a", size_bytes=1_000
        ) == frozenset()
        await session.close()

    asyncio.run(run())
