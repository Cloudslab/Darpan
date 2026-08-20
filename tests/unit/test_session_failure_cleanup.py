from __future__ import annotations

import asyncio

import pytest

from darpan.runtime.session import Session


class RecordingBackend:
    supported_action_kinds = frozenset()

    def __init__(self, *, fail_start: bool = False, fail_close: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_close = fail_close
        self.started = 0
        self.closed = 0

    async def start(self, context) -> None:
        self.started += 1
        if self.fail_start:
            raise RuntimeError("backend start failed")

    async def apply(self, action) -> None:
        raise AssertionError("not used")

    async def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("backend close failed")


class RecordingService:
    def __init__(self, *, fail_start: bool = False, fail_close: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_close = fail_close
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1
        if self.fail_start:
            raise RuntimeError("service start failed")

    async def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("service close failed")


def test_failed_backend_start_does_not_mark_session_started():
    async def run():
        backend = RecordingBackend(fail_start=True)
        session = Session(backend)
        with pytest.raises(RuntimeError, match="backend start failed"):
            await session.start()
        assert session._started is False
        assert backend.closed == 0

    asyncio.run(run())


def test_failed_service_start_rolls_back_started_services_and_backend():
    async def run():
        backend = RecordingBackend()
        first = RecordingService()
        second = RecordingService(fail_start=True)
        session = Session(backend)
        session.add_service(first)
        session.add_service(second)
        with pytest.raises(RuntimeError, match="service start failed"):
            await session.start()
        assert session._started is False
        assert first.closed == 1
        assert backend.closed == 1

    asyncio.run(run())


def test_close_attempts_all_cleanup_even_if_a_service_close_fails():
    async def run():
        backend = RecordingBackend()
        first = RecordingService(fail_close=True)
        second = RecordingService()
        session = Session(backend)
        session.add_service(first)
        session.add_service(second)
        await session.start()
        with pytest.raises(ExceptionGroup) as caught:
            await session.close()
        assert first.closed == 1
        assert second.closed == 1
        assert backend.closed == 1
        assert any("service close failed" in str(exc) for exc in caught.value.exceptions)

    asyncio.run(run())
