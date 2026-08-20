from __future__ import annotations

import asyncio

from darpan import Darpan


class Service:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True


def test_session_manages_background_service_lifecycle():
    async def run():
        session = Darpan.twin()
        service = session.add_service(Service())
        await session.start()
        assert service.started is True
        await session.close()
        assert service.closed is True

    asyncio.run(run())


def test_session_close_breaks_runtime_reference_cycles():
    async def run():
        for session in (Darpan.real(), Darpan.twin()):
            service = session.add_service(Service())
            def listener(event, state):
                return None

            session.subscribe(listener)
            await session.start()
            assert getattr(session.backend, "context", None) is not None
            await session.close()
            assert service.closed is True
            assert getattr(session.backend, "context", None) is None
            assert session._listeners == []
            assert session._services == []

    asyncio.run(run())
