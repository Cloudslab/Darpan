from __future__ import annotations

from darpan.runtime.clock import VirtualClock
from darpan.runtime.session import Session
from darpan.twin.backend import TwinBackend


def test_session_uses_backend_clock_when_clock_is_not_explicitly_supplied():
    clock = VirtualClock(7.0)
    backend = TwinBackend(clock=clock)
    session = Session(backend)
    assert session.clock is clock
    assert session.clock is backend.clock
    assert session.clock.now() == 7.0
