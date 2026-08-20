from __future__ import annotations

import pytest

import darpan.runtime.clock as clock_module


def test_wall_clock_uses_high_resolution_monotonic_counter(monkeypatch) -> None:
    readings = iter((100.0, 100.000_125))
    monkeypatch.setattr(clock_module, "perf_counter", lambda: next(readings))

    clock = clock_module.WallClock()

    assert clock.now() == pytest.approx(0.000_125)
