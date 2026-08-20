"""Expected-vs-observed comparison used by Twin fidelity and diagnosis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Comparison:
    expected: float
    observed: float

    @property
    def residual(self) -> float:
        return self.observed - self.expected

    @property
    def relative_error(self) -> float:
        if abs(self.observed) < 1e-12:
            return 0.0 if abs(self.expected) < 1e-12 else float("inf")
        return abs(self.expected - self.observed) / abs(self.observed)
