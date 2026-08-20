from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, median


def aggregate(values: Iterable[float]) -> dict[str, float]:
    data = list(values)
    if not data:
        return {}
    return {
        "mean": mean(data),
        "median": median(data),
        "min": min(data),
        "max": max(data),
    }
