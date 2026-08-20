from __future__ import annotations


def combine_uncertainties(*values: float | None) -> float | None:
    known = [max(0.0, min(1.0, value)) for value in values if value is not None]
    if not known:
        return None
    # Independent-failure style aggregation; monotonic and bounded.
    remaining = 1.0
    for value in known:
        remaining *= 1.0 - value
    return 1.0 - remaining
