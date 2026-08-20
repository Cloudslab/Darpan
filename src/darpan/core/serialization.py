"""Small JSON-safe serialization helpers used by public domain objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_primitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
