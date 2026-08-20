"""Append-only event logs with idempotent duplicate handling."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from darpan.core.event import Event


class EventLog(Protocol):
    def append(self, event: Event) -> bool: ...

    def __iter__(self) -> Iterable[Event]: ...

    def __len__(self) -> int: ...


class InMemoryEventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._ids: set[str] = set()

    def append(self, event: Event) -> bool:
        if event.id in self._ids:
            return False
        self._ids.add(event.id)
        self._events.append(event)
        return True

    def __iter__(self):
        return iter(tuple(self._events))

    def __len__(self) -> int:
        return len(self._events)

    def events_since(self, index: int) -> tuple[Event, ...]:
        return tuple(self._events[index:])


class JsonlEventLog(InMemoryEventLog):
    def __init__(self, path: str | Path, *, replay_existing: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if replay_existing and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        super().append(Event.from_dict(json.loads(line)))

    def append(self, event: Event) -> bool:
        if not super().append(event):
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), separators=(",", ":")))
            handle.write("\n")
        return True
