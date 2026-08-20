from __future__ import annotations

from darpan.core.event import Event
from darpan.runtime.event_log import InMemoryEventLog, JsonlEventLog


def test_event_log_is_idempotent():
    log = InMemoryEventLog()
    event = Event("test.event", 1.0, "test")
    assert log.append(event)
    assert not log.append(event)
    assert len(log) == 1


def test_jsonl_event_log_replays(tmp_path):
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path)
    log.append(Event("test.one", 1.0, "test"))
    log.append(Event("test.two", 2.0, "test"))
    replay = JsonlEventLog(path)
    assert [event.kind for event in replay] == ["test.one", "test.two"]
