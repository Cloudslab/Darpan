from __future__ import annotations

import json

from darpan.runtime.event_log import JsonlEventLog
from darpan.runtime.state_store import StateStore


def _replay(args) -> None:
    log = JsonlEventLog(args.event_log, replay_existing=True)
    store = StateStore()
    for event in log:
        store.apply(event)
    state = store.state
    print(
        json.dumps(
            {
                "events": len(log),
                "time": state.time,
                "nodes": len(state.nodes),
                "applications": len(state.application_instances),
                "components": len(state.components),
            },
            indent=2,
        )
    )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("replay", help="rebuild state from a Darpan EventLog")
    parser.add_argument("event_log")
    parser.set_defaults(func=_replay)
