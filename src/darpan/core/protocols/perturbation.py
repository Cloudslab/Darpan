from __future__ import annotations

from typing import Any, Protocol

from ..event import Event
from ..measurement import Measurement


class ScenarioEditor(Protocol):
    def remove_node(self, node_id: str) -> None: ...

    def change_link(self, link_id: str, **changes: Any) -> None: ...

    def set_measurement(self, measurement: Measurement) -> None: ...

    def inject_event(self, event: Event) -> None: ...


class Perturbation(Protocol):
    name: str

    def apply(self, scenario: ScenarioEditor) -> None: ...
