from __future__ import annotations

import asyncio

from darpan import Darpan, Measurement
from darpan.core.event import Event, EventKind
from darpan.core.protocols.model import ModelPrediction
from darpan.core.serialization import to_primitive
from darpan.experiment.metric import MeasurementPeak
from darpan.twin.models.registry import ModelRegistry


class ThermalModel:
    name = "thermal"

    def __init__(self):
        self.temperature = 40.0

    def observe(self, event, state):
        if event.kind == EventKind.MEASUREMENT_OBSERVED:
            raw = event.payload["measurement"]
            if raw["name"] == "temperature":
                self.temperature = float(raw["value"])

    def advance(self, start_time, end_time, state):
        self.temperature += max(0.0, end_time - start_time) * 0.1

    def predict(self, query, state):
        return ModelPrediction(self.temperature, uncertainty=0.1)

    def snapshot(self):
        return {"temperature": self.temperature}

    def restore(self, snapshot):
        self.temperature = float(snapshot["temperature"])


def test_new_system_quantity_needs_no_core_change(small_system):
    async def run():
        models = ModelRegistry([ThermalModel()])
        session = Darpan.twin(models=models)
        metric = MeasurementPeak("temperature", name="peak_temperature")
        session.subscribe(metric.observe)
        await session.start()
        await session.register_system(small_system)
        measurement = Measurement(
            "temperature",
            73.0,
            unit="degC",
            target="edge-1",
            timestamp=session.clock.now(),
            source="research.thermal",
        )
        await session.emit(
            Event(
                EventKind.MEASUREMENT_OBSERVED,
                session.clock.now(),
                "research.thermal",
                payload={"measurement": to_primitive(measurement)},
            )
        )
        assert session.state.nodes["edge-1"].measurement("temperature").value == 73.0
        assert metric.result() == 73.0
        await session.close()

    asyncio.run(run())
