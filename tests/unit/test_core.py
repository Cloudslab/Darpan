from __future__ import annotations

import pytest

from darpan import ApplicationSpec, ComponentSpec, FlowSpec, ResourceSpec
from darpan.core.measurement import Measurement
from darpan.core.resource import ResourceState


def test_application_rejects_cycles():
    with pytest.raises(ValueError):
        ApplicationSpec(
            "cycle",
            (ComponentSpec("a"), ComponentSpec("b")),
            (FlowSpec("a", "b"), FlowSpec("b", "a")),
        )


def test_extensible_resource_names_are_not_hard_coded():
    gpu = ResourceSpec("accelerator.gpu", 2, "device")
    assert gpu.name == "accelerator.gpu"
    state = ResourceState(gpu.name, gpu.capacity, allocated=1, unit=gpu.unit)
    assert state.available == 1


def test_extensible_measurement_supports_thermal():
    measurement = Measurement(
        "temperature",
        72.5,
        unit="degC",
        target="edge-1",
        source="sensor",
        uncertainty=0.2,
    )
    assert measurement.name == "temperature"
    assert measurement.value == 72.5
