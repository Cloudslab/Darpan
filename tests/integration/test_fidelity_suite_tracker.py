from __future__ import annotations

import asyncio
import sys

from darpan import Action, ApplicationSpec, ComponentSpec, FlowSpec, LinkSpec, NodeSpec
from darpan.core.event import EventKind
from darpan.core.resource import ResourceSpec
from darpan.core.topology import SystemSpec
from darpan.experiment.fidelity import FidelitySuiteTracker
from darpan.runtime.real.backend import RealBackend
from darpan.runtime.session import Session
from darpan.twin.models.artifact import ArtifactSizeModel
from darpan.twin.models.execution import ExecutionTimeModel
from darpan.twin.models.network import NetworkDelayModel
from darpan.twin.models.registry import ModelRegistry


def test_fidelity_suite_tracks_execution_artifact_and_network_before_calibration():
    async def run():
        models = ModelRegistry(
            [
                ExecutionTimeModel(calibration_rate=1.0),
                ArtifactSizeModel(calibration_rate=1.0),
                NetworkDelayModel(calibration_rate=1.0),
            ]
        )
        tracker = FidelitySuiteTracker(models)
        session = Session(RealBackend())
        session.subscribe(tracker.observe)
        system = SystemSpec(
            nodes=(
                NodeSpec("n1", resources=(ResourceSpec("cpu", 1),)),
                NodeSpec("n2", resources=(ResourceSpec("cpu", 1),)),
            ),
            links=(
                LinkSpec(
                    "n1-n2",
                    "n1",
                    "n2",
                    latency_ms=1,
                    bandwidth_mbps=100,
                ),
            ),
        )
        size = 4096
        app = ApplicationSpec(
            "fidelity-flow",
            components=(
                ComponentSpec(
                    "producer",
                    command=(
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path('out.bin').write_bytes(b'x'*{size})",
                    ),
                ),
                ComponentSpec(
                    "consumer",
                    command=(
                        sys.executable,
                        "-c",
                        "from pathlib import Path; assert len(Path('in.bin').read_bytes()) > 0",
                    ),
                ),
            ),
            flows=(
                FlowSpec(
                    "producer",
                    "consumer",
                    data_size_bytes=size,
                    artifact="out.bin",
                    target_path="in.bin",
                ),
            ),
        )
        await session.start()
        await session.register_system(system)
        instance = await session.submit_application(app)
        await session.apply(Action.place(f"{instance}:producer", "n1"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.COMPONENT_READY
            and event.subject == f"{instance}:consumer",
            timeout=3,
        )
        await session.apply(Action.place(f"{instance}:consumer", "n2"))
        await session.wait_for(
            lambda event, state: event.kind == EventKind.APPLICATION_COMPLETED,
            timeout=3,
        )

        assert len(tracker.execution.samples) == 2
        assert len(tracker.artifact.samples) == 2
        assert len(tracker.network.samples) == 1
        assert tracker.network.samples[0].metadata["size_bytes"] == size
        summaries = tracker.summaries()
        assert summaries["execution"].samples == 2
        assert summaries["artifact"].samples == 2
        assert summaries["network"].samples == 1
        assert models.get("network").snapshot()["scales"]
        await session.close()

    asyncio.run(run())
