from __future__ import annotations

from darpan import Action, Darpan, DigitalTwin
from darpan.twin.kernel import DiscreteEventQueue


class RecordingKernel(DiscreteEventQueue):
    def __init__(self):
        super().__init__()
        self.scheduled = 0

    def schedule(self, event, at):
        self.scheduled += 1
        return super().schedule(event, at)


def test_darpan_twin_accepts_replaceable_simulation_kernel(small_system, small_app):
    import asyncio

    async def run():
        kernel = RecordingKernel()
        session = Darpan.twin(kernel=kernel)
        await session.start()
        await session.register_system(small_system)
        instance = await session.submit_application(small_app)
        await session.apply(Action.place(f"{instance}:a", "edge-1"))
        assert kernel.scheduled > 0
        await session.close()

    asyncio.run(run())


def test_digital_twin_uses_kernel_factory_for_scenarios(small_system):
    import asyncio

    created = []

    def factory():
        kernel = RecordingKernel()
        created.append(kernel)
        return kernel

    async def run():
        base = Darpan.twin()
        await base.start()
        await base.register_system(small_system)
        twin = DigitalTwin(kernel_factory=factory)
        session = twin.session(twin.scenario(base.state))
        await session.start()
        await session.close()
        await base.close()

    asyncio.run(run())
    assert len(created) == 1
