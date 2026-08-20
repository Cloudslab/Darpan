import asyncio

from darpan import DigitalTwin
from darpan.core.codec import load_application, load_system
from darpan.experiment.baselines import RoundRobinPolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner


async def main():
    system = load_system("configs/systems/edge_fog_cloud.yaml")
    app = load_application("configs/workloads/example.yaml")
    # Build a canonical state with the system description, then fork it.
    from darpan import Darpan
    base = Darpan.twin()
    await base.start()
    await base.register_system(system)
    twin = DigitalTwin()
    scenario = twin.scenario(base.state).change_link("edge-fog", latency_ms=50)
    session = twin.session(scenario)
    runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
    result = await runner.run(system, app, RoundRobinPolicy())
    print(result.metrics)
    await session.close()
    await base.close()


asyncio.run(main())
