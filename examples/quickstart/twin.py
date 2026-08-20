from darpan import Darpan
from darpan.core.codec import load_application, load_system
from darpan.experiment.baselines import RoundRobinPolicy
from darpan.experiment.metric import ApplicationLatency
from darpan.experiment.runner import ExperimentRunner

system = load_system("configs/systems/edge_fog_cloud.yaml")
app = load_application("configs/workloads/example.yaml")
session = Darpan.twin()
runner = ExperimentRunner(session, metrics=[ApplicationLatency()])
result = runner.run_sync(system, app, RoundRobinPolicy())
print(result.metrics)
