from darpan_rl import RLConfig, Trainer

from darpan.adapters.rl import DarpanEnv
from darpan.core.codec import load_application, load_system

system = load_system("configs/systems/edge_fog_cloud.yaml")
app = load_application("configs/workloads/example.yaml")
env = DarpanEnv(system=system, application=app, runtime="twin")
trainer = Trainer.create("PPO", env, RLConfig(batch_episodes=4))
print(trainer.train(env, episodes=8))
env.close()
