"""A third-party RL algorithm only needs DarpanEnv."""

from darpan.adapters.rl import DarpanEnv
from darpan.core.codec import load_application, load_system

system = load_system("configs/systems/edge_fog_cloud.yaml")
app = load_application("configs/workloads/example.yaml")
env = DarpanEnv(system=system, application=app, runtime="twin")
observation, info = env.reset()
while True:
    valid = [index for index, allowed in enumerate(info["action_mask"]) if allowed]
    action = valid[0]
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("episode reward:", reward)
        break
env.close()
