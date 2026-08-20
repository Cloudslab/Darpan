from __future__ import annotations

from ..config import RLConfig
from .a2c import A2C
from .a3c import A3C
from .appo import APPO
from .dqn import DQN
from .impala import IMPALA
from .ppo import PPO


def create_algorithm(
    name: str,
    observation_dim: int,
    action_dim: int,
    config: RLConfig | None = None,
):
    config = config or RLConfig(algorithm=name.upper())
    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
        "A3C": A3C,
        "DQN": DQN,
        "IMPALA": IMPALA,
        "APPO": APPO,
    }
    try:
        cls = algorithms[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown Darpan RL algorithm: {name}") from exc
    return cls(observation_dim, action_dim, config)


__all__ = ["A2C", "A3C", "APPO", "DQN", "IMPALA", "PPO", "create_algorithm"]
