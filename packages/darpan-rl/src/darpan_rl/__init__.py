"""Official Darpan reinforcement-learning companion library."""

from ._version import __version__
from .algorithms import A2C, A3C, APPO, DQN, IMPALA, PPO, create_algorithm
from .autonomous import TwinAssistedTrainer, TwinLearningConfig
from .config import RLConfig
from .trainer import A3CTrainer, Trainer

__all__ = [
    "__version__",
    "A2C",
    "A3C",
    "A3CTrainer",
    "APPO",
    "DQN",
    "IMPALA",
    "PPO",
    "RLConfig",
    "Trainer",
    "TwinAssistedTrainer",
    "TwinLearningConfig",
    "create_algorithm",
]
