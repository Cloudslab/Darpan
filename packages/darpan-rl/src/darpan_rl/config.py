from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RLConfig:
    algorithm: str = "PPO"
    device: str = "cpu"
    seed: int = 7
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    batch_episodes: int = 4
    replay_capacity: int = 20_000
    replay_batch_size: int = 64
    target_sync_updates: int = 20
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    max_policy_lag: int = 32
