from __future__ import annotations

import numpy as np
from darpan_rl.trainer import Trainer


class _Algorithm:
    def act(self, observation, mask, deterministic=False):
        del observation, mask, deterministic
        return type(
            "Output",
            (),
            {"action": 0, "log_prob": 0.0, "value": 0.0, "policy_version": 0},
        )()


class _TruncatingEnv:
    episode_uncertainty = 0.0

    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        return np.asarray([0.0]), {"action_mask": [True]}

    def step(self, action):
        del action
        self.steps += 1
        return (
            np.asarray([1.0]),
            1.0,
            False,
            True,
            {"action_mask": [True]},
        )


def test_trainer_stops_episode_on_truncation() -> None:
    env = _TruncatingEnv()
    trainer = Trainer(_Algorithm())
    trajectory = trainer.collect_episode(env)
    assert env.steps == 1
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].done is True
    assert trajectory.steps[0].next_observation is None
