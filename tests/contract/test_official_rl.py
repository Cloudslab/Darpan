from __future__ import annotations

import numpy as np
from darpan_rl import RLConfig
from darpan_rl.algorithms import create_algorithm
from darpan_rl.types import Step, Trajectory


def test_all_official_algorithms_use_generic_vector_contract():
    obs = np.asarray([0.2, 0.4, 1.0, 0.0], dtype=np.float32)
    mask = np.asarray([True, True, False], dtype=np.bool_)
    for name in ("PPO", "A2C", "A3C", "DQN", "IMPALA", "APPO"):
        algorithm = create_algorithm(
            name,
            4,
            3,
            RLConfig(
                algorithm=name,
                hidden_dim=16,
                update_epochs=1,
                minibatch_size=2,
                replay_batch_size=2,
            ),
        )
        output = algorithm.act(obs, mask)
        step = Step(
            observation=obs,
            action_mask=mask,
            action=output.action,
            reward=1.0,
            next_observation=None,
            next_action_mask=None,
            done=True,
            log_prob=output.log_prob,
            value=output.value,
            policy_version=output.policy_version,
        )
        metrics = algorithm.update([Trajectory([step])])
        assert isinstance(metrics, dict)
