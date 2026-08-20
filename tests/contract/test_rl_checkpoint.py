from __future__ import annotations

import torch
from darpan_rl import RLConfig, Trainer
from darpan_rl.algorithms import create_algorithm
from darpan_rl.trainer import EpisodeStats


def test_trainer_checkpoint_roundtrip(tmp_path):
    algorithm = create_algorithm(
        "PPO",
        observation_dim=4,
        action_dim=3,
        config=RLConfig(
            algorithm="PPO",
            hidden_dim=16,
            update_epochs=1,
            minibatch_size=2,
        ),
    )
    algorithm.policy_version = 9
    algorithm.episodes = 12
    trainer = Trainer(algorithm)
    trainer.history.append(EpisodeStats(reward=3.5, steps=4, source="twin"))

    path = trainer.save_checkpoint(tmp_path / "nested" / "ppo.pt")
    restored = Trainer.load_checkpoint(path)

    assert restored.algorithm.policy_version == 9
    assert restored.algorithm.episodes == 12
    assert restored.algorithm.observation_dim == 4
    assert restored.algorithm.action_dim == 3
    assert restored.algorithm.config.hidden_dim == 16
    assert restored.history == [EpisodeStats(reward=3.5, steps=4, source="twin")]

    original = trainer.algorithm.state_dict()["model"]
    loaded = restored.algorithm.state_dict()["model"]
    assert original.keys() == loaded.keys()
    assert all(torch.equal(original[key], loaded[key]) for key in original)


def test_checkpoint_can_override_device(tmp_path):
    algorithm = create_algorithm("DQN", 2, 2, RLConfig(algorithm="DQN", hidden_dim=8))
    trainer = Trainer(algorithm)
    path = trainer.save_checkpoint(tmp_path / "dqn.pt")

    restored = Trainer.load_checkpoint(path, device="cpu")

    assert restored.algorithm.config.device == "cpu"
    assert restored.algorithm.device.type == "cpu"
