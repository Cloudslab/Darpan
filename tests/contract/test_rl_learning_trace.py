from __future__ import annotations

from darpan_rl.trainer import EpisodeStats, Trainer

from darpan.experiment.learning import load_learning_trace


class _Algorithm:
    pass


def test_rl_trainer_exports_real_quality_with_virtual_interaction_accounting(tmp_path) -> None:
    trainer = Trainer(_Algorithm())
    trainer.history = [
        EpisodeStats(1.0, 10, "real", duration_s=1.0),
        EpisodeStats(2.0, 30, "twin", duration_s=0.5, model_uncertainty=0.2),
        EpisodeStats(3.0, 20, "twin", duration_s=0.5, model_uncertainty=0.1),
        EpisodeStats(4.0, 8, "real", duration_s=1.0),
    ]

    path = trainer.export_learning_trace(tmp_path / "learning.jsonl")
    points = load_learning_trace(path)
    assert len(points) == 2
    assert points[0].quality == 1.0
    assert points[0].real_interactions == 10
    assert points[0].virtual_interactions == 0
    assert points[1].quality == 4.0
    assert points[1].real_interactions == 18
    assert points[1].virtual_interactions == 50
    assert points[1].wall_time_s == 3.0
    assert points[1].metadata["quality_kind"] == "real_episode_return"
