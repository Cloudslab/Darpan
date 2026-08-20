from __future__ import annotations

from ..types import Trajectory
from .ppo import PPO


class APPO(PPO):
    """Asynchronous PPO with explicit stale-policy filtering."""

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        filtered = [
            trajectory
            for trajectory in trajectories
            if all(
                self.policy_version - step.policy_version <= self.config.max_policy_lag
                for step in trajectory.steps
            )
        ]
        metrics = super().update(filtered)
        metrics["discarded_stale"] = float(len(trajectories) - len(filtered))
        return metrics
