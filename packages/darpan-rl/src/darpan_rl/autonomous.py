"""Confidence-aware Twin-assisted online learning over public Darpan APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from darpan.adapters.rl import DarpanEnv

from .trainer import Trainer
from .types import Trajectory


@dataclass(slots=True)
class TwinLearningConfig:
    virtual_rollouts_per_real: int = 4
    min_confidence: float = 0.35
    max_virtual_uncertainty: float = 0.65
    min_sample_weight: float = 0.05
    max_sample_weight: float = 0.50
    max_virtual_weight_per_real: float = 2.0


class TwinAssistedTrainer:
    """Real experience stays weight 1; uncertain Twin experience is down-weighted."""

    def __init__(
        self,
        trainer: Trainer,
        *,
        real_env_factory: Callable[[], DarpanEnv],
        twin_env_factory: Callable[[], DarpanEnv],
        config: TwinLearningConfig | None = None,
        after_real: Callable[[Trajectory], None] | None = None,
    ) -> None:
        self.trainer = trainer
        self.real_env_factory = real_env_factory
        self.twin_env_factory = twin_env_factory
        self.config = config or TwinLearningConfig()
        self.after_real = after_real

    def _weight(self, uncertainty: float) -> float:
        if uncertainty > self.config.max_virtual_uncertainty:
            return 0.0
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        if confidence < self.config.min_confidence:
            return 0.0
        span = self.config.max_sample_weight - self.config.min_sample_weight
        return min(
            self.config.max_sample_weight,
            self.config.min_sample_weight + confidence * span,
        )

    def train(self, real_episodes: int) -> list[dict[str, float]]:
        updates: list[dict[str, float]] = []
        for _ in range(real_episodes):
            real_env = self.real_env_factory()
            try:
                real = self.trainer.collect_episode(real_env, source="real")
            finally:
                real_env.close()
            real.sample_weight = 1.0
            if self.after_real is not None:
                self.after_real(real)

            virtual: list[Trajectory] = []
            remaining_weight = self.config.max_virtual_weight_per_real
            for _ in range(self.config.virtual_rollouts_per_real):
                twin_env = self.twin_env_factory()
                try:
                    trajectory = self.trainer.collect_episode(
                        twin_env,
                        source="twin",
                        sample_weight=1.0,
                    )
                finally:
                    twin_env.close()
                weight = min(self._weight(trajectory.model_uncertainty), remaining_weight)
                if weight <= 0:
                    continue
                trajectory.sample_weight = weight
                virtual.append(trajectory)
                remaining_weight -= weight
                if remaining_weight <= 0:
                    break
            updates.append(self.trainer.algorithm.update([real, *virtual]))
        return updates
