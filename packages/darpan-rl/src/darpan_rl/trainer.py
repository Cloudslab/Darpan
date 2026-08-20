"""Training loops over the public DarpanEnv contract only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch

from darpan.adapters.rl import DarpanEnv
from darpan.experiment.learning import LearningPoint, write_learning_trace

from .algorithms import create_algorithm
from .algorithms.base import Algorithm
from .config import RLConfig
from .types import Step, Trajectory


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    reward: float
    steps: int
    source: str
    duration_s: float = 0.0
    model_uncertainty: float = 0.0


class Trainer:
    def __init__(self, algorithm: Algorithm) -> None:
        self.algorithm = algorithm
        self.history: list[EpisodeStats] = []

    @classmethod
    def create(
        cls,
        algorithm: str,
        env: DarpanEnv,
        config: RLConfig | None = None,
    ) -> Trainer:
        observation, _ = env.reset(seed=None if config is None else config.seed)
        instance = create_algorithm(
            algorithm,
            observation_dim=int(np.asarray(observation).size),
            action_dim=env.action_size,
            config=config,
        )
        return cls(instance)

    def collect_episode(
        self,
        env: DarpanEnv,
        *,
        deterministic: bool = False,
        source: str = "real",
        sample_weight: float = 1.0,
    ) -> Trajectory:
        started_at = perf_counter()
        observation, info = env.reset()
        steps: list[Step] = []
        done = False
        while not done:
            mask = np.asarray(info["action_mask"], dtype=np.bool_)
            output = self.algorithm.act(
                np.asarray(observation, dtype=np.float32),
                mask,
                deterministic=deterministic,
            )
            next_observation, reward, terminated, truncated, next_info = env.step(
                output.action
            )
            done = bool(terminated or truncated)
            steps.append(
                Step(
                    observation=np.asarray(observation, dtype=np.float32).copy(),
                    action_mask=mask.copy(),
                    action=output.action,
                    reward=float(reward),
                    next_observation=(
                        None
                        if done
                        else np.asarray(next_observation, dtype=np.float32).copy()
                    ),
                    next_action_mask=(
                        None
                        if done
                        else np.asarray(next_info["action_mask"], dtype=np.bool_).copy()
                    ),
                    done=done,
                    log_prob=output.log_prob,
                    value=output.value,
                    policy_version=output.policy_version,
                )
            )
            observation, info = next_observation, next_info
        trajectory = Trajectory(
            steps,
            source=source,
            sample_weight=sample_weight,
            model_uncertainty=float(getattr(env, "episode_uncertainty", 0.0)),
        )
        self.history.append(
            EpisodeStats(
                trajectory.reward,
                len(steps),
                source,
                duration_s=perf_counter() - started_at,
                model_uncertainty=trajectory.model_uncertainty,
            )
        )
        return trajectory

    def train(self, env: DarpanEnv, episodes: int) -> list[dict[str, float]]:
        batch: list[Trajectory] = []
        updates: list[dict[str, float]] = []
        for _ in range(episodes):
            batch.append(self.collect_episode(env))
            if len(batch) >= self.algorithm.config.batch_episodes:
                updates.append(self.algorithm.update(batch))
                batch.clear()
        if batch:
            updates.append(self.algorithm.update(batch))
        return updates

    def export_learning_trace(self, path: str | Path) -> Path:
        """Export algorithm-neutral learning evidence from real episode returns.

        Twin episodes contribute to the cumulative virtual-interaction count but
        are not themselves treated as policy-quality observations.  The next
        real episode therefore measures quality after preceding Twin-assisted
        updates without letting virtual reward self-certify the policy.
        """

        real_interactions = 0
        virtual_interactions = 0
        elapsed = 0.0
        points: list[LearningPoint] = []
        for index, episode in enumerate(self.history, start=1):
            elapsed += max(0.0, float(episode.duration_s))
            if episode.source == "real":
                real_interactions += int(episode.steps)
                points.append(
                    LearningPoint(
                        quality=float(episode.reward),
                        real_interactions=real_interactions,
                        virtual_interactions=virtual_interactions,
                        wall_time_s=elapsed,
                        metadata={
                            "episode": index,
                            "quality_kind": "real_episode_return",
                            "model_uncertainty": float(episode.model_uncertainty),
                        },
                    )
                )
            else:
                virtual_interactions += int(episode.steps)
        if not points:
            raise ValueError("learning trace requires at least one real episode")
        return write_learning_trace(path, points)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist algorithm, optimizer and trainer state in a portable checkpoint."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "darpan-rl.checkpoint",
            "version": 1,
            "algorithm": type(self.algorithm).__name__.upper(),
            "observation_dim": self.algorithm.observation_dim,
            "action_dim": self.algorithm.action_dim,
            "config": asdict(self.algorithm.config),
            "algorithm_state": self.algorithm.state_dict(),
            "history": [asdict(item) for item in self.history],
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> Trainer:
        """Restore a checkpoint without requiring the original environment."""

        payload = torch.load(
            Path(path),
            map_location=device or "cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("invalid Darpan RL checkpoint payload")
        if payload.get("format") != "darpan-rl.checkpoint":
            raise ValueError("not a Darpan RL checkpoint")
        if int(payload.get("version", 0)) != 1:
            raise ValueError(
                f"unsupported Darpan RL checkpoint version: {payload.get('version')}"
            )

        config_data = dict(payload["config"])
        if device is not None:
            config_data["device"] = device
        config = RLConfig(**config_data)
        algorithm = create_algorithm(
            str(payload["algorithm"]),
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            config=config,
        )
        algorithm.load_state_dict(payload["algorithm_state"])
        trainer = cls(algorithm)
        trainer.history = [EpisodeStats(**item) for item in payload.get("history", [])]
        return trainer


class A3CTrainer:
    """Threaded A3C actor workers with local models and shared gradient application."""

    def __init__(self, shared_algorithm, env_factory, *, workers: int = 2) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from .algorithms.a3c import A3C

        if not isinstance(shared_algorithm, A3C):
            raise TypeError("A3CTrainer requires a darpan_rl.A3C algorithm")
        self.shared = shared_algorithm
        self.env_factory = env_factory
        self.workers = max(1, workers)
        self._lock = Lock()
        self._executor_class = ThreadPoolExecutor

    def _worker(self, episodes: int) -> list[dict[str, float]]:
        from .algorithms.a3c import A3C

        local = A3C(
            self.shared.observation_dim,
            self.shared.action_dim,
            self.shared.config,
        )
        trainer = Trainer(local)
        metrics: list[dict[str, float]] = []
        env = self.env_factory()
        try:
            for _ in range(episodes):
                with self._lock:
                    local.model.load_state_dict(self.shared.model.state_dict())
                    local.policy_version = self.shared.policy_version
                trajectory = trainer.collect_episode(env)
                gradients, worker_metrics = local.compute_gradients([trajectory])
                with self._lock:
                    self.shared.apply_gradients(gradients)
                    self.shared.episodes += 1
                    worker_metrics["policy_version"] = float(
                        self.shared.policy_version
                    )
                metrics.append(worker_metrics)
        finally:
            env.close()
        return metrics

    def train(self, episodes: int) -> list[dict[str, float]]:
        assignments = [episodes // self.workers] * self.workers
        for index in range(episodes % self.workers):
            assignments[index] += 1
        with self._executor_class(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self._worker, count)
                for count in assignments
                if count > 0
            ]
            results: list[dict[str, float]] = []
            for future in futures:
                results.extend(future.result())
        return results
