"""Thin adapter for GA/NSGA/PSO/MILP-style candidate evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from darpan.experiment.constraint import Constraint
from darpan.experiment.objective import Objective


@dataclass(frozen=True, slots=True)
class Evaluation:
    metrics: Mapping[str, float]
    objectives: tuple[float, ...]
    feasible: bool


class Evaluator:
    """Algorithm-agnostic candidate evaluator.

    The caller supplies a function that materializes canonical Actions from a
    candidate and a function that executes them using a Darpan session/scenario.
    This keeps optimization libraries outside Darpan Core.
    """

    def __init__(
        self,
        *,
        objectives: tuple[Objective, ...],
        constraints: tuple[Constraint, ...] = (),
    ) -> None:
        self.objectives = objectives
        self.constraints = constraints

    def evaluate_metrics(self, metrics: Mapping[str, float]) -> Evaluation:
        objectives = tuple(objective.value(metrics) for objective in self.objectives)
        feasible = all(constraint.satisfied(metrics) for constraint in self.constraints)
        return Evaluation(metrics, objectives, feasible)
