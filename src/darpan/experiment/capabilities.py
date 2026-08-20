"""Capability-level assessments built from shared Darpan/Twin primitives.

These helpers deliberately do not create six independent subsystems. They turn
canonical comparisons, predictions, simulation results, and action events into
small reproducible reports suitable for experiments and paper benchmarks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from statistics import mean

from darpan.core.event import Event, EventKind
from darpan.twin.comparison import Comparison
from darpan.twin.prediction import Prediction
from darpan.twin.result import SimulationResult


@dataclass(frozen=True, slots=True)
class TrustworthinessReport:
    samples: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_relative_error: float


@dataclass(frozen=True, slots=True)
class AdaptationReport:
    before_error: float
    after_error: float
    absolute_improvement: float
    relative_improvement: float | None

    @property
    def improved(self) -> bool:
        return self.after_error < self.before_error


@dataclass(frozen=True, slots=True)
class ProactiveAssessment:
    predicted_value: float
    threshold: float
    relation: str
    violation_predicted: bool
    uncertainty: float | None
    interval: tuple[float, float] | None
    horizon_s: float | None


@dataclass(frozen=True, slots=True)
class ProactiveSample:
    predicted_violation: bool
    observed_violation: bool
    lead_time_s: float | None = None
    uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class ProactiveBenchmarkReport:
    samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1_score: float | None
    accuracy: float
    mean_true_positive_lead_time_s: float | None
    mean_uncertainty: float | None


@dataclass(frozen=True, slots=True)
class ActionExplanation:
    action_id: str
    kind: str | None
    target: str | None
    requested_by: str | None
    accepted: bool | None
    outcome: str
    reason: str | None
    lifecycle: tuple[str, ...]
    correlation_id: str | None
    request_payload: Mapping[str, object] = field(default_factory=dict)
    request_metadata: Mapping[str, object] = field(default_factory=dict)
    policy_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExplainabilityBenchmarkReport:
    actions: int
    decision_coverage: float
    terminal_coverage: float
    rationale_coverage: float
    payload_coverage: float
    negative_outcomes: int
    negative_reason_coverage: float | None
    outcomes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ExplorationReport:
    metric: str
    baseline: float
    alternatives: Mapping[str, float]
    deltas: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    baseline: float
    scenario_scores: Mapping[str, float]
    worst_case: float
    worst_case_scenario: str
    worst_case_degradation: float
    mean_degradation: float
    tolerance: float
    within_tolerance_fraction: float


def assess_trustworthiness(
    comparisons: Iterable[Comparison],
) -> TrustworthinessReport:
    """Summarize expected-vs-observed Twin fidelity."""

    items = tuple(comparisons)
    if not items:
        raise ValueError("trustworthiness assessment requires at least one comparison")
    residuals = [item.residual for item in items]
    relative_errors = [item.relative_error for item in items]
    finite_relative = [value for value in relative_errors if math.isfinite(value)]
    mean_relative_error = (
        mean(finite_relative) if finite_relative else float("inf")
    )
    return TrustworthinessReport(
        samples=len(items),
        mean_absolute_error=mean(abs(value) for value in residuals),
        root_mean_squared_error=math.sqrt(mean(value * value for value in residuals)),
        mean_relative_error=mean_relative_error,
    )


def assess_adaptation(
    before: Iterable[Comparison],
    after: Iterable[Comparison],
) -> AdaptationReport:
    """Measure whether online calibration reduced Twin prediction error."""

    before_report = assess_trustworthiness(before)
    after_report = assess_trustworthiness(after)
    before_error = before_report.mean_absolute_error
    after_error = after_report.mean_absolute_error
    improvement = before_error - after_error
    relative = None if abs(before_error) < 1e-12 else improvement / before_error
    return AdaptationReport(
        before_error=before_error,
        after_error=after_error,
        absolute_improvement=improvement,
        relative_improvement=relative,
    )


def assess_proactive(
    prediction: Prediction,
    *,
    threshold: float,
    relation: str = ">",
) -> ProactiveAssessment:
    """Turn a future Twin prediction into an explicit threshold assessment."""

    try:
        estimate = float(prediction.estimate)
    except (TypeError, ValueError) as exc:
        raise TypeError("proactive assessment requires a numeric prediction") from exc
    predicates: dict[str, Callable[[float, float], bool]] = {
        ">": lambda value, limit: value > limit,
        ">=": lambda value, limit: value >= limit,
        "<": lambda value, limit: value < limit,
        "<=": lambda value, limit: value <= limit,
    }
    try:
        predicate = predicates[relation]
    except KeyError as exc:
        raise ValueError("relation must be one of: >, >=, <, <=") from exc
    return ProactiveAssessment(
        predicted_value=estimate,
        threshold=float(threshold),
        relation=relation,
        violation_predicted=predicate(estimate, float(threshold)),
        uncertainty=prediction.uncertainty,
        interval=prediction.interval,
        horizon_s=prediction.horizon_s,
    )


def assess_proactive_benchmark(
    samples: Iterable[ProactiveSample],
) -> ProactiveBenchmarkReport:
    """Score proactive warnings against subsequently observed threshold outcomes."""

    items = tuple(samples)
    if not items:
        raise ValueError("proactive benchmark requires at least one sample")
    tp = sum(item.predicted_violation and item.observed_violation for item in items)
    fp = sum(item.predicted_violation and not item.observed_violation for item in items)
    tn = sum(not item.predicted_violation and not item.observed_violation for item in items)
    fn = sum(not item.predicted_violation and item.observed_violation for item in items)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    lead_times = tuple(
        float(item.lead_time_s)
        for item in items
        if item.predicted_violation
        and item.observed_violation
        and item.lead_time_s is not None
    )
    uncertainties = tuple(
        float(item.uncertainty)
        for item in items
        if item.uncertainty is not None
    )
    return ProactiveBenchmarkReport(
        samples=len(items),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=(tp + tn) / len(items),
        mean_true_positive_lead_time_s=None if not lead_times else mean(lead_times),
        mean_uncertainty=None if not uncertainties else mean(uncertainties),
    )


def explain_action(events: Iterable[Event], action_id: str) -> ActionExplanation:
    """Explain one action using only its canonical lifecycle events."""

    lifecycle_kinds = {
        EventKind.ACTION_REQUESTED,
        EventKind.ACTION_ACCEPTED,
        EventKind.ACTION_REJECTED,
        EventKind.ACTION_STARTED,
        EventKind.ACTION_COMPLETED,
        EventKind.ACTION_FAILED,
    }
    matched = tuple(
        event
        for event in events
        if event.kind in lifecycle_kinds and event.payload.get("action_id") == action_id
    )
    if not matched:
        raise KeyError(f"no action lifecycle found for {action_id!r}")

    requested = next(
        (event for event in matched if event.kind == EventKind.ACTION_REQUESTED), None
    )
    decision = next(
        (
            event
            for event in matched
            if event.kind in {EventKind.ACTION_ACCEPTED, EventKind.ACTION_REJECTED}
        ),
        None,
    )
    failed = next(
        (event for event in matched if event.kind == EventKind.ACTION_FAILED), None
    )
    completed = next(
        (event for event in matched if event.kind == EventKind.ACTION_COMPLETED), None
    )
    if failed is not None:
        outcome = "failed"
    elif completed is not None:
        outcome = "completed"
    elif decision is not None and decision.kind == EventKind.ACTION_REJECTED:
        outcome = "rejected"
    else:
        outcome = "in_progress"

    representative = requested or decision or matched[0]
    accepted = None
    reason = None
    if decision is not None:
        accepted = decision.kind == EventKind.ACTION_ACCEPTED
        reason = decision.payload.get("reason")
    if failed is not None:
        reason = failed.payload.get("error") or reason

    request_payload = (
        {} if requested is None else dict(requested.payload.get("action_payload", {}))
    )
    request_metadata = (
        {} if requested is None else dict(requested.payload.get("metadata", {}))
    )
    policy_reason = request_metadata.get("reason", request_metadata.get("explanation"))
    return ActionExplanation(
        action_id=action_id,
        kind=representative.payload.get("kind"),
        target=representative.subject,
        requested_by=None if requested is None else requested.source,
        accepted=accepted,
        outcome=outcome,
        reason=None if reason is None else str(reason),
        lifecycle=tuple(event.kind for event in matched),
        correlation_id=representative.correlation_id,
        request_payload=request_payload,
        request_metadata=request_metadata,
        policy_reason=None if policy_reason is None else str(policy_reason),
    )


def assess_explainability(
    events: Iterable[Event],
) -> ExplainabilityBenchmarkReport:
    """Measure audit coverage of all canonical action lifecycles in a trace."""

    items = tuple(events)
    action_ids = tuple(
        dict.fromkeys(
            str(event.payload["action_id"])
            for event in items
            if event.kind == EventKind.ACTION_REQUESTED
            and event.payload.get("action_id") is not None
        )
    )
    if not action_ids:
        raise ValueError("explainability benchmark requires at least one requested action")
    explanations = tuple(explain_action(items, action_id) for action_id in action_ids)
    decisions = sum(item.accepted is not None for item in explanations)
    terminal_outcomes = {"completed", "failed", "rejected"}
    terminals = sum(item.outcome in terminal_outcomes for item in explanations)
    rationale = sum(bool(item.policy_reason) for item in explanations)
    payload = sum(bool(item.request_payload) for item in explanations)
    negative = tuple(
        item for item in explanations if item.outcome in {"failed", "rejected"}
    )
    negative_reason_coverage = (
        None
        if not negative
        else sum(bool(item.reason) for item in negative) / len(negative)
    )
    outcomes: dict[str, int] = {}
    for item in explanations:
        outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1
    count = len(explanations)
    return ExplainabilityBenchmarkReport(
        actions=count,
        decision_coverage=decisions / count,
        terminal_coverage=terminals / count,
        rationale_coverage=rationale / count,
        payload_coverage=payload / count,
        negative_outcomes=len(negative),
        negative_reason_coverage=negative_reason_coverage,
        outcomes=outcomes,
    )


def explore_scenarios(
    baseline: SimulationResult,
    alternatives: Mapping[str, SimulationResult],
    *,
    metric: str,
    score: Callable[[SimulationResult], float],
) -> ExplorationReport:
    """Compare what-if simulation outcomes with one shared scoring function."""

    baseline_score = float(score(baseline))
    values = {name: float(score(result)) for name, result in alternatives.items()}
    return ExplorationReport(
        metric=metric,
        baseline=baseline_score,
        alternatives=values,
        deltas={name: value - baseline_score for name, value in values.items()},
    )


def assess_robustness(
    baseline: float,
    scenario_scores: Mapping[str, float],
    *,
    higher_is_better: bool = True,
    tolerance: float = 0.1,
) -> RobustnessReport:
    """Quantify worst-case degradation under fault/perturbation scenarios.

    ``tolerance`` is a fractional degradation from the baseline. For a latency
    metric where lower is better set ``higher_is_better=False``.
    """

    if not scenario_scores:
        raise ValueError("robustness assessment requires at least one scenario")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    baseline = float(baseline)
    if abs(baseline) < 1e-12:
        raise ValueError("robustness baseline must be non-zero")
    values = {name: float(value) for name, value in scenario_scores.items()}

    def degradation(value: float) -> float:
        raw = (baseline - value) / abs(baseline)
        return raw if higher_is_better else -raw

    degradations = {name: degradation(value) for name, value in values.items()}
    worst_name = max(degradations, key=degradations.__getitem__)
    return RobustnessReport(
        baseline=baseline,
        scenario_scores=values,
        worst_case=values[worst_name],
        worst_case_scenario=worst_name,
        worst_case_degradation=degradations[worst_name],
        mean_degradation=mean(degradations.values()),
        tolerance=float(tolerance),
        within_tolerance_fraction=(
            sum(value <= tolerance for value in degradations.values()) / len(degradations)
        ),
    )
