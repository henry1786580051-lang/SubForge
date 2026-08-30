"""Deterministic shadow-mode repair planning for typed quality diagnostics."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from subforge.core.translate.quality.diagnostics import (
    DiagnosticCategory,
    QualityDiagnostic,
    RepairStrategy,
)
from subforge.core.translate.quality.session import TranslationSession

SHADOW_REPAIR_SCHEMA_VERSION = 1


class PlanDisposition(str, Enum):
    PLANNED = "planned"
    NO_DIAGNOSTICS = "no_diagnostics"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class PlannedReasoningMode(str, Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class SessionMode(str, Enum):
    UNKNOWN = "unknown"
    MONOLOGUE = "monologue"
    DIALOGUE = "dialogue"


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    supports_reasoning: bool = False


@dataclass(frozen=True, slots=True)
class RepairBudget:
    max_total_attempts: int = 3
    max_reasoning_attempts: int = 1


@dataclass(frozen=True, slots=True)
class RepairHistory:
    total_attempts: int = 0
    reasoning_attempts: int = 0


@dataclass(frozen=True, slots=True)
class RepairPlan:
    disposition: PlanDisposition
    strategy: RepairStrategy
    affected_keys: tuple[int, ...]
    context_radius: int
    reasoning_mode: PlannedReasoningMode
    maximum_attempts: int
    required_rule_ids: tuple[str, ...]
    session_mode: SessionMode
    fallback: str
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShadowRepairObservation:
    plan: RepairPlan
    count: int


@dataclass(frozen=True, slots=True)
class ShadowRepairComparison:
    plan: RepairPlan
    legacy_strategy: RepairStrategy
    legacy_reasoning_mode: PlannedReasoningMode
    matches: bool
    count: int


@dataclass(frozen=True, slots=True)
class ShadowRepairSummary:
    """Immutable, text-free evidence for planner-versus-legacy evaluation."""

    schema_version: int
    recorded_plan_observations: int
    unique_recorded_plans: int
    dropped_plan_observations: int
    recorded_comparison_observations: int
    unique_recorded_comparisons: int
    dropped_comparison_observations: int
    matched_comparisons: int
    mismatched_comparisons: int
    uncompared_recorded_plans: int
    dispositions: tuple[tuple[str, int], ...]
    planned_strategies: tuple[tuple[str, int], ...]
    planned_reasoning_modes: tuple[tuple[str, int], ...]
    session_modes: tuple[tuple[str, int], ...]
    diagnostic_rules: tuple[tuple[str, int], ...]
    comparison_routes: tuple[tuple[str, int], ...]

    @staticmethod
    def _mapping(items: tuple[tuple[str, int], ...]) -> dict[str, int]:
        return {key: value for key, value in items}

    def to_dict(self) -> dict[str, Any]:
        plan_count = self.recorded_plan_observations
        comparison_count = self.recorded_comparison_observations
        return {
            "schema_version": self.schema_version,
            "counts": {
                "recorded_plan_observations": plan_count,
                "unique_recorded_plans": self.unique_recorded_plans,
                "dropped_plan_observations": self.dropped_plan_observations,
                "recorded_comparison_observations": comparison_count,
                "unique_recorded_comparisons": self.unique_recorded_comparisons,
                "dropped_comparison_observations": (
                    self.dropped_comparison_observations
                ),
                "matched_comparisons": self.matched_comparisons,
                "mismatched_comparisons": self.mismatched_comparisons,
                "uncompared_recorded_plans": self.uncompared_recorded_plans,
            },
            "rates": {
                "comparison_coverage": (
                    round(comparison_count / plan_count, 4) if plan_count else 0.0
                ),
                "match_rate": (
                    round(self.matched_comparisons / comparison_count, 4)
                    if comparison_count
                    else 0.0
                ),
            },
            "dispositions": self._mapping(self.dispositions),
            "planned_strategies": self._mapping(self.planned_strategies),
            "planned_reasoning_modes": self._mapping(
                self.planned_reasoning_modes
            ),
            "session_modes": self._mapping(self.session_modes),
            "diagnostic_rules": self._mapping(self.diagnostic_rules),
            "comparison_routes": self._mapping(self.comparison_routes),
        }


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def plan_repair(
    diagnostics: tuple[QualityDiagnostic, ...],
    *,
    capabilities: ProviderCapabilities = ProviderCapabilities(),
    budget: RepairBudget = RepairBudget(),
    history: RepairHistory = RepairHistory(),
    session: TranslationSession | None = None,
    cancelled: bool = False,
) -> RepairPlan:
    """Choose a conservative plan without executing or mutating anything."""
    rule_ids = _unique(tuple(item.rule_id for item in diagnostics))
    affected_keys = tuple(
        sorted({key for diagnostic in diagnostics for key in diagnostic.cue_keys})
    )
    session_mode = (
        SessionMode.UNKNOWN
        if session is None
        else SessionMode.DIALOGUE
        if session.is_multispeaker
        else SessionMode.MONOLOGUE
    )
    if cancelled:
        return RepairPlan(
            disposition=PlanDisposition.CANCELLED,
            strategy=RepairStrategy.NONE,
            affected_keys=affected_keys,
            context_radius=0,
            reasoning_mode=PlannedReasoningMode.DISABLED,
            maximum_attempts=0,
            required_rule_ids=rule_ids,
            session_mode=session_mode,
            fallback="legacy_result",
            rationale=("task_cancelled",),
        )
    if not diagnostics:
        return RepairPlan(
            disposition=PlanDisposition.NO_DIAGNOSTICS,
            strategy=RepairStrategy.NONE,
            affected_keys=(),
            context_radius=0,
            reasoning_mode=PlannedReasoningMode.DISABLED,
            maximum_attempts=0,
            required_rule_ids=(),
            session_mode=session_mode,
            fallback="legacy_result",
            rationale=("no_diagnostics",),
        )
    remaining_attempts = max(0, budget.max_total_attempts - history.total_attempts)
    if remaining_attempts == 0:
        return RepairPlan(
            disposition=PlanDisposition.BUDGET_EXHAUSTED,
            strategy=RepairStrategy.NONE,
            affected_keys=affected_keys,
            context_radius=0,
            reasoning_mode=PlannedReasoningMode.DISABLED,
            maximum_attempts=0,
            required_rule_ids=rule_ids,
            session_mode=session_mode,
            fallback="legacy_result",
            rationale=("total_attempt_budget_exhausted",),
        )

    has_preservation_failure = any(
        rule_id.startswith(("number.", "entity.")) for rule_id in rule_ids
    )
    has_reasoning_candidate = any(
        diagnostic.category in {DiagnosticCategory.OWNERSHIP, DiagnosticCategory.FLUENCY}
        or diagnostic.rule_id.startswith("translation.boundary")
        for diagnostic in diagnostics
    )
    if has_preservation_failure:
        strategy = RepairStrategy.LOCAL_REWRITE
        context_radius = 1
        rationale = ("source_fact_missing", "local_context_only")
    elif all(item.repair_strategy == RepairStrategy.DETERMINISTIC for item in diagnostics):
        strategy = RepairStrategy.DETERMINISTIC
        context_radius = 0
        rationale = ("all_diagnostics_deterministic",)
    else:
        strategy = RepairStrategy.RETRY
        context_radius = 0
        rationale = ("hard_invariant_failed", "preserve_legacy_retry_policy")

    reasoning_available = (
        has_reasoning_candidate
        and capabilities.supports_reasoning
        and history.reasoning_attempts < budget.max_reasoning_attempts
    )
    reasoning_mode = (
        PlannedReasoningMode.ENABLED
        if reasoning_available
        else PlannedReasoningMode.DISABLED
    )
    if reasoning_available:
        context_radius = max(context_radius, 2)
        rationale = (*rationale, "confirmed_high_risk_window")
    if session_mode == SessionMode.DIALOGUE:
        rationale = (*rationale, "dialogue_context_is_read_only")

    return RepairPlan(
        disposition=PlanDisposition.PLANNED,
        strategy=strategy,
        affected_keys=affected_keys,
        context_radius=context_radius,
        reasoning_mode=reasoning_mode,
        maximum_attempts=min(remaining_attempts, 1),
        required_rule_ids=rule_ids,
        session_mode=session_mode,
        fallback="legacy_result",
        rationale=rationale,
    )


class ShadowRepairRecorder:
    """Aggregate plans without logging subtitle text or growing without bound."""

    def __init__(self, max_unique_plans: int = 128):
        if max_unique_plans <= 0:
            raise ValueError("max_unique_plans must be positive")
        self._max_unique_plans = max_unique_plans
        self._lock = threading.Lock()
        self._observations: dict[RepairPlan, int] = {}
        self._comparisons: dict[
            tuple[RepairPlan, RepairStrategy, PlannedReasoningMode], int
        ] = {}
        self._attempts: dict[tuple[tuple[str, ...], tuple[int, ...]], RepairHistory] = {}
        self._dropped_unique_plans = 0
        self._dropped_unique_comparisons = 0

    @staticmethod
    def _signature(
        diagnostics: tuple[QualityDiagnostic, ...],
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        return (
            _unique(tuple(item.rule_id for item in diagnostics)),
            tuple(sorted({key for item in diagnostics for key in item.cue_keys})),
        )

    def _record_locked(self, plan: RepairPlan) -> None:
        if plan in self._observations:
            self._observations[plan] += 1
            return
        if len(self._observations) >= self._max_unique_plans:
            self._dropped_unique_plans += 1
            return
        self._observations[plan] = 1

    def record(self, plan: RepairPlan) -> None:
        with self._lock:
            self._record_locked(plan)

    def plan_and_record(
        self,
        diagnostics: tuple[QualityDiagnostic, ...],
        *,
        capabilities: ProviderCapabilities = ProviderCapabilities(),
        budget: RepairBudget = RepairBudget(),
        session: TranslationSession | None = None,
        cancelled: bool = False,
    ) -> RepairPlan:
        """Atomically plan from prior shadow observations and record the decision."""
        signature = self._signature(diagnostics)
        with self._lock:
            history = self._attempts.get(signature, RepairHistory())
            plan = plan_repair(
                diagnostics,
                capabilities=capabilities,
                budget=budget,
                history=history,
                session=session,
                cancelled=cancelled,
            )
            if signature in self._attempts or len(self._attempts) < self._max_unique_plans:
                self._attempts[signature] = RepairHistory(
                    total_attempts=history.total_attempts + 1,
                    reasoning_attempts=(
                        history.reasoning_attempts
                        + int(plan.reasoning_mode == PlannedReasoningMode.ENABLED)
                    ),
                )
            self._record_locked(plan)
            return plan

    def snapshot(self) -> tuple[ShadowRepairObservation, ...]:
        with self._lock:
            return tuple(
                ShadowRepairObservation(plan=plan, count=count)
                for plan, count in self._observations.items()
            )

    def record_legacy_action(
        self,
        plan: RepairPlan,
        *,
        strategy: RepairStrategy,
        reasoning_mode: PlannedReasoningMode,
    ) -> None:
        """Compare one observed legacy action with a previously recorded plan."""
        key = (plan, strategy, reasoning_mode)
        with self._lock:
            if key in self._comparisons:
                self._comparisons[key] += 1
            elif len(self._comparisons) >= self._max_unique_plans:
                self._dropped_unique_comparisons += 1
            else:
                self._comparisons[key] = 1

    def comparison_snapshot(self) -> tuple[ShadowRepairComparison, ...]:
        with self._lock:
            return tuple(
                ShadowRepairComparison(
                    plan=plan,
                    legacy_strategy=strategy,
                    legacy_reasoning_mode=reasoning_mode,
                    matches=(
                        plan.strategy == strategy
                        and plan.reasoning_mode == reasoning_mode
                    ),
                    count=count,
                )
                for (plan, strategy, reasoning_mode), count in self._comparisons.items()
            )

    def summary(self) -> ShadowRepairSummary:
        """Aggregate bounded shadow evidence without retaining cue text or keys."""

        with self._lock:
            observations = tuple(self._observations.items())
            comparisons = tuple(self._comparisons.items())
            dispositions: dict[str, int] = {}
            strategies: dict[str, int] = {}
            reasoning_modes: dict[str, int] = {}
            session_modes: dict[str, int] = {}
            diagnostic_rules: dict[str, int] = {}
            comparison_routes: dict[str, int] = {}

            recorded_plans = 0
            for plan, count in observations:
                recorded_plans += count
                for mapping, key in (
                    (dispositions, plan.disposition.value),
                    (strategies, plan.strategy.value),
                    (reasoning_modes, plan.reasoning_mode.value),
                    (session_modes, plan.session_mode.value),
                ):
                    mapping[key] = mapping.get(key, 0) + count
                for rule_id in plan.required_rule_ids:
                    diagnostic_rules[rule_id] = diagnostic_rules.get(rule_id, 0) + count

            recorded_comparisons = 0
            matched_comparisons = 0
            for (plan, strategy, reasoning_mode), count in comparisons:
                recorded_comparisons += count
                matches = (
                    plan.strategy == strategy
                    and plan.reasoning_mode == reasoning_mode
                )
                matched_comparisons += count if matches else 0
                route = (
                    f"{plan.strategy.value}/{plan.reasoning_mode.value}"
                    f"->{strategy.value}/{reasoning_mode.value}"
                )
                comparison_routes[route] = comparison_routes.get(route, 0) + count

            def sorted_counts(values: dict[str, int]) -> tuple[tuple[str, int], ...]:
                return tuple(sorted(values.items()))

            return ShadowRepairSummary(
                schema_version=SHADOW_REPAIR_SCHEMA_VERSION,
                recorded_plan_observations=recorded_plans,
                unique_recorded_plans=len(observations),
                dropped_plan_observations=self._dropped_unique_plans,
                recorded_comparison_observations=recorded_comparisons,
                unique_recorded_comparisons=len(comparisons),
                dropped_comparison_observations=self._dropped_unique_comparisons,
                matched_comparisons=matched_comparisons,
                mismatched_comparisons=recorded_comparisons - matched_comparisons,
                uncompared_recorded_plans=max(0, recorded_plans - recorded_comparisons),
                dispositions=sorted_counts(dispositions),
                planned_strategies=sorted_counts(strategies),
                planned_reasoning_modes=sorted_counts(reasoning_modes),
                session_modes=sorted_counts(session_modes),
                diagnostic_rules=sorted_counts(diagnostic_rules),
                comparison_routes=sorted_counts(comparison_routes),
            )

    @property
    def dropped_unique_plans(self) -> int:
        with self._lock:
            return self._dropped_unique_plans

    @property
    def dropped_unique_comparisons(self) -> int:
        with self._lock:
            return self._dropped_unique_comparisons
