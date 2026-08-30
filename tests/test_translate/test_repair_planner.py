from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate.quality import (
    DiagnosticCategory,
    DiagnosticSeverity,
    PlanDisposition,
    PlannedReasoningMode,
    PreservedTokenViolation,
    ProviderCapabilities,
    QualityDiagnostic,
    RepairBudget,
    RepairHistory,
    RepairStrategy,
    SessionMode,
    ShadowRepairRecorder,
    build_translation_session,
    inspect_preserved_token_violations,
    plan_repair,
)
from subforge.core.translate.types import TargetLanguage


def _diagnostic(
    rule_id: str,
    *,
    category: DiagnosticCategory = DiagnosticCategory.COMPLETENESS,
    cue_keys: tuple[int, ...] = (1,),
    repair_strategy: RepairStrategy = RepairStrategy.RETRY,
) -> QualityDiagnostic:
    return QualityDiagnostic(
        rule_id=rule_id,
        category=category,
        severity=DiagnosticSeverity.ERROR,
        confidence=1.0,
        cue_keys=cue_keys,
        evidence=(),
        repair_strategy=repair_strategy,
        message="legacy feedback",
    )


def test_planner_returns_noop_without_diagnostics():
    plan = plan_repair(())

    assert plan.disposition == PlanDisposition.NO_DIAGNOSTICS
    assert plan.strategy == RepairStrategy.NONE
    assert plan.maximum_attempts == 0


def test_planner_keeps_hard_schema_failures_on_legacy_retry_policy():
    plan = plan_repair((_diagnostic("schema.missing_key", cue_keys=(2, 3)),))

    assert plan.disposition == PlanDisposition.PLANNED
    assert plan.strategy == RepairStrategy.RETRY
    assert plan.affected_keys == (2, 3)
    assert plan.reasoning_mode == PlannedReasoningMode.DISABLED
    assert plan.maximum_attempts == 1
    assert plan.fallback == "legacy_result"


def test_planner_uses_local_non_reasoning_rewrite_for_missing_source_facts():
    diagnostics = inspect_preserved_token_violations(
        (PreservedTokenViolation("7", "2026", "number"),)
    )

    plan = plan_repair(
        diagnostics,
        capabilities=ProviderCapabilities(supports_reasoning=True),
    )

    assert plan.strategy == RepairStrategy.LOCAL_REWRITE
    assert plan.context_radius == 1
    assert plan.reasoning_mode == PlannedReasoningMode.DISABLED
    assert plan.required_rule_ids == ("number.value_missing",)


def test_planner_reserves_reasoning_for_confirmed_high_risk_windows():
    plan = plan_repair(
        (
            _diagnostic(
                "translation.boundary.target.reporting_complement",
                category=DiagnosticCategory.OWNERSHIP,
                cue_keys=(9, 10),
                repair_strategy=RepairStrategy.LOCAL_REWRITE,
            ),
        ),
        capabilities=ProviderCapabilities(supports_reasoning=True),
    )

    assert plan.reasoning_mode == PlannedReasoningMode.ENABLED
    assert plan.context_radius == 2
    assert "confirmed_high_risk_window" in plan.rationale


def test_planner_records_dialogue_session_without_changing_repair_tier():
    session = build_translation_session(
        ASRData(
            [
                ASRDataSeg("Question", 0, 500, speaker_id="host"),
                ASRDataSeg("Answer", 600, 1100, speaker_id="guest"),
            ]
        ),
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
    )

    plan = plan_repair(
        (_diagnostic("translation.empty"),),
        session=session,
    )

    assert plan.session_mode == SessionMode.DIALOGUE
    assert plan.strategy == RepairStrategy.RETRY
    assert "dialogue_context_is_read_only" in plan.rationale


def test_planner_fails_closed_when_attempt_budget_is_exhausted():
    plan = plan_repair(
        (_diagnostic("translation.empty"),),
        budget=RepairBudget(max_total_attempts=3),
        history=RepairHistory(total_attempts=3),
    )

    assert plan.disposition == PlanDisposition.BUDGET_EXHAUSTED
    assert plan.strategy == RepairStrategy.NONE
    assert plan.maximum_attempts == 0


def test_shadow_recorder_aggregates_equal_plans_and_caps_unique_entries():
    recorder = ShadowRepairRecorder(max_unique_plans=1)
    first = plan_repair((_diagnostic("translation.empty"),))
    second = plan_repair((_diagnostic("translation.placeholder", cue_keys=(2,)),))

    recorder.record(first)
    recorder.record(first)
    recorder.record(second)

    observations = recorder.snapshot()
    assert len(observations) == 1
    assert observations[0].count == 2
    assert recorder.dropped_unique_plans == 1


def test_shadow_recorder_applies_attempt_history_without_executing_plans():
    recorder = ShadowRepairRecorder()
    diagnostics = (_diagnostic("translation.empty"),)

    plans = [
        recorder.plan_and_record(
            diagnostics,
            budget=RepairBudget(max_total_attempts=3),
        )
        for _ in range(4)
    ]

    assert [plan.disposition for plan in plans] == [
        PlanDisposition.PLANNED,
        PlanDisposition.PLANNED,
        PlanDisposition.PLANNED,
        PlanDisposition.BUDGET_EXHAUSTED,
    ]
    assert sum(item.count for item in recorder.snapshot()) == 4


def test_shadow_recorder_compares_plan_with_observed_legacy_action():
    recorder = ShadowRepairRecorder()
    plan = recorder.plan_and_record(
        inspect_preserved_token_violations(
            (PreservedTokenViolation("5", "GTI", "entity"),)
        )
    )

    recorder.record_legacy_action(
        plan,
        strategy=RepairStrategy.RETRY,
        reasoning_mode=PlannedReasoningMode.DISABLED,
    )

    comparisons = recorder.comparison_snapshot()
    assert len(comparisons) == 1
    assert comparisons[0].plan.strategy == RepairStrategy.LOCAL_REWRITE
    assert comparisons[0].legacy_strategy == RepairStrategy.RETRY
    assert comparisons[0].matches is False
    assert comparisons[0].count == 1


def test_shadow_summary_is_text_free_and_exposes_admission_evidence():
    recorder = ShadowRepairRecorder()
    plan = recorder.plan_and_record(
        inspect_preserved_token_violations(
            (PreservedTokenViolation("5", "GTI", "entity"),)
        )
    )
    recorder.record_legacy_action(
        plan,
        strategy=RepairStrategy.RETRY,
        reasoning_mode=PlannedReasoningMode.DISABLED,
    )

    payload = recorder.summary().to_dict()

    assert payload["counts"] == {
        "recorded_plan_observations": 1,
        "unique_recorded_plans": 1,
        "dropped_plan_observations": 0,
        "recorded_comparison_observations": 1,
        "unique_recorded_comparisons": 1,
        "dropped_comparison_observations": 0,
        "matched_comparisons": 0,
        "mismatched_comparisons": 1,
        "uncompared_recorded_plans": 0,
    }
    assert payload["rates"] == {
        "comparison_coverage": 1.0,
        "match_rate": 0.0,
    }
    assert payload["comparison_routes"] == {"local_rewrite/disabled->retry/disabled": 1}
    assert "GTI" not in str(payload)
    assert "affected_keys" not in str(payload)


def test_shadow_recorder_bounds_unique_comparison_storage():
    recorder = ShadowRepairRecorder(max_unique_plans=1)
    first = recorder.plan_and_record((_diagnostic("translation.empty"),))
    second = recorder.plan_and_record(
        (_diagnostic("translation.placeholder", cue_keys=(2,)),)
    )

    recorder.record_legacy_action(
        first,
        strategy=RepairStrategy.RETRY,
        reasoning_mode=PlannedReasoningMode.DISABLED,
    )
    recorder.record_legacy_action(
        second,
        strategy=RepairStrategy.RETRY,
        reasoning_mode=PlannedReasoningMode.DISABLED,
    )

    assert len(recorder.comparison_snapshot()) == 1
    assert recorder.dropped_unique_comparisons == 1
    assert recorder.summary().to_dict()["counts"]["dropped_comparison_observations"] == 1
