import json

import pytest

from scripts.translation_quality.efficiency import (
    aggregate_efficiency_payloads,
    load_efficiency_payload,
    write_efficiency_report,
)


def _snapshot(task_id: str, *, tokens: int, cached_tokens: int):
    return {
        "schema_version": 1,
        "task_id": task_id,
        "workload_id": f"source-{task_id}",
        "pipeline": {"variant": "candidate", "revision": "phase8-r1"},
        "cache_state": "explicit-client-no-disk-cache",
        "metrics": {
            "wall_duration_ms": 1000,
            "request_attempts": 2,
            "successful_requests": 1,
            "failed_requests": 1,
            "api_duration_ms": 800,
            "retry_count": 1,
            "retry_wait_ms": 200,
            "rate_limit_retries": 1,
            "transient_retries": 0,
            "tokens": tokens,
            "prompt_tokens": 100,
            "cached_tokens": cached_tokens,
            "cache_creation_tokens": 0,
            "completion_tokens": 50,
            "reasoning_tokens": 10,
            "reasoning_enabled_requests": 1,
            "reasoning_disabled_requests": 0,
            "reasoning_default_requests": 1,
        },
        "stages": [],
    }


def _shadow(*, matched: int, mismatched: int):
    comparisons = matched + mismatched
    return {
        "schema_version": 1,
        "counts": {
            "recorded_plan_observations": comparisons,
            "unique_recorded_plans": 1,
            "dropped_plan_observations": 0,
            "recorded_comparison_observations": comparisons,
            "unique_recorded_comparisons": 1,
            "dropped_comparison_observations": 0,
            "matched_comparisons": matched,
            "mismatched_comparisons": mismatched,
            "uncompared_recorded_plans": 0,
        },
        "rates": {"comparison_coverage": 1.0, "match_rate": 0.0},
        "dispositions": {"planned": comparisons},
        "planned_strategies": {"retry": comparisons},
        "planned_reasoning_modes": {"disabled": comparisons},
        "session_modes": {"monologue": comparisons},
        "diagnostic_rules": {"translation.empty": comparisons},
        "comparison_routes": {"retry/disabled->retry/disabled": comparisons},
    }


def _canonical_evidence(*, supported: int, rejected: int):
    matches = supported + rejected
    return {
        "schema_version": 1,
        "counts": {
            "terminology_line_count": 4,
            "asr_labeled_line_count": 2,
            "parseable_mapping_count": 2,
            "mapping_with_source_match_count": 1 if matches else 0,
            "source_mapping_match_count": matches,
            "supported_source_mapping_match_count": supported,
            "rejected_source_mapping_match_count": rejected,
        },
    }


def test_efficiency_aggregation_sums_snapshots_and_recomputes_cache_rate(tmp_path):
    payload = aggregate_efficiency_payloads(
        (
            _snapshot("task-1", tokens=150, cached_tokens=25),
            _snapshot("task-2", tokens=160, cached_tokens=75),
        )
    )

    assert payload["snapshot_count"] == 2
    assert payload["pipeline"] == {
        "variant": "candidate",
        "revision": "phase8-r1",
    }
    assert payload["aggregate"]["tokens"] == 310
    assert payload["aggregate"]["provider_cache_hit_rate"] == 0.5

    path = tmp_path / "efficiency.json"
    write_efficiency_report(payload, path)
    assert load_efficiency_payload(path) == json.loads(path.read_text(encoding="utf-8"))


def test_efficiency_aggregation_rejects_duplicate_task_ids():
    with pytest.raises(ValueError, match="Duplicate efficiency task id"):
        aggregate_efficiency_payloads(
            (
                _snapshot("same", tokens=100, cached_tokens=0),
                _snapshot("same", tokens=100, cached_tokens=0),
            )
        )


def test_efficiency_aggregation_includes_repair_shadow_evidence():
    first = _snapshot("task-1", tokens=100, cached_tokens=0)
    second = _snapshot("task-2", tokens=100, cached_tokens=0)
    first["repair_shadow"] = _shadow(matched=2, mismatched=0)
    second["repair_shadow"] = _shadow(matched=1, mismatched=1)

    payload = aggregate_efficiency_payloads((first, second))

    shadow = payload["repair_shadow"]
    assert shadow["snapshot_count"] == 2
    assert shadow["counts"]["matched_comparisons"] == 3
    assert shadow["counts"]["mismatched_comparisons"] == 1
    assert shadow["rates"] == {
        "comparison_coverage": 1.0,
        "match_rate": 0.75,
    }


def test_efficiency_aggregation_rejects_mixed_shadow_availability():
    first = _snapshot("task-1", tokens=100, cached_tokens=0)
    second = _snapshot("task-2", tokens=100, cached_tokens=0)
    first["repair_shadow"] = _shadow(matched=1, mismatched=0)

    with pytest.raises(ValueError, match="mix present and missing"):
        aggregate_efficiency_payloads((first, second))


def test_efficiency_aggregation_includes_text_free_canonical_evidence():
    first = _snapshot("task-1", tokens=100, cached_tokens=0)
    second = _snapshot("task-2", tokens=100, cached_tokens=0)
    first["canonical_evidence"] = _canonical_evidence(supported=1, rejected=2)
    second["canonical_evidence"] = _canonical_evidence(supported=2, rejected=0)

    payload = aggregate_efficiency_payloads((first, second))

    evidence = payload["canonical_evidence"]
    assert evidence["snapshot_count"] == 2
    assert evidence["counts"]["source_mapping_match_count"] == 5
    assert evidence["counts"]["supported_source_mapping_match_count"] == 3
    assert evidence["counts"]["rejected_source_mapping_match_count"] == 2


def test_efficiency_aggregation_rejects_mixed_canonical_evidence_availability():
    first = _snapshot("task-1", tokens=100, cached_tokens=0)
    second = _snapshot("task-2", tokens=100, cached_tokens=0)
    first["canonical_evidence"] = _canonical_evidence(supported=1, rejected=0)

    with pytest.raises(ValueError, match="mix present and missing canonical evidence"):
        aggregate_efficiency_payloads((first, second))

