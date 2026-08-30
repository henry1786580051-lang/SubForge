import json

import pytest

from scripts.translation_quality.comparison import (
    compare_evaluation_reports,
    write_comparison_report,
)


def _report(*, hard_failures: int = 0, duplicates: int = 0):
    aggregate = {
        "sample_count": 1,
        "source_cue_count": 2,
        "machine_cue_count": 2,
        "gold_cue_count": 2,
        "structurally_exact_samples": 1,
        "hard_failure_count": hard_failures,
        "empty_targets": 0,
        "placeholder_targets": 0,
        "reasoning_leaks": 0,
        "source_copy_targets": 0,
        "untranslated_targets": 0,
        "adjacent_duplicate_risks": duplicates,
        "human_changed_cues": 1,
        "requires_alignment_samples": 0,
    }
    return {
        "corpus_id": "fixture",
        "manifest_hash": "same-hash",
        "aggregate": aggregate,
        "samples": [
            {
                "sample_id": "sample-1",
                "title": "Fixture | Sample",
                "split": "development",
                "machine": {"cue_count": 2},
                "hard_failure_count": hard_failures,
            }
        ],
    }


def _efficiency(*, calls: int = 20, tokens: int = 1000, latency: int = 10_000):
    return {
        "schema_version": 1,
        "cache_state": "disabled",
        "snapshot_count": 2,
        "workload_identity": "same-workload",
        "aggregate": {
            "successful_requests": calls,
            "request_attempts": calls,
            "tokens": tokens,
            "wall_duration_ms": latency,
            "reasoning_enabled_requests": 2,
        },
    }


def _repair_shadow(*, plans: int = 2, comparisons: int = 2, dropped: int = 0):
    return {
        "schema_version": 1,
        "snapshot_count": 2,
        "counts": {
            "recorded_plan_observations": plans,
            "dropped_plan_observations": dropped,
            "recorded_comparison_observations": comparisons,
            "dropped_comparison_observations": dropped,
            "matched_comparisons": max(0, comparisons - 1),
            "mismatched_comparisons": min(1, comparisons),
            "uncompared_recorded_plans": max(0, plans - comparisons),
        },
        "rates": {"comparison_coverage": comparisons / plans if plans else 0.0},
        "comparison_routes": {"retry/disabled->retry/disabled": comparisons},
    }


def test_comparison_accepts_equal_or_improved_candidate_and_writes_text_free_report(tmp_path):
    payload = compare_evaluation_reports(
        _report(duplicates=1),
        _report(duplicates=0),
    )

    assert payload["accepted"] is True
    assert payload["metrics"]["adjacent_duplicate_risks"] == {
        "legacy": 1,
        "candidate": 0,
        "delta": -1,
    }

    write_comparison_report(payload, tmp_path)
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert saved["accepted"] is True
    assert "Fixture \\| Sample" in markdown
    assert "Detailed cue text is intentionally omitted" in markdown


def test_comparison_rejects_candidate_hard_regression():
    payload = compare_evaluation_reports(
        _report(hard_failures=0),
        _report(hard_failures=1),
    )

    assert payload["accepted"] is False
    assert payload["gates"]["hard_failure_count"] is False


def test_comparison_rejects_different_sample_identity():
    candidate = _report()
    candidate["samples"][0]["sample_id"] = "other"

    with pytest.raises(ValueError, match="different sample sets"):
        compare_evaluation_reports(_report(), candidate)


def test_comparison_rejects_changed_source_or_gold_identity():
    legacy = _report()
    candidate = _report()
    legacy["comparison_identity"] = "source-gold-a"
    candidate["comparison_identity"] = "source-gold-b"

    payload = compare_evaluation_reports(legacy, candidate)

    assert payload["accepted"] is False
    assert payload["gates"]["corpus_identity"] is False


def test_comparison_accepts_efficiency_at_five_percent_budget():
    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=_efficiency(),
        candidate_efficiency=_efficiency(calls=21, tokens=1050, latency=10_500),
    )

    assert payload["accepted"] is True
    assert payload["efficiency"]["accepted"] is True
    assert payload["gates"]["efficiency_tokens"] is True


def test_comparison_rejects_efficiency_or_reasoning_regression():
    candidate = _efficiency(calls=22, tokens=1060, latency=10_600)
    candidate["aggregate"]["reasoning_enabled_requests"] = 3

    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=_efficiency(),
        candidate_efficiency=candidate,
    )

    assert payload["accepted"] is False
    assert payload["gates"]["efficiency_successful_requests"] is False
    assert payload["gates"]["efficiency_tokens"] is False
    assert payload["gates"]["efficiency_reasoning_enabled_requests"] is False


def test_comparison_requires_both_efficiency_reports():
    with pytest.raises(ValueError, match="must be supplied together"):
        compare_evaluation_reports(
            _report(),
            _report(),
            legacy_efficiency=_efficiency(),
        )


def test_comparison_rejects_different_efficiency_workloads():
    candidate = _efficiency()
    candidate["workload_identity"] = "other-workload"

    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=_efficiency(),
        candidate_efficiency=candidate,
    )

    assert payload["accepted"] is False
    assert payload["gates"]["efficiency_workload_identity"] is False


def test_comparison_accepts_complete_repair_shadow_evidence():
    legacy = _efficiency()
    candidate = _efficiency()
    legacy["repair_shadow"] = _repair_shadow()
    candidate["repair_shadow"] = _repair_shadow()

    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=legacy,
        candidate_efficiency=candidate,
    )

    assert payload["accepted"] is True
    assert payload["efficiency"]["repair_shadow"]["accepted"] is True
    assert payload["gates"]["efficiency_repair_shadow_candidate_observed_plans"] is True


def test_comparison_rejects_incomplete_repair_shadow_evidence():
    legacy = _efficiency()
    candidate = _efficiency()
    legacy["repair_shadow"] = _repair_shadow()
    candidate["repair_shadow"] = _repair_shadow(plans=2, comparisons=1, dropped=1)

    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=legacy,
        candidate_efficiency=candidate,
    )

    assert payload["accepted"] is False
    gates = payload["efficiency"]["repair_shadow"]["gates"]
    assert gates["candidate_full_comparison_coverage"] is False
    assert gates["candidate_no_dropped_plans"] is False


def test_comparison_rejects_one_sided_repair_shadow_evidence():
    legacy = _efficiency()
    candidate = _efficiency()
    candidate["repair_shadow"] = _repair_shadow()

    payload = compare_evaluation_reports(
        _report(),
        _report(),
        legacy_efficiency=legacy,
        candidate_efficiency=candidate,
    )

    assert payload["accepted"] is False
    shadow = payload["efficiency"]["repair_shadow"]
    assert shadow["available_for_both"] is False
    assert shadow["gates"] == {"available_for_both": False}
