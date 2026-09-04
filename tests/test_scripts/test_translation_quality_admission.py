import copy
import hashlib
import json

import pytest

from scripts.evaluate_translation_quality import _parser
from scripts.translation_quality.admission import (
    assess_admission,
    load_admission_policy,
    validate_admission_policy,
)
from scripts.translation_quality.comparison import compare_evaluation_reports


def _report():
    aggregate = dict.fromkeys(
        (
            "hard_failure_count",
            "empty_targets",
            "placeholder_targets",
            "reasoning_leaks",
            "source_copy_targets",
            "untranslated_targets",
            "adjacent_duplicate_risks",
            "requires_alignment_samples",
            "human_changed_cues",
        ),
        0,
    )
    aggregate.update(
        dict.fromkeys(
            (
                "sample_count",
                "source_cue_count",
                "machine_cue_count",
                "gold_cue_count",
                "structurally_exact_samples",
            ),
            1,
        )
    )
    return {
        "corpus_id": "fixture",
        "manifest_hash": "fixture",
        "comparison_identity": "same",
        "aggregate": aggregate,
        "samples": [
            {
                "sample_id": "one",
                "machine": {
                    "empty_targets": [],
                    "placeholder_targets": [],
                    "reasoning_leaks": [],
                },
            }
        ],
    }


def _efficiency():
    return {
        "schema_version": 1,
        "cache_state": "disabled",
        "workload_identity": "same",
        "snapshot_count": 1,
        "aggregate": {
            "tokens": 1000,
            "wall_duration_ms": 10000,
            "successful_requests": 10,
            "request_attempts": 10,
            "reasoning_enabled_requests": 2,
            "reasoning_tokens": 100,
        },
    }


def _policy(track="quality"):
    return {
        "schema_version": 2,
        "track": track,
        "candidate_revision": "test-v1",
        "affected_modes": ["single"],
        "budgets": {
            "tokens": {"max_ratio": 1.1, "absolute_allowance": 0},
            "wall_duration_ms": {"max_ratio": 1.15, "absolute_allowance": 0},
        },
    }


def _assess(old=None, new=None, before=None, after=None, policy=None):
    old = _report() if old is None else old
    new = _report() if new is None else new
    before = _efficiency() if before is None else before
    after = _efficiency() if after is None else after
    comparison = compare_evaluation_reports(
        old,
        new,
        legacy_efficiency=before,
        candidate_efficiency=after,
    )
    return comparison, assess_admission(
        comparison,
        old,
        new,
        policy or _policy(),
        legacy_efficiency=before,
        candidate_efficiency=after,
    )


def test_quality_gain_can_be_reviewed_above_old_budget_without_changing_history():
    after = _efficiency()
    after["aggregate"].update(tokens=1070, wall_duration_ms=10800, reasoning_enabled_requests=3)
    historical, result = _assess(after=after)
    assert historical["accepted"] is False
    assert result["decision"] == "review"
    assert result["production_adoption"] is False
    assert "independent_mode_scoped_quality_review" in result["required_review"]


@pytest.mark.parametrize(
    "metric", ["untranslated_targets", "adjacent_duplicate_risks", "source_copy_targets"]
)
def test_heuristic_signals_require_review_not_automatic_rejection(metric):
    new = _report()
    new["aggregate"][metric] = 1
    _, result = _assess(new=new)
    assert result["decision"] == "observe"
    assert not result["blockers"]


@pytest.mark.parametrize("metric", ["empty_targets", "placeholder_targets", "reasoning_leaks"])
def test_per_sample_hard_failure_cannot_be_hidden_by_aggregate_improvement(metric):
    old, new = _report(), _report()
    old["aggregate"][metric] = 2
    new["aggregate"][metric] = 1
    new["samples"][0]["machine"][metric] = [1]
    _, result = _assess(old=old, new=new)
    assert result["decision"] == "blocked"
    assert f"new_hard_signal:{metric}" in result["blockers"]


def test_existing_defect_does_not_require_candidate_to_fix_entire_product():
    old, new = _report(), _report()
    old["samples"][0]["machine"]["empty_targets"] = [1, 2]
    new["samples"][0]["machine"]["empty_targets"] = [2]
    _, result = _assess(old=old, new=new, policy=_policy("bugfix"))
    assert result["decision"] == "observe"
    assert not result["blockers"]


def test_an_over_budget_candidate_is_retained_for_observation_not_adopted():
    after = _efficiency()
    after["aggregate"]["tokens"] = 1200
    _, result = _assess(after=after)
    assert result["decision"] == "observe"
    assert result["budgets"]["tokens"]["status"] == "exceeded"
    assert result["production_adoption"] is False


def test_efficiency_track_requires_observed_efficiency_gain():
    _, result = _assess(policy=_policy("efficiency"))
    assert "no_observed_efficiency_gain" in result["observations"]
    after = _efficiency()
    after["aggregate"]["tokens"] = 900
    _, result = _assess(after=after, policy=_policy("efficiency"))
    assert result["decision"] == "review"


def test_zero_baseline_has_explicit_absolute_reasoning_allowance():
    policy = _policy()
    policy["budgets"]["reasoning_tokens"] = {"max_ratio": 1, "absolute_allowance": 100}
    before = _efficiency()
    before["aggregate"]["reasoning_tokens"] = 0
    _, result = _assess(before=before, policy=policy)
    assert result["budgets"]["reasoning_tokens"]["status"] == "within"


def test_missing_reasoning_usage_is_unknown_not_zero():
    policy = _policy()
    policy["budgets"]["reasoning_tokens"] = {"max_ratio": 1.1, "absolute_allowance": 0}
    after = _efficiency()
    del after["aggregate"]["reasoning_tokens"]
    _, result = _assess(after=after, policy=policy)
    assert result["budgets"]["reasoning_tokens"] == {"status": "unknown"}
    assert result["decision"] == "observe"


def test_changed_identity_blocks_comparison():
    new = _report()
    new["comparison_identity"] = "changed"
    _, result = _assess(new=new)
    assert result["decision"] == "blocked"


def test_holdout_redacted_counts_supported_without_reading_cue_text():
    old = _report()
    old["samples"][0]["machine"] = dict.fromkeys(
        (
            "empty_targets_count",
            "placeholder_targets_count",
            "reasoning_leaks_count",
        ),
        0,
    )
    _, result = _assess(old=old, new=copy.deepcopy(old))
    assert result["decision"] == "review"


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), "1.1"])
def test_invalid_budget_is_rejected(bad):
    policy = _policy()
    policy["budgets"]["tokens"]["max_ratio"] = bad
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_admission_policy(policy)


@pytest.mark.parametrize(
    "field,value", [("track", []), ("schema_version", 2.0), ("affected_modes", []), ("budgets", {})]
)
def test_malformed_policy_rejected(field, value):
    policy = _policy()
    policy[field] = value
    with pytest.raises(ValueError):
        validate_admission_policy(policy)


def test_policy_hash_prevents_silent_budget_edit(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_admission_policy(path, sha) == _policy()
    path.write_text(json.dumps(_policy("bugfix")))
    with pytest.raises(ValueError, match="SHA-256"):
        load_admission_policy(path, sha)


@pytest.mark.parametrize(
    "tokens,hard_signal,decision,exit_code",
    [(1070, False, "review", 0), (1200, False, "observe", 2), (1070, True, "blocked", 3)],
)
def test_v2_cli_emits_screening_and_retains_historical_verdict(
    tmp_path,
    capsys,
    tokens,
    hard_signal,
    decision,
    exit_code,
):
    reports = {
        "legacy-report": _report(),
        "candidate-report": _report(),
        "legacy-efficiency": _efficiency(),
        "candidate-efficiency": _efficiency(),
        "policy": _policy(),
    }
    reports["candidate-efficiency"]["aggregate"]["tokens"] = tokens
    if hard_signal:
        reports["candidate-report"]["samples"][0]["machine"]["empty_targets"] = [1]
    args = ["compare", "--output-dir", str(tmp_path / "result"), "--fail-on-regression"]
    for flag, payload in reports.items():
        path = tmp_path / f"{flag}.json"
        path.write_text(json.dumps(payload))
        args.extend([f"--{flag}", str(path)])
    args.extend(
        ["--policy-sha256", hashlib.sha256((tmp_path / "policy.json").read_bytes()).hexdigest()]
    )
    parsed = _parser().parse_args(args)
    assert parsed.handler(parsed) == exit_code
    report = json.loads((tmp_path / "result/comparison.json").read_text())
    assert report["accepted"] is False
    assert report["admission"]["decision"] == decision
    assert "PRODUCTION_ADOPTION=not_assessed" in capsys.readouterr().out
    assert "Historical v1 Gates" in (tmp_path / "result/comparison.md").read_text()


def test_budget_ceiling_overflow_cannot_authorize_unlimited_cost():
    policy = _policy()
    policy["budgets"]["tokens"]["max_ratio"] = 1e308
    with pytest.raises(ValueError, match="ceiling"):
        _assess(policy=policy)


def test_empty_or_missing_sample_evidence_stays_observation():
    old, new = _report(), _report()
    old["samples"] = new["samples"] = []
    _, result = _assess(old=old, new=new)
    assert result["decision"] == "observe"
    new = _report()
    new["samples"][0]["machine"] = None
    # The existing comparison requires a machine object; admission itself also
    # treats incomplete evidence as unknown rather than proving safety.
    comparison, _ = _assess()
    result = assess_admission(comparison, _report(), new, _policy())
    assert "missing_sample_evidence:empty_targets" in result["observations"]


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1])
def test_non_finite_and_negative_metrics_rejected(value):
    after = _efficiency()
    after["aggregate"]["tokens"] = value
    with pytest.raises(ValueError, match="finite and non-negative"):
        _assess(after=after)
