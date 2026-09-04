"""Prospective, track-specific screening; never authorizes production adoption."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

TRACKS = {"bugfix", "quality", "efficiency"}
BUDGET_METRICS = {
    "tokens",
    "wall_duration_ms",
    "successful_requests",
    "request_attempts",
    "reasoning_tokens",
    "reasoning_enabled_requests",
}
HARD_SIGNALS = ("empty_targets", "placeholder_targets", "reasoning_leaks")
REVIEW_SIGNALS = (
    "source_copy_targets",
    "untranslated_targets",
    "adjacent_duplicate_risks",
)


def _number(value: Any, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def load_admission_policy(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Admission policy SHA-256 does not match the frozen policy")
    policy = json.loads(raw)
    validate_admission_policy(policy)
    return policy


def validate_admission_policy(policy: Any) -> None:
    fields = {"schema_version", "track", "candidate_revision", "affected_modes", "budgets"}
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError("Admission policy has missing or unknown fields")
    if (
        type(policy["schema_version"]) is not int
        or policy["schema_version"] != 2
        or not isinstance(policy["track"], str)
        or policy["track"] not in TRACKS
    ):
        raise ValueError("Admission policy requires schema 2 and a supported track")
    if (
        not isinstance(policy["candidate_revision"], str)
        or not policy["candidate_revision"].strip()
    ):
        raise ValueError("Admission policy requires a candidate revision")
    modes = policy["affected_modes"]
    if (
        not isinstance(modes, list)
        or not modes
        or any(mode not in ("single", "dialogue", "mixed") for mode in modes)
        or len(modes) != len(set(modes))
    ):
        raise ValueError("Admission policy requires distinct affected modes")
    budgets = policy["budgets"]
    if not isinstance(budgets, dict) or not {"tokens", "wall_duration_ms"} <= set(budgets):
        raise ValueError("Freeze at least tokens and wall_duration_ms budgets")
    for metric, limit in budgets.items():
        if metric not in BUDGET_METRICS:
            raise ValueError(f"Unsupported budget metric: {metric}")
        if not isinstance(limit, dict) or set(limit) != {"max_ratio", "absolute_allowance"}:
            raise ValueError("Each budget needs max_ratio and absolute_allowance")
        _number(limit["max_ratio"], f"{metric}.max_ratio")
        _number(limit["absolute_allowance"], f"{metric}.absolute_allowance")


def _count(sample: dict[str, Any], metric: str) -> int | None:
    machine = sample.get("machine", {})
    if not isinstance(machine, dict):
        return None
    if metric in machine and isinstance(machine[metric], list):
        return len(machine[metric])
    count = machine.get(f"{metric}_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None


def assess_admission(
    comparison: dict[str, Any],
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
    *,
    legacy_efficiency: dict[str, Any] | None = None,
    candidate_efficiency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Screen evidence, separating suspected defects from proven quality gains.

    A 'review' result only permits manual admission review. Source/gold timing
    differences and heuristic risks cannot establish candidate-caused damage.
    """
    validate_admission_policy(policy)
    blockers: list[str] = []
    observations: list[str] = []
    if not comparison["comparison_identity_match"]:
        blockers.append("source_gold_identity_mismatch")
    if not legacy.get("comparison_identity") or not candidate.get("comparison_identity"):
        observations.append("missing_source_gold_identity")
    if not legacy["samples"] or not candidate["samples"]:
        observations.append("missing_samples")

    # Check each sample: improvements elsewhere must not hide a local failure.
    for old, new in zip(legacy["samples"], candidate["samples"]):
        for metric in HARD_SIGNALS:
            old_count, new_count = _count(old, metric), _count(new, metric)
            if old_count is None or new_count is None:
                observations.append(f"missing_sample_evidence:{metric}")
            elif new_count > old_count:
                blockers.append(f"new_hard_signal:{metric}")
            elif new_count:
                observations.append(f"remaining_baseline_signal:{metric}")
    for metric in REVIEW_SIGNALS:
        if comparison["metrics"][metric]["candidate"]:
            observations.append(f"needs_semantic_review:{metric}")
    if comparison["metrics"]["structurally_exact_samples"]["delta"] < 0:
        observations.append("source_gold_segmentation_difference")

    budget_results: dict[str, Any] = {}
    if legacy_efficiency is None or candidate_efficiency is None:
        observations.append("missing_efficiency_evidence")
    else:
        efficiency = comparison["efficiency"]
        for name in ("cache_state", "workload_identity", "snapshot_count"):
            if not efficiency["gates"][name]:
                blockers.append(f"incomparable_efficiency:{name}")
        for report in (legacy_efficiency, candidate_efficiency):
            if not report.get("workload_identity") or report.get("cache_state") in (
                None,
                "unknown",
            ):
                observations.append("missing_efficiency_identity")
        shadow = efficiency.get("repair_shadow")
        if shadow is not None and not shadow["accepted"]:
            observations.append("incomplete_repair_shadow_evidence")
        old_metrics = legacy_efficiency.get("aggregate") or legacy_efficiency.get("metrics", {})
        new_metrics = candidate_efficiency.get("aggregate") or candidate_efficiency.get(
            "metrics", {}
        )
        for metric, limit in policy["budgets"].items():
            if metric not in old_metrics or metric not in new_metrics:
                observations.append(f"missing_budget_measurement:{metric}")
                budget_results[metric] = {"status": "unknown"}
                continue
            before = _number(old_metrics[metric], metric)
            after = _number(new_metrics[metric], metric)
            ceiling = before * limit["max_ratio"] + limit["absolute_allowance"]
            _number(ceiling, f"{metric}.ceiling")
            within = after <= ceiling
            budget_results[metric] = {
                "legacy": before,
                "candidate": after,
                "ceiling": ceiling,
                "status": "within" if within else "exceeded",
            }
            if not within:
                observations.append(f"budget_exceeded:{metric}")
        if policy["track"] == "efficiency" and not any(
            result.get("candidate", 0) < result.get("legacy", 0)
            for result in budget_results.values()
        ):
            observations.append("no_observed_efficiency_gain")

    return {
        "schema_version": 2,
        "track": policy["track"],
        "candidate_revision": policy["candidate_revision"],
        "affected_modes": policy["affected_modes"],
        "decision": "blocked" if blockers else "observe" if observations else "review",
        "blockers": sorted(set(blockers)),
        "observations": sorted(set(observations)),
        "budgets": budget_results,
        "production_adoption": False,
        "required_review": [
            "source_word_timing_and_speaker_integrity",
            "confirmed_severe_errors_and_risk_false_positives",
            "candidate_activation_and_causal_evidence",
            "independent_mode_scoped_quality_review",
            "repeat_variance_and_rate_limit_wait",
            "bounded_reasoning_scope_and_cost",
        ],
    }
