"""Side-by-side comparison for isolated translation-quality pipeline reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

LOWER_IS_BETTER_METRICS = (
    "hard_failure_count",
    "empty_targets",
    "placeholder_targets",
    "reasoning_leaks",
    "source_copy_targets",
    "untranslated_targets",
    "adjacent_duplicate_risks",
)
HIGHER_IS_BETTER_METRICS = ("structurally_exact_samples",)
INFORMATIONAL_METRICS = (
    "sample_count",
    "source_cue_count",
    "machine_cue_count",
    "gold_cue_count",
    "human_changed_cues",
    "requires_alignment_samples",
)
EFFICIENCY_BUDGET_RATIO = 1.05
EFFICIENCY_BUDGET_METRICS = (
    "successful_requests",
    "request_attempts",
    "tokens",
    "wall_duration_ms",
)
EFFICIENCY_REASONING_METRICS = ("reasoning_enabled_requests",)
EFFICIENCY_INFORMATIONAL_METRICS = (
    "failed_requests",
    "api_duration_ms",
    "retry_count",
    "retry_wait_ms",
    "rate_limit_retries",
    "transient_retries",
    "prompt_tokens",
    "cached_tokens",
    "cache_creation_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "reasoning_disabled_requests",
    "reasoning_default_requests",
    "provider_cache_hit_rate",
)


def load_evaluation_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation report must contain a JSON object: {path}")
    if not isinstance(payload.get("aggregate"), dict):
        raise ValueError(f"Evaluation report has no aggregate object: {path}")
    if not isinstance(payload.get("samples"), list):
        raise ValueError(f"Evaluation report has no samples list: {path}")
    return payload


def compare_evaluation_reports(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    *,
    legacy_efficiency: dict[str, Any] | None = None,
    candidate_efficiency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legacy_ids = _sample_ids(legacy)
    candidate_ids = _sample_ids(candidate)
    if legacy.get("corpus_id") != candidate.get("corpus_id"):
        raise ValueError("Legacy and candidate reports use different corpus ids")
    if legacy_ids != candidate_ids:
        raise ValueError("Legacy and candidate reports use different sample sets or order")
    legacy_identity = legacy.get("comparison_identity")
    candidate_identity = candidate.get("comparison_identity")
    identity_match = (
        legacy_identity == candidate_identity
        if legacy_identity is not None or candidate_identity is not None
        else True
    )

    metric_names = (
        *LOWER_IS_BETTER_METRICS,
        *HIGHER_IS_BETTER_METRICS,
        *INFORMATIONAL_METRICS,
    )
    metrics = {
        metric: _metric_comparison(legacy, candidate, metric)
        for metric in metric_names
    }
    gates = {
        metric: metrics[metric]["candidate"] <= metrics[metric]["legacy"]
        for metric in LOWER_IS_BETTER_METRICS
    }
    gates.update(
        {
            metric: metrics[metric]["candidate"] >= metrics[metric]["legacy"]
            for metric in HIGHER_IS_BETTER_METRICS
        }
    )
    gates["sample_identity"] = True
    gates["corpus_identity"] = identity_match

    efficiency = None
    if (legacy_efficiency is None) != (candidate_efficiency is None):
        raise ValueError("Legacy and candidate efficiency reports must be supplied together")
    if legacy_efficiency is not None and candidate_efficiency is not None:
        efficiency = _compare_efficiency(legacy_efficiency, candidate_efficiency)
        gates.update(
            {
                f"efficiency_{name}": passed
                for name, passed in efficiency["gates"].items()
            }
        )

    legacy_samples = {sample["sample_id"]: sample for sample in legacy["samples"]}
    candidate_samples = {sample["sample_id"]: sample for sample in candidate["samples"]}
    samples = [
        _sample_comparison(legacy_samples[sample_id], candidate_samples[sample_id])
        for sample_id in legacy_ids
    ]
    return {
        "corpus_id": legacy.get("corpus_id"),
        "legacy_manifest_hash": legacy.get("manifest_hash"),
        "candidate_manifest_hash": candidate.get("manifest_hash"),
        "manifest_hash_match": legacy.get("manifest_hash") == candidate.get("manifest_hash"),
        "comparison_identity_match": identity_match,
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "efficiency": efficiency,
        "samples": samples,
    }


def _compare_efficiency(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    legacy_metrics = legacy.get("aggregate") or legacy.get("metrics")
    candidate_metrics = candidate.get("aggregate") or candidate.get("metrics")
    if not isinstance(legacy_metrics, dict) or not isinstance(candidate_metrics, dict):
        raise ValueError("Efficiency reports must contain aggregate or metrics objects")
    metric_names = (
        *EFFICIENCY_BUDGET_METRICS,
        *EFFICIENCY_REASONING_METRICS,
        *EFFICIENCY_INFORMATIONAL_METRICS,
    )
    metrics = {
        metric: _efficiency_metric_comparison(legacy_metrics, candidate_metrics, metric)
        for metric in metric_names
    }
    gates = {
        metric: metrics[metric]["candidate"]
        <= metrics[metric]["legacy"] * EFFICIENCY_BUDGET_RATIO
        for metric in EFFICIENCY_BUDGET_METRICS
    }
    gates.update(
        {
            metric: metrics[metric]["candidate"] <= metrics[metric]["legacy"]
            for metric in EFFICIENCY_REASONING_METRICS
        }
    )
    gates["cache_state"] = legacy.get("cache_state") == candidate.get("cache_state")
    gates["workload_identity"] = legacy.get("workload_identity") == candidate.get(
        "workload_identity"
    )
    legacy_count = int(legacy.get("snapshot_count", 1))
    candidate_count = int(candidate.get("snapshot_count", 1))
    gates["snapshot_count"] = legacy_count == candidate_count
    repair_shadow = _compare_repair_shadow(
        legacy.get("repair_shadow"),
        candidate.get("repair_shadow"),
    )
    if repair_shadow is not None:
        gates.update(
            {
                f"repair_shadow_{name}": passed
                for name, passed in repair_shadow["gates"].items()
            }
        )
    return {
        "budget_ratio": EFFICIENCY_BUDGET_RATIO,
        "legacy_cache_state": legacy.get("cache_state"),
        "candidate_cache_state": candidate.get("cache_state"),
        "legacy_snapshot_count": legacy_count,
        "candidate_snapshot_count": candidate_count,
        "workload_identity_match": gates["workload_identity"],
        "metrics": metrics,
        "repair_shadow": repair_shadow,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _compare_repair_shadow(
    legacy: Any,
    candidate: Any,
) -> dict[str, Any] | None:
    if legacy is None and candidate is None:
        return None
    if not isinstance(legacy, dict) or not isinstance(candidate, dict):
        return {
            "available_for_both": False,
            "legacy": legacy if isinstance(legacy, dict) else None,
            "candidate": candidate if isinstance(candidate, dict) else None,
            "gates": {"available_for_both": False},
            "accepted": False,
        }
    legacy_counts = legacy.get("counts")
    candidate_counts = candidate.get("counts")
    if not isinstance(legacy_counts, dict) or not isinstance(candidate_counts, dict):
        raise ValueError("Repair shadow reports must contain counts objects")

    def count(values: dict[str, Any], key: str) -> int:
        value = values.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Repair shadow count {key!r} must be a non-negative integer")
        return value

    keys = (
        "recorded_plan_observations",
        "dropped_plan_observations",
        "recorded_comparison_observations",
        "dropped_comparison_observations",
        "matched_comparisons",
        "mismatched_comparisons",
        "uncompared_recorded_plans",
    )
    counts = {
        key: {
            "legacy": count(legacy_counts, key),
            "candidate": count(candidate_counts, key),
        }
        for key in keys
    }
    legacy_plan_count = counts["recorded_plan_observations"]["legacy"]
    legacy_comparison_count = counts["recorded_comparison_observations"]["legacy"]
    candidate_plan_count = counts["recorded_plan_observations"]["candidate"]
    candidate_comparison_count = counts["recorded_comparison_observations"]["candidate"]
    gates = {
        "available_for_both": True,
        "snapshot_count": int(legacy.get("snapshot_count", 1))
        == int(candidate.get("snapshot_count", 1)),
        "legacy_observed_plans": legacy_plan_count > 0,
        "legacy_full_comparison_coverage": (
            legacy_comparison_count == legacy_plan_count
            and counts["uncompared_recorded_plans"]["legacy"] == 0
        ),
        "legacy_no_dropped_plans": counts["dropped_plan_observations"]["legacy"] == 0,
        "legacy_no_dropped_comparisons": (
            counts["dropped_comparison_observations"]["legacy"] == 0
        ),
        "candidate_observed_plans": candidate_plan_count > 0,
        "candidate_full_comparison_coverage": (
            candidate_comparison_count == candidate_plan_count
            and counts["uncompared_recorded_plans"]["candidate"] == 0
        ),
        "candidate_no_dropped_plans": (
            counts["dropped_plan_observations"]["candidate"] == 0
        ),
        "candidate_no_dropped_comparisons": (
            counts["dropped_comparison_observations"]["candidate"] == 0
        ),
    }
    return {
        "available_for_both": True,
        "counts": counts,
        "legacy_rates": legacy.get("rates", {}),
        "candidate_rates": candidate.get("rates", {}),
        "candidate_comparison_routes": candidate.get("comparison_routes", {}),
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _efficiency_metric_comparison(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
) -> dict[str, int | float]:
    legacy_value = _efficiency_numeric(legacy, metric)
    candidate_value = _efficiency_numeric(candidate, metric)
    ratio = candidate_value / legacy_value if legacy_value else None
    return {
        "legacy": legacy_value,
        "candidate": candidate_value,
        "delta": candidate_value - legacy_value,
        "ratio": round(ratio, 4) if ratio is not None else 0.0 if candidate_value == 0 else -1.0,
    }


def _efficiency_numeric(metrics: dict[str, Any], metric: str) -> int | float:
    value = metrics.get(metric, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Efficiency report metric {metric!r} is not numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Efficiency report metric {metric!r} must be finite and non-negative")
    return value


def write_comparison_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        _comparison_markdown(payload),
        encoding="utf-8",
    )


def _sample_ids(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for sample in report["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
            raise ValueError("Evaluation report contains an invalid sample entry")
        ids.append(sample["sample_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation report contains duplicate sample ids")
    return ids


def _numeric_metric(report: dict[str, Any], metric: str) -> int | float:
    value = report["aggregate"].get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evaluation report metric {metric!r} is not numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Evaluation report metric {metric!r} must be finite and non-negative")
    return value


def _metric_comparison(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
) -> dict[str, int | float]:
    legacy_value = _numeric_metric(legacy, metric)
    candidate_value = _numeric_metric(candidate, metric)
    return {
        "legacy": legacy_value,
        "candidate": candidate_value,
        "delta": candidate_value - legacy_value,
    }


def _sample_comparison(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    legacy_hard = int(legacy.get("hard_failure_count", 0))
    candidate_hard = int(candidate.get("hard_failure_count", 0))
    return {
        "sample_id": legacy["sample_id"],
        "title": legacy.get("title", ""),
        "split": legacy.get("split", ""),
        "legacy_hard_failures": legacy_hard,
        "candidate_hard_failures": candidate_hard,
        "hard_failure_delta": candidate_hard - legacy_hard,
        "legacy_machine_cues": int(legacy.get("machine", {}).get("cue_count", 0)),
        "candidate_machine_cues": int(candidate.get("machine", {}).get("cue_count", 0)),
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Translation Pipeline Comparison",
        "",
        f"- Corpus: `{payload['corpus_id']}`",
        f"- Accepted by static gates: {'yes' if payload['accepted'] else 'no'}",
        f"- Manifest hashes match: {'yes' if payload['manifest_hash_match'] else 'no'}",
        "- Source/gold comparison identity matches: "
        f"{'yes' if payload['comparison_identity_match'] else 'no'}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Legacy | Candidate | Delta | Gate |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    if admission := payload.get("admission"):
        lines[2:2] = [
            "## Track-Specific Screening (v2)",
            "",
            f"- Track: `{admission['track']}`",
            f"- Decision: `{admission['decision']}`",
            f"- Frozen policy SHA-256: `{admission['policy_sha256']}`",
            "- Production adoption: NOT ASSESSED. Manual evidence review is mandatory.",
            "- Blockers: " + (", ".join(admission["blockers"]) or "none"),
            "- Observations: " + (", ".join(admission["observations"]) or "none"),
            "- Required reviews: " + ", ".join(admission["required_review"]),
            "",
            "| Frozen budget | Legacy | Candidate | Ceiling | Status |",
            "| --- | ---: | ---: | ---: | --- |",
            *[
                f"| {name} | {values.get('legacy', 'unknown')} | "
                f"{values.get('candidate', 'unknown')} | {values.get('ceiling', 'unknown')} | "
                f"{values['status']} |"
                for name, values in admission["budgets"].items()
            ],
            "",
            "## Historical v1 Gates (informational for v2)",
            "",
        ]
    gates = payload["gates"]
    for metric, values in payload["metrics"].items():
        gate = gates.get(metric)
        gate_text = "pass" if gate is True else "fail" if gate is False else "info"
        lines.append(
            f"| {metric} | {values['legacy']} | {values['candidate']} | "
            f"{values['delta']:+} | {gate_text} |"
        )
    efficiency = payload.get("efficiency")
    if efficiency is not None:
        lines.extend(
            (
                "",
                "## Efficiency",
                "",
                f"- Accepted: {'yes' if efficiency['accepted'] else 'no'}",
                f"- Budget: candidate <= legacy x {efficiency['budget_ratio']}",
                f"- Cache state: `{efficiency['legacy_cache_state']}` / "
                f"`{efficiency['candidate_cache_state']}`",
                "- Workload identity matches: "
                f"{'yes' if efficiency['workload_identity_match'] else 'no'}",
                "",
                "| Metric | Legacy | Candidate | Delta | Gate |",
                "| --- | ---: | ---: | ---: | :---: |",
            )
        )
        efficiency_gates = efficiency["gates"]
        for metric, values in efficiency["metrics"].items():
            gate = efficiency_gates.get(metric)
            gate_text = "pass" if gate is True else "fail" if gate is False else "info"
            lines.append(
                f"| {metric} | {values['legacy']} | {values['candidate']} | "
                f"{values['delta']:+} | {gate_text} |"
            )
        repair_shadow = efficiency.get("repair_shadow")
        if repair_shadow is not None:
            lines.extend(
                (
                    "",
                    "## Repair Shadow Evidence",
                    "",
                    f"- Accepted: {'yes' if repair_shadow['accepted'] else 'no'}",
                    f"- Available for both pipelines: "
                    f"{'yes' if repair_shadow['available_for_both'] else 'no'}",
                    "- Subtitle text and cue keys are intentionally excluded.",
                )
            )
            if repair_shadow.get("counts"):
                lines.extend(
                    (
                        "",
                        "| Metric | Legacy | Candidate |",
                        "| --- | ---: | ---: |",
                    )
                )
                for metric, values in repair_shadow["counts"].items():
                    lines.append(
                        f"| {metric} | {values['legacy']} | {values['candidate']} |"
                    )
    lines.extend(
        (
            "",
            "## Samples",
            "",
            "| Sample | Split | Legacy hard failures | Candidate hard failures | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        )
    )
    for sample in payload["samples"]:
        title = str(sample["title"]).replace("|", "\\|")
        lines.append(
            f"| {title} | {sample['split']} | {sample['legacy_hard_failures']} | "
            f"{sample['candidate_hard_failures']} | {sample['hard_failure_delta']:+} |"
        )
    lines.extend(("", "Detailed cue text is intentionally omitted from this report.", ""))
    return "\n".join(lines)
