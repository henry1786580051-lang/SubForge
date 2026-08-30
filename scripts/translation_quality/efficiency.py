"""Text-free efficiency report aggregation for translation pipeline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

EFFICIENCY_SCHEMA_VERSION = 1
SHADOW_REPAIR_SCHEMA_VERSION = 1
CANONICAL_EVIDENCE_SCHEMA_VERSION = 1

_SUM_METRICS = (
    "wall_duration_ms",
    "request_attempts",
    "successful_requests",
    "failed_requests",
    "api_duration_ms",
    "retry_count",
    "retry_wait_ms",
    "rate_limit_retries",
    "transient_retries",
    "tokens",
    "prompt_tokens",
    "cached_tokens",
    "cache_creation_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "reasoning_enabled_requests",
    "reasoning_disabled_requests",
    "reasoning_default_requests",
)

_SHADOW_COUNT_METRICS = (
    "recorded_plan_observations",
    "unique_recorded_plans",
    "dropped_plan_observations",
    "recorded_comparison_observations",
    "unique_recorded_comparisons",
    "dropped_comparison_observations",
    "matched_comparisons",
    "mismatched_comparisons",
    "uncompared_recorded_plans",
)

_SHADOW_COUNTER_MAPS = (
    "dispositions",
    "planned_strategies",
    "planned_reasoning_modes",
    "session_modes",
    "diagnostic_rules",
    "comparison_routes",
)

_CANONICAL_EVIDENCE_COUNT_METRICS = (
    "terminology_line_count",
    "asr_labeled_line_count",
    "parseable_mapping_count",
    "mapping_with_source_match_count",
    "source_mapping_match_count",
    "supported_source_mapping_match_count",
    "rejected_source_mapping_match_count",
)


def _numeric(metrics: dict[str, Any], key: str) -> int | float:
    value = metrics.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Efficiency metric {key!r} is not numeric")
    if value < 0:
        raise ValueError(f"Efficiency metric {key!r} cannot be negative")
    return value


def _validate_shadow_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Repair shadow evidence must be a JSON object")
    if payload.get("schema_version") != SHADOW_REPAIR_SCHEMA_VERSION:
        raise ValueError("Repair shadow evidence uses an unsupported schema")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Repair shadow evidence has no counts object")
    for key in _SHADOW_COUNT_METRICS:
        value = _numeric(counts, key)
        if not isinstance(value, int):
            raise ValueError(f"Repair shadow count {key!r} must be an integer")
    for key in _SHADOW_COUNTER_MAPS:
        values = payload.get(key)
        if not isinstance(values, dict):
            raise ValueError(f"Repair shadow evidence has no {key} object")
        for label, value in values.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"Repair shadow {key} labels must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Repair shadow {key} counts must be non-negative integers")
    return payload


def _aggregate_shadow_payloads(payloads: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    counts = {key: 0 for key in _SHADOW_COUNT_METRICS}
    counter_maps: dict[str, dict[str, int]] = {
        key: {} for key in _SHADOW_COUNTER_MAPS
    }
    for raw_payload in payloads:
        payload = _validate_shadow_payload(raw_payload)
        payload_counts = payload["counts"]
        for key in _SHADOW_COUNT_METRICS:
            counts[key] += int(payload_counts[key])
        for key in _SHADOW_COUNTER_MAPS:
            aggregate_values = counter_maps[key]
            for label, value in payload[key].items():
                aggregate_values[label] = aggregate_values.get(label, 0) + int(value)
    plan_count = counts["recorded_plan_observations"]
    comparison_count = counts["recorded_comparison_observations"]
    return {
        "schema_version": SHADOW_REPAIR_SCHEMA_VERSION,
        "snapshot_count": len(payloads),
        "counts": counts,
        "rates": {
            "comparison_coverage": (
                round(comparison_count / plan_count, 4) if plan_count else 0.0
            ),
            "match_rate": (
                round(counts["matched_comparisons"] / comparison_count, 4)
                if comparison_count
                else 0.0
            ),
        },
        **{
            key: dict(sorted(values.items()))
            for key, values in counter_maps.items()
        },
    }


def _validate_canonical_evidence_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Canonical evidence must be a JSON object")
    if payload.get("schema_version") != CANONICAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Canonical evidence uses an unsupported schema")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Canonical evidence has no counts object")
    for key in _CANONICAL_EVIDENCE_COUNT_METRICS:
        value = _numeric(counts, key)
        if not isinstance(value, int):
            raise ValueError(f"Canonical evidence count {key!r} must be an integer")
    if counts["parseable_mapping_count"] > counts["terminology_line_count"]:
        raise ValueError("Canonical evidence has more mappings than terminology lines")
    if counts["mapping_with_source_match_count"] > counts["parseable_mapping_count"]:
        raise ValueError("Canonical evidence has more matched mappings than mappings")
    classified = (
        counts["supported_source_mapping_match_count"]
        + counts["rejected_source_mapping_match_count"]
    )
    if classified != counts["source_mapping_match_count"]:
        raise ValueError("Canonical evidence source matches are not fully classified")
    return payload


def _aggregate_canonical_evidence_payloads(
    payloads: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    counts = {key: 0 for key in _CANONICAL_EVIDENCE_COUNT_METRICS}
    for raw_payload in payloads:
        payload = _validate_canonical_evidence_payload(raw_payload)
        for key in _CANONICAL_EVIDENCE_COUNT_METRICS:
            counts[key] += int(payload["counts"][key])
    return {
        "schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
        "snapshot_count": len(payloads),
        "counts": counts,
    }


def load_efficiency_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Efficiency file must contain a JSON object: {path}")
    if payload.get("schema_version") != EFFICIENCY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported efficiency schema in {path}")
    metrics_key = "aggregate" if "aggregate" in payload else "metrics"
    metrics = payload.get(metrics_key)
    if not isinstance(metrics, dict):
        raise ValueError(f"Efficiency file has no {metrics_key} object: {path}")
    for key in _SUM_METRICS:
        _numeric(metrics, key)
    repair_shadow = payload.get("repair_shadow")
    if repair_shadow is not None:
        _validate_shadow_payload(repair_shadow)
    canonical_evidence = payload.get("canonical_evidence")
    if canonical_evidence is not None:
        _validate_canonical_evidence_payload(canonical_evidence)
    return payload


def aggregate_efficiency_payloads(
    snapshots: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    items = tuple(snapshots)
    if not items:
        raise ValueError("At least one efficiency snapshot is required")
    totals: dict[str, int | float] = {key: 0 for key in _SUM_METRICS}
    cache_states: set[str] = set()
    variants: set[str] = set()
    revisions: set[str] = set()
    task_ids: set[str] = set()
    workload_ids: list[str] = []
    repair_shadows: list[dict[str, Any]] = []
    canonical_evidence_snapshots: list[dict[str, Any]] = []
    for payload in items:
        if payload.get("schema_version") != EFFICIENCY_SCHEMA_VERSION:
            raise ValueError("Efficiency snapshots use an unsupported schema")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Efficiency snapshot has no metrics object")
        for key in _SUM_METRICS:
            totals[key] += _numeric(metrics, key)
        cache_states.add(str(payload.get("cache_state") or "unknown"))
        pipeline = payload.get("pipeline")
        if not isinstance(pipeline, dict):
            raise ValueError("Efficiency snapshot has no pipeline identity")
        variant = str(pipeline.get("variant") or "")
        revision = str(pipeline.get("revision") or "")
        if not variant or not revision:
            raise ValueError("Efficiency snapshot has an incomplete pipeline identity")
        variants.add(variant)
        revisions.add(revision)
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise ValueError("Efficiency snapshot has no task id")
        if task_id in task_ids:
            raise ValueError(f"Duplicate efficiency task id: {task_id}")
        task_ids.add(task_id)
        workload_id = str(payload.get("workload_id") or "")
        if not workload_id:
            raise ValueError(f"Efficiency snapshot {task_id!r} has no workload id")
        workload_ids.append(workload_id)
        repair_shadow = payload.get("repair_shadow")
        if repair_shadow is not None:
            repair_shadows.append(_validate_shadow_payload(repair_shadow))
        canonical_evidence = payload.get("canonical_evidence")
        if canonical_evidence is not None:
            canonical_evidence_snapshots.append(
                _validate_canonical_evidence_payload(canonical_evidence)
            )
    if len(variants) != 1 or len(revisions) != 1:
        raise ValueError("Efficiency snapshots must use one pipeline identity")
    if len(cache_states) != 1:
        raise ValueError("Efficiency snapshots must use one cache state")
    prompt_tokens = totals["prompt_tokens"]
    totals["provider_cache_hit_rate"] = (
        round(totals["cached_tokens"] / prompt_tokens, 4) if prompt_tokens else 0.0
    )
    if repair_shadows and len(repair_shadows) != len(items):
        raise ValueError(
            "Efficiency snapshots cannot mix present and missing repair shadow evidence"
        )
    if canonical_evidence_snapshots and len(canonical_evidence_snapshots) != len(items):
        raise ValueError(
            "Efficiency snapshots cannot mix present and missing canonical evidence"
        )
    result = {
        "schema_version": EFFICIENCY_SCHEMA_VERSION,
        "pipeline": {
            "variant": next(iter(variants)),
            "revision": next(iter(revisions)),
        },
        "cache_state": next(iter(cache_states)),
        "snapshot_count": len(items),
        "workload_identity": hashlib.sha256(
            json.dumps(sorted(workload_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "aggregate": totals,
    }
    if repair_shadows:
        result["repair_shadow"] = _aggregate_shadow_payloads(tuple(repair_shadows))
    if canonical_evidence_snapshots:
        result["canonical_evidence"] = _aggregate_canonical_evidence_payloads(
            tuple(canonical_evidence_snapshots)
        )
    return result


def write_efficiency_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
