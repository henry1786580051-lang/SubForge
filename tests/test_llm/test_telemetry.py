from types import SimpleNamespace

import pytest

from subforge.core.llm.telemetry import (
    LLMTaskTelemetry,
    configure_client_telemetry,
    snapshot_client_telemetry,
)


def test_task_telemetry_records_text_free_attempt_retry_and_stage_totals():
    clock_values = iter((10.0, 12.5))
    telemetry = LLMTaskTelemetry(clock=lambda: next(clock_values))
    telemetry.configure(
        task_id="task-1",
        workload_id="source-sha256",
        pipeline_variant="candidate",
        pipeline_revision="phase8-r1",
    )
    telemetry.record_attempt(
        {
            "stage": "translate",
            "model": "deepseek-v4-flash",
            "duration_ms": 900,
            "tokens": 150,
            "prompt_tokens": 100,
            "cached_tokens": 40,
            "completion_tokens": 50,
            "reasoning_tokens": 20,
        },
        succeeded=True,
        reasoning_mode="enabled",
    )
    telemetry.record_attempt(
        {
            "stage": "translate",
            "model": "deepseek-v4-flash",
            "duration_ms": 100,
        },
        succeeded=False,
        reasoning_mode="disabled",
    )
    telemetry.record_retry(kind="rate_limit", wait_seconds=1.25)

    payload = telemetry.snapshot().to_dict()

    assert payload["task_id"] == "task-1"
    assert payload["workload_id"] == "source-sha256"
    assert payload["pipeline"] == {
        "variant": "candidate",
        "revision": "phase8-r1",
    }
    assert payload["metrics"] == {
        "wall_duration_ms": 2500,
        "request_attempts": 2,
        "successful_requests": 1,
        "failed_requests": 1,
        "api_duration_ms": 1000,
        "retry_count": 1,
        "retry_wait_ms": 1250,
        "rate_limit_retries": 1,
        "transient_retries": 0,
        "tokens": 150,
        "prompt_tokens": 100,
        "cached_tokens": 40,
        "cache_creation_tokens": 0,
        "completion_tokens": 50,
        "reasoning_tokens": 20,
        "reasoning_enabled_requests": 1,
        "reasoning_disabled_requests": 1,
        "reasoning_default_requests": 0,
        "provider_cache_hit_rate": 0.4,
    }
    assert payload["stages"] == [
        {
            "stage": "translate",
            "request_attempts": 2,
            "successful_requests": 1,
            "failed_requests": 1,
            "api_duration_ms": 1000,
            "tokens": 150,
            "prompt_tokens": 100,
            "cached_tokens": 40,
            "cache_creation_tokens": 0,
            "completion_tokens": 50,
            "reasoning_tokens": 20,
            "reasoning_enabled_requests": 1,
            "reasoning_disabled_requests": 1,
            "reasoning_default_requests": 0,
            "models": ["deepseek-v4-flash"],
        }
    ]


def test_client_telemetry_configuration_is_ignored_for_uninstrumented_test_client():
    client = SimpleNamespace()

    configure_client_telemetry(
        client,
        task_id="task-1",
        workload_id="source-sha256",
        pipeline_variant="candidate",
        pipeline_revision="phase8-r1",
    )

    assert snapshot_client_telemetry(client) is None


def test_task_identity_cannot_change_after_first_request():
    telemetry = LLMTaskTelemetry()
    telemetry.record_attempt({}, succeeded=True, reasoning_mode="default")

    with pytest.raises(RuntimeError, match="cannot change"):
        telemetry.configure(
            task_id="late",
            workload_id="source-sha256",
            pipeline_variant="candidate",
            pipeline_revision="phase8-r1",
        )
