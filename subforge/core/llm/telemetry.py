"""Task-scoped, text-free telemetry for LLM pipeline evaluation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

TELEMETRY_SCHEMA_VERSION = 1


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


@dataclass(frozen=True)
class LLMStageTelemetrySnapshot:
    """Immutable request totals for one pipeline stage."""

    stage: str
    request_attempts: int
    successful_requests: int
    failed_requests: int
    api_duration_ms: int
    tokens: int
    prompt_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    reasoning_enabled_requests: int
    reasoning_disabled_requests: int
    reasoning_default_requests: int
    models: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "request_attempts": self.request_attempts,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "api_duration_ms": self.api_duration_ms,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_enabled_requests": self.reasoning_enabled_requests,
            "reasoning_disabled_requests": self.reasoning_disabled_requests,
            "reasoning_default_requests": self.reasoning_default_requests,
            "models": list(self.models),
        }


@dataclass(frozen=True)
class LLMTaskTelemetrySnapshot:
    """Immutable efficiency evidence for one task-scoped LLM client."""

    schema_version: int
    task_id: str
    workload_id: str
    pipeline_variant: str
    pipeline_revision: str
    cache_state: str
    wall_duration_ms: int
    request_attempts: int
    successful_requests: int
    failed_requests: int
    api_duration_ms: int
    retry_count: int
    retry_wait_ms: int
    rate_limit_retries: int
    transient_retries: int
    tokens: int
    prompt_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    reasoning_enabled_requests: int
    reasoning_disabled_requests: int
    reasoning_default_requests: int
    stages: tuple[LLMStageTelemetrySnapshot, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "workload_id": self.workload_id,
            "pipeline": {
                "variant": self.pipeline_variant,
                "revision": self.pipeline_revision,
            },
            "cache_state": self.cache_state,
            "metrics": {
                "wall_duration_ms": self.wall_duration_ms,
                "request_attempts": self.request_attempts,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "api_duration_ms": self.api_duration_ms,
                "retry_count": self.retry_count,
                "retry_wait_ms": self.retry_wait_ms,
                "rate_limit_retries": self.rate_limit_retries,
                "transient_retries": self.transient_retries,
                "tokens": self.tokens,
                "prompt_tokens": self.prompt_tokens,
                "cached_tokens": self.cached_tokens,
                "cache_creation_tokens": self.cache_creation_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "reasoning_enabled_requests": self.reasoning_enabled_requests,
                "reasoning_disabled_requests": self.reasoning_disabled_requests,
                "reasoning_default_requests": self.reasoning_default_requests,
                "provider_cache_hit_rate": (
                    round(self.cached_tokens / self.prompt_tokens, 4)
                    if self.prompt_tokens
                    else 0.0
                ),
            },
            "stages": [stage.to_dict() for stage in self.stages],
        }


class LLMTaskTelemetry:
    """Thread-safe accumulator attached to one explicit LLM client."""

    _COUNTER_KEYS = (
        "request_attempts",
        "successful_requests",
        "failed_requests",
        "api_duration_ms",
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

    def __init__(
        self,
        *,
        cache_state: str = "explicit-client-no-disk-cache",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started_at = clock()
        self._task_id = ""
        self._workload_id = ""
        self._pipeline_variant = "legacy"
        self._pipeline_revision = "legacy"
        self._cache_state = str(cache_state or "unknown")
        self._totals = {key: 0 for key in self._COUNTER_KEYS}
        self._retry_count = 0
        self._retry_wait_ms = 0
        self._rate_limit_retries = 0
        self._transient_retries = 0
        self._stages: dict[str, dict[str, Any]] = {}

    def configure(
        self,
        *,
        task_id: str,
        workload_id: str,
        pipeline_variant: str,
        pipeline_revision: str,
        cache_state: str | None = None,
    ) -> None:
        """Freeze task identity before the first provider request."""
        with self._lock:
            if self._totals["request_attempts"]:
                raise RuntimeError("LLM telemetry identity cannot change after requests begin")
            self._task_id = str(task_id or "")
            self._workload_id = str(workload_id or "")
            self._pipeline_variant = str(pipeline_variant or "legacy")
            self._pipeline_revision = str(pipeline_revision or "legacy")
            if cache_state is not None:
                self._cache_state = str(cache_state or "unknown")

    def record_attempt(
        self,
        entry: dict[str, Any],
        *,
        succeeded: bool,
        reasoning_mode: str,
    ) -> None:
        """Record one completed provider attempt without retaining prompt text."""
        mode = reasoning_mode if reasoning_mode in {"enabled", "disabled"} else "default"
        values = {
            "request_attempts": 1,
            "successful_requests": int(succeeded),
            "failed_requests": int(not succeeded),
            "api_duration_ms": _non_negative_int(entry.get("duration_ms")),
            "tokens": _non_negative_int(entry.get("tokens")),
            "prompt_tokens": _non_negative_int(entry.get("prompt_tokens")),
            "cached_tokens": _non_negative_int(entry.get("cached_tokens")),
            "cache_creation_tokens": _non_negative_int(
                entry.get("cache_creation_tokens")
            ),
            "completion_tokens": _non_negative_int(entry.get("completion_tokens")),
            "reasoning_tokens": _non_negative_int(entry.get("reasoning_tokens")),
            "reasoning_enabled_requests": int(mode == "enabled"),
            "reasoning_disabled_requests": int(mode == "disabled"),
            "reasoning_default_requests": int(mode == "default"),
        }
        stage_name = str(entry.get("stage") or "llm").strip() or "llm"
        model = str(entry.get("model") or "").strip()
        with self._lock:
            for key, value in values.items():
                self._totals[key] += value
            stage = self._stages.setdefault(
                stage_name,
                {
                    **{key: 0 for key in self._COUNTER_KEYS},
                    "models": set(),
                },
            )
            for key, value in values.items():
                stage[key] += value
            if model:
                stage["models"].add(model)

    def record_retry(self, *, kind: str, wait_seconds: float) -> None:
        """Record an already-selected retry delay without changing retry policy."""
        wait_ms = max(0, round(float(wait_seconds) * 1000))
        with self._lock:
            self._retry_count += 1
            self._retry_wait_ms += wait_ms
            if kind == "rate_limit":
                self._rate_limit_retries += 1
            else:
                self._transient_retries += 1

    def snapshot(self) -> LLMTaskTelemetrySnapshot:
        """Return a detached immutable snapshot safe for task results or disk."""
        with self._lock:
            elapsed_ms = max(0, round((self._clock() - self._started_at) * 1000))
            totals = dict(self._totals)
            stages = tuple(
                LLMStageTelemetrySnapshot(
                    stage=stage_name,
                    **{
                        key: int(stage[key])
                        for key in self._COUNTER_KEYS
                    },
                    models=tuple(sorted(stage["models"])),
                )
                for stage_name, stage in sorted(self._stages.items())
            )
            return LLMTaskTelemetrySnapshot(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                task_id=self._task_id,
                workload_id=self._workload_id,
                pipeline_variant=self._pipeline_variant,
                pipeline_revision=self._pipeline_revision,
                cache_state=self._cache_state,
                wall_duration_ms=elapsed_ms,
                retry_count=self._retry_count,
                retry_wait_ms=self._retry_wait_ms,
                rate_limit_retries=self._rate_limit_retries,
                transient_retries=self._transient_retries,
                stages=stages,
                **totals,
            )


def telemetry_for_client(client: Any) -> LLMTaskTelemetry | None:
    """Return task telemetry only when the client owns a compatible accumulator."""
    telemetry = getattr(client, "_subforge_telemetry", None)
    return telemetry if isinstance(telemetry, LLMTaskTelemetry) else None


def configure_client_telemetry(
    client: Any,
    *,
    task_id: str,
    workload_id: str = "",
    pipeline_variant: str,
    pipeline_revision: str,
    cache_state: str = "explicit-client-no-disk-cache",
) -> None:
    """Attach pipeline identity to a task client before requests begin."""
    telemetry = telemetry_for_client(client)
    if telemetry is not None:
        telemetry.configure(
            task_id=task_id,
            workload_id=workload_id,
            pipeline_variant=pipeline_variant,
            pipeline_revision=pipeline_revision,
            cache_state=cache_state,
        )


def snapshot_client_telemetry(client: Any) -> LLMTaskTelemetrySnapshot | None:
    """Snapshot telemetry without exposing the mutable accumulator."""
    telemetry = telemetry_for_client(client)
    return telemetry.snapshot() if telemetry is not None else None
