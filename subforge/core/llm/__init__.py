"""LLM unified client module."""

from .check_llm import check_llm_connection, get_available_models
from .check_whisper import check_whisper_connection
from .client import (
    call_llm,
    cancel_client_requests,
    close_client,
    create_client,
    get_llm_client,
    is_deepseek_v4_model,
    is_glm_53_model,
    is_kimi_k3_model,
    is_nemotron_3_ultra_model,
    prefers_native_reasoning,
)
from .response import (
    get_response_history_message,
    get_response_text,
    parse_json_object,
    strip_reasoning_blocks,
)
from .telemetry import (
    LLMTaskTelemetrySnapshot,
    configure_client_telemetry,
    snapshot_client_telemetry,
)

__all__ = [
    "call_llm",
    "cancel_client_requests",
    "close_client",
    "create_client",
    "get_llm_client",
    "is_deepseek_v4_model",
    "is_glm_53_model",
    "is_kimi_k3_model",
    "is_nemotron_3_ultra_model",
    "prefers_native_reasoning",
    "check_llm_connection",
    "get_available_models",
    "check_whisper_connection",
    "get_response_text",
    "get_response_history_message",
    "parse_json_object",
    "strip_reasoning_blocks",
    "LLMTaskTelemetrySnapshot",
    "configure_client_telemetry",
    "snapshot_client_telemetry",
]
