"""Unified LLM client for the application."""

import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List, Literal, Optional
from urllib.parse import urlparse, urlunparse

import anthropic
import openai
from openai import OpenAI

from subforge.core.utils.cache import get_llm_cache, memoize
from subforge.core.utils.logger import setup_logger

from .anthropic_client import MiniMaxAnthropicClient
from .request_logger import create_logging_http_client, log_llm_error, log_llm_response
from .telemetry import LLMTaskTelemetry, telemetry_for_client

_global_client: Optional[Any] = None
_global_client_identity: Optional[tuple[str, str]] = None
_client_lock = threading.Lock()

logger = setup_logger("llm_client")

# Timeout for LLM API calls (seconds)
LLM_TIMEOUT = 120.0
KIMI_K3_REQUEST_TIMEOUT = 300.0
NEMOTRON_3_ULTRA_REQUEST_TIMEOUT = 300.0
LMSTUDIO_LOCAL_REQUEST_TIMEOUT = 300.0
NEMOTRON_3_ULTRA_TRANSIENT_MAX_ATTEMPTS = 12
NEMOTRON_3_ULTRA_RETRY_SPACING = 1.5
KIMI_K3_RATE_LIMIT_RETRY_SPACING = 2.0
KIMI_K3_RATE_LIMIT_MAX_WAIT = 600.0
PERSISTENT_RATE_LIMIT_MAX_WAIT = 60.0
PERSISTENT_TRANSIENT_MAX_ATTEMPTS = 3
ReasoningMode = Literal["default", "enabled", "disabled"]

_kimi_k3_retry_lock = threading.Lock()
_kimi_k3_next_retry_at = 0.0
_nemotron_ultra_retry_lock = threading.Lock()
_nemotron_ultra_next_retry_at = 0.0


class LLMRequestCancelled(RuntimeError):
    """Raised when a task cancels an in-flight or waiting LLM request."""


def is_anthropic_base_url(base_url: str) -> bool:
    """Return whether a URL targets an Anthropic-compatible endpoint."""
    path = urlparse(str(base_url or "").strip()).path.rstrip("/").lower()
    return path.endswith("/anthropic")


def normalize_base_url(base_url: str) -> str:
    """Normalize API base URL by ensuring /v1 suffix when needed."""
    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if not path:
        path = "/v1"

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return normalized


def create_client(base_url: str, api_key: str, *, timeout: float = LLM_TIMEOUT) -> Any:
    """Create a protocol-aware client with explicit credentials."""
    base_url = normalize_base_url(base_url)
    if not base_url or not api_key:
        raise ValueError("base_url and api_key are required")
    log_context: dict[str, str] = {}
    http_client = create_logging_http_client(log_context=log_context)
    if is_anthropic_base_url(base_url):
        client = MiniMaxAnthropicClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
    else:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
    setattr(client, "_subforge_log_context", log_context)
    setattr(client, "_subforge_base_url", base_url)
    setattr(client, "_subforge_cancel_event", threading.Event())
    setattr(client, "_subforge_telemetry", LLMTaskTelemetry())
    return client


def cancel_client_requests(client: Any) -> None:
    """Interrupt retry waits and close active transports for one task client."""
    event = getattr(client, "_subforge_cancel_event", None)
    if isinstance(event, threading.Event):
        event.set()
    close_client(client)


def close_client(client: Any) -> None:
    """Close a task-scoped LLM client without coupling callers to its SDK."""
    if bool(getattr(client, "_subforge_closed", False)):
        return
    try:
        setattr(client, "_subforge_closed", True)
    except Exception:
        pass
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("Failed to close LLM client", exc_info=True)


def _raise_if_request_cancelled(client: Any) -> None:
    event = getattr(client, "_subforge_cancel_event", None)
    if isinstance(event, threading.Event) and event.is_set():
        raise LLMRequestCancelled("LLM request cancelled")


def _wait_for_retry(client: Any, seconds: float) -> None:
    """Wait for provider recovery while remaining responsive to cancellation."""
    event = getattr(client, "_subforge_cancel_event", None)
    if isinstance(event, threading.Event):
        if event.wait(timeout=max(0.0, seconds)):
            raise LLMRequestCancelled("LLM request cancelled")
        return
    time.sleep(seconds)


def _record_retry_telemetry(client: Any, *, kind: str, wait_seconds: float) -> None:
    """Observe a selected retry delay without changing provider behavior."""
    target = client if client is not None else _global_client
    if target is None:
        return
    telemetry = telemetry_for_client(target)
    if telemetry is not None:
        telemetry.record_retry(kind=kind, wait_seconds=wait_seconds)


def set_client_log_context(client: Any, **context: str) -> None:
    """Attach task metadata without changing the public client constructor."""
    target = getattr(client, "_subforge_log_context", None)
    if isinstance(target, dict):
        target.update({key: value for key, value in context.items() if value})


def get_llm_client() -> Any:
    """Get global LLM client instance (thread-safe singleton)."""
    global _global_client, _global_client_identity

    base_url = normalize_base_url(os.getenv("OPENAI_BASE_URL", "").strip())
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not base_url or not api_key:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY environment variables must be set")
    identity = (base_url, api_key)

    if _global_client is None or _global_client_identity != identity:
        with _client_lock:
            if _global_client is None or _global_client_identity != identity:
                _global_client = create_client(base_url, api_key)
                _global_client_identity = identity

    return _global_client


def _is_retryable_standard_error(error: BaseException) -> bool:
    """Retry temporary provider failures without masking bad requests or auth errors."""
    return isinstance(
        error,
        (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )


def _is_minimax_m3_model(model: str) -> bool:
    """Return whether the selected model needs MiniMax M3's persistent 429 policy."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(model or "").lower())
    return normalized == "minimaxm3"


def _is_nvidia_client(client: Any = None) -> bool:
    """Return whether this request uses NVIDIA's OpenAI-compatible endpoint."""
    if client is None:
        base_url = os.getenv("OPENAI_BASE_URL", "")
    else:
        base_url = getattr(client, "_subforge_base_url", "")
        if not base_url:
            base_url = getattr(client, "base_url", "")
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    return hostname == "integrate.api.nvidia.com"


def _is_deepseek_client(client: Any = None) -> bool:
    """Return whether this request uses DeepSeek's official API endpoint."""
    if client is None:
        base_url = os.getenv("OPENAI_BASE_URL", "")
    else:
        base_url = getattr(client, "_subforge_base_url", "")
        if not base_url:
            base_url = getattr(client, "base_url", "")
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    return hostname == "api.deepseek.com"


def _normalized_model_leaf(model: str) -> str:
    """Normalize the provider-independent model name from a namespaced ID."""
    leaf = str(model or "").strip().strip("/").lower().rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", leaf)


def _is_deepseek_model(model: str) -> bool:
    return _normalized_model_leaf(model).startswith("deepseek")


def is_deepseek_v4_model(model: str) -> bool:
    """Return whether a bare or namespaced model ID selects DeepSeek V4."""
    return _normalized_model_leaf(model).startswith("deepseekv4")


def is_kimi_k3_model(model: str) -> bool:
    """Return whether a bare or namespaced model ID selects Kimi K3."""
    return _normalized_model_leaf(model) == "kimik3"


def is_nemotron_3_ultra_model(model: str) -> bool:
    """Return whether a model ID selects Nemotron 3 Ultra 550B A55B."""
    return _normalized_model_leaf(model) in {
        "nemotron3ultra550ba55b",
        "nvidianemotron3ultra550ba55b",
    }


def is_qwen_38_model(model: str) -> bool:
    """Return whether a model ID selects the Qwen 3.8 family."""
    return _normalized_model_leaf(model).startswith("qwen38")


def is_lmstudio_client(client: Any = None) -> bool:
    """Return whether a request targets a loopback LM Studio server."""
    if client is None:
        base_url = os.getenv("OPENAI_BASE_URL", "")
    else:
        base_url = getattr(client, "_subforge_base_url", "")
        if not base_url:
            base_url = getattr(client, "base_url", "")
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    return hostname in {"127.0.0.1", "::1", "localhost"}


def is_lmstudio_qwen_38_request(model: str, client: Any = None) -> bool:
    """Return whether Qwen 3.8 is served by the local LM Studio endpoint."""
    return is_qwen_38_model(model) and is_lmstudio_client(client)


def constrain_local_llm_workload(
    model: str,
    client: Any,
    *,
    concurrency: int,
    batch_size: int,
) -> tuple[int, int]:
    """Keep memory-bound local models within a stable request envelope."""
    if is_lmstudio_qwen_38_request(model, client):
        return min(concurrency, 1), min(batch_size, 10)
    return concurrency, batch_size


def _is_zhipu_client(client: Any = None) -> bool:
    """Return whether this request uses Zhipu's official OpenAI endpoint."""
    if client is None:
        base_url = os.getenv("OPENAI_BASE_URL", "")
    else:
        base_url = getattr(client, "_subforge_base_url", "")
        if not base_url:
            base_url = getattr(client, "base_url", "")
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    return hostname == "open.bigmodel.cn"


def is_glm_53_model(model: str) -> bool:
    """Return whether a bare or namespaced model ID selects GLM 5.3."""
    return _normalized_model_leaf(model).startswith("glm53")


def prefers_native_reasoning(model: str) -> bool:
    """Return whether a model benefits from SubForge's selective thinking path.

    DeepSeek V4 and Nemotron 3 Ultra can switch thinking on for sparse repairs.
    GLM 5.3 and Kimi K3 map the same routing to model-specific effort levels;
    each provider adapter sends only controls documented by that endpoint.
    Other OpenAI-compatible providers receive no speculative fields.
    """
    return (
        is_deepseek_v4_model(model)
        or is_glm_53_model(model)
        or is_kimi_k3_model(model)
        or is_nemotron_3_ultra_model(model)
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Read Retry-After as seconds or an HTTP date."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = str(headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _persistent_rate_limit_wait_seconds(error: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return retry_after
    base = min(PERSISTENT_RATE_LIMIT_MAX_WAIT, 5.0 * (2 ** min(attempt - 1, 4)))
    return min(PERSISTENT_RATE_LIMIT_MAX_WAIT, base + random.uniform(0.0, 1.0))


def _reserve_kimi_k3_retry_wait_seconds(
    base_wait_seconds: float,
    attempt: int,
) -> float:
    """Stagger K3 retries and cool down sustained free-endpoint throttling."""
    global _kimi_k3_next_retry_at

    now = time.monotonic()
    if attempt <= 5:
        cooldown = max(0.0, base_wait_seconds)
    else:
        cooldown_step = min(4, (attempt - 6) // 5)
        cooldown = max(
            base_wait_seconds,
            min(KIMI_K3_RATE_LIMIT_MAX_WAIT, 60.0 * (2**cooldown_step)),
        )
    with _kimi_k3_retry_lock:
        retry_at = max(
            now + cooldown,
            _kimi_k3_next_retry_at,
        )
        _kimi_k3_next_retry_at = retry_at + KIMI_K3_RATE_LIMIT_RETRY_SPACING
    return max(0.0, retry_at - now)


def _reserve_nemotron_ultra_retry_wait_seconds(base_wait_seconds: float) -> float:
    """Stagger concurrent retries against Nemotron Ultra's shared free pool."""
    global _nemotron_ultra_next_retry_at

    now = time.monotonic()
    with _nemotron_ultra_retry_lock:
        retry_at = max(now + max(0.0, base_wait_seconds), _nemotron_ultra_next_retry_at)
        _nemotron_ultra_next_retry_at = (
            retry_at + NEMOTRON_3_ULTRA_RETRY_SPACING
        )
    return max(0.0, retry_at - now)


def _call_llm_once(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Perform one LLM API request and log its result."""
    if client is None:
        client = get_llm_client()
    _raise_if_request_cancelled(client)

    reasoning_mode = kwargs.pop("_subforge_reasoning_mode", "default")
    max_output_tokens = kwargs.pop("_subforge_max_output_tokens", None)
    # This value only separates disk-cache namespaces for global clients. It
    # must never be forwarded to an OpenAI-compatible provider.
    kwargs.pop("_subforge_cache_namespace", None)
    request_kwargs = dict(kwargs)
    deepseek_request = _is_deepseek_client(client) and _is_deepseek_model(model)
    nvidia_request = _is_nvidia_client(client)
    nvidia_deepseek_v4_request = nvidia_request and is_deepseek_v4_model(model)
    nvidia_glm_53_request = nvidia_request and is_glm_53_model(model)
    nvidia_kimi_k3_request = nvidia_request and is_kimi_k3_model(model)
    nvidia_nemotron_ultra_request = nvidia_request and is_nemotron_3_ultra_model(model)
    zhipu_glm_53_request = _is_zhipu_client(client) and is_glm_53_model(model)
    lmstudio_qwen_38_request = is_lmstudio_qwen_38_request(model, client)
    if deepseek_request and reasoning_mode != "default":
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
        extra_body["thinking"] = {"type": reasoning_mode}
        request_kwargs["extra_body"] = extra_body
        if reasoning_mode == "enabled":
            request_kwargs.setdefault("reasoning_effort", "high")
    if deepseek_request and max_output_tokens is not None:
        request_kwargs["max_tokens"] = max(256, int(max_output_tokens))

    if nvidia_deepseek_v4_request:
        # NVIDIA exposes DeepSeek V4 reasoning through reasoning_effort rather
        # than DeepSeek's provider-specific thinking object.
        if reasoning_mode == "enabled":
            requested_effort = str(request_kwargs.pop("reasoning_effort", "") or "").lower()
            request_kwargs["reasoning_effort"] = (
                requested_effort if requested_effort in {"high", "max"} else "high"
            )
        elif reasoning_mode == "disabled":
            request_kwargs["reasoning_effort"] = "none"
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max(256, int(max_output_tokens))

    if nvidia_glm_53_request and max_output_tokens is not None:
        # NVIDIA's public GLM contract currently documents only standard
        # OpenAI-compatible controls. Model-family prompts and audit routing
        # still apply, while unsupported thinking fields stay provider-local.
        request_kwargs["max_tokens"] = max(256, int(max_output_tokens))

    if nvidia_kimi_k3_request:
        # Kimi K3 always thinks. Use its documented low/high effort control so
        # routine batches stay economical while confirmed repairs get more depth.
        requested_effort = str(request_kwargs.pop("reasoning_effort", "") or "").lower()
        request_kwargs["reasoning_effort"] = (
            requested_effort
            if reasoning_mode == "enabled" and requested_effort in {"high", "max"}
            else "high"
            if reasoning_mode == "enabled"
            else "low"
        )
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max(256, int(max_output_tokens))
        request_kwargs["temperature"] = 1.0
        request_kwargs.setdefault("top_p", 0.95)
        # NVIDIA's free K3 endpoint can queue a valid request for longer than
        # the generic two-minute client deadline. Avoid turning slow service
        # into duplicate retries and additional rate-limit pressure.
        request_kwargs.setdefault("timeout", KIMI_K3_REQUEST_TIMEOUT)

    if nvidia_nemotron_ultra_request:
        # The hosted endpoint defaults to generating a full reasoning trace.
        # Keep constrained subtitle work direct and reserve thinking for the
        # sparse semantic repairs selected by the existing audit pipeline.
        request_kwargs.pop("reasoning_effort", None)
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        chat_template_kwargs["enable_thinking"] = reasoning_mode == "enabled"
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        request_kwargs["extra_body"] = extra_body
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max(256, int(max_output_tokens))
        request_kwargs["temperature"] = 1.0
        request_kwargs.setdefault("top_p", 0.95)
        request_kwargs.setdefault("timeout", NEMOTRON_3_ULTRA_REQUEST_TIMEOUT)

    if zhipu_glm_53_request:
        # GLM 5.3/5.3-Flash reject thinking.type=disabled. Preserve SubForge's
        # selective policy by mapping routine work to low effort and confirmed
        # semantic rewrites to high effort instead of forwarding the raw switch.
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
        extra_body["thinking"] = {"type": "enabled"}
        request_kwargs["extra_body"] = extra_body
        requested_effort = str(request_kwargs.pop("reasoning_effort", "") or "").lower()
        if reasoning_mode == "enabled":
            request_kwargs["reasoning_effort"] = (
                requested_effort if requested_effort in {"high", "max"} else "high"
            )
        else:
            request_kwargs["reasoning_effort"] = "low"
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max(256, int(max_output_tokens))
        request_kwargs["temperature"] = 1.0
        request_kwargs.setdefault("top_p", 0.95)

    if lmstudio_qwen_38_request:
        # LM Studio's Qwen 3.8 template ignores enable_thinking=False and can
        # place the entire answer in reasoning_content. Its OpenAI-compatible
        # reasoning_effort control is reliable: routine work must use none,
        # while confirmed semantic repairs use low. High effort is much slower
        # on local Apple Silicon and did not improve the validated repair probe.
        request_kwargs.pop("extra_body", None)
        request_kwargs.pop("reasoning_effort", None)
        request_kwargs["reasoning_effort"] = (
            "low" if reasoning_mode == "enabled" else "none"
        )
        request_kwargs["temperature"] = 0.1 if reasoning_mode == "enabled" else 0.0
        if max_output_tokens is not None:
            local_budget = 1536 if reasoning_mode == "enabled" else 1024
            request_kwargs["max_tokens"] = max(
                256,
                min(local_budget, int(max_output_tokens)),
            )
        request_kwargs.setdefault("timeout", LMSTUDIO_LOCAL_REQUEST_TIMEOUT)

    # DeepSeek ignores sampling parameters in thinking mode. Omitting them keeps
    # request logs honest and avoids relying on compatibility-only behavior.
    thinking_deepseek_request = (
        deepseek_request or nvidia_deepseek_v4_request
    ) and reasoning_mode == "enabled"
    if (
        not zhipu_glm_53_request
        and not nvidia_kimi_k3_request
        and not nvidia_nemotron_ultra_request
        and not lmstudio_qwen_38_request
        and not thinking_deepseek_request
    ):
        request_kwargs["temperature"] = temperature

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            **request_kwargs,
        )
    except Exception as exc:
        telemetry_entry = log_llm_error(exc)
        telemetry = telemetry_for_client(client)
        if telemetry is not None and telemetry_entry is not None:
            telemetry.record_attempt(
                telemetry_entry,
                succeeded=False,
                reasoning_mode=reasoning_mode,
            )
        raise

    telemetry_entry = log_llm_response(response)
    telemetry = telemetry_for_client(client)
    if telemetry is not None and telemetry_entry is not None:
        telemetry.record_attempt(
            telemetry_entry,
            succeeded=True,
            reasoning_mode=reasoning_mode,
        )
    return response


def _call_standard_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Call non-M3 models with the existing bounded retry policy."""
    for attempt in range(1, 11):
        try:
            return _call_llm_once(messages, model, temperature, client=client, **kwargs)
        except Exception as error:
            if not _is_retryable_standard_error(error) or attempt >= 10:
                raise
            upper = min(60.0, max(5.0, float(2**attempt)))
            wait_seconds = random.uniform(5.0, upper)
            logger.warning(
                "Transient LLM API error (%s), waiting %.1fs before retry %d/10",
                type(error).__name__,
                wait_seconds,
                attempt + 1,
            )
            _record_retry_telemetry(
                client,
                kind=("rate_limit" if isinstance(error, openai.RateLimitError) else "transient"),
                wait_seconds=wait_seconds,
            )
            _wait_for_retry(client, wait_seconds)
    raise RuntimeError("LLM retry loop exited unexpectedly")


def _call_until_provider_available(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    provider_name: str = "LLM provider",
    stagger_rate_limits: bool = False,
    stagger_nemotron_rate_limits: bool = False,
    transient_max_attempts: int = PERSISTENT_TRANSIENT_MAX_ATTEMPTS,
    stagger_transient_errors: bool = False,
    **kwargs: Any,
) -> Any:
    """Wait through rate limits until a persistent provider accepts the request."""
    attempt = 0
    transient_attempt = 0
    while True:
        try:
            return _call_llm_once(messages, model, temperature, client=client, **kwargs)
        except (openai.RateLimitError, anthropic.RateLimitError) as error:
            attempt += 1
            wait_seconds = _persistent_rate_limit_wait_seconds(error, attempt)
            if stagger_rate_limits:
                wait_seconds = _reserve_kimi_k3_retry_wait_seconds(wait_seconds, attempt)
            elif stagger_nemotron_rate_limits:
                wait_seconds = _reserve_nemotron_ultra_retry_wait_seconds(wait_seconds)
            logger.warning(
                "%s is rate limited; waiting %.1fs before retry %d. "
                "The task will remain active until service recovers.",
                provider_name,
                wait_seconds,
                attempt,
            )
            _record_retry_telemetry(
                client,
                kind="rate_limit",
                wait_seconds=wait_seconds,
            )
            _wait_for_retry(client, wait_seconds)
        except (
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as error:
            transient_attempt += 1
            if transient_attempt >= transient_max_attempts:
                raise
            wait_seconds = min(30.0, float(2**transient_attempt))
            if stagger_transient_errors:
                wait_seconds = _reserve_nemotron_ultra_retry_wait_seconds(wait_seconds)
            logger.warning(
                "%s request failed with %s; waiting %.1fs before bounded retry %d/%d.",
                provider_name,
                type(error).__name__,
                wait_seconds,
                transient_attempt + 1,
                transient_max_attempts,
            )
            _record_retry_telemetry(
                client,
                kind="transient",
                wait_seconds=wait_seconds,
            )
            _wait_for_retry(client, wait_seconds)


def _call_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to the model-specific retry policy."""
    if _is_nvidia_client(client):
        nemotron_ultra = is_nemotron_3_ultra_model(model)
        return _call_until_provider_available(
            messages,
            model,
            temperature,
            client=client,
            provider_name="NVIDIA API",
            stagger_rate_limits=is_kimi_k3_model(model),
            stagger_nemotron_rate_limits=nemotron_ultra,
            transient_max_attempts=(
                NEMOTRON_3_ULTRA_TRANSIENT_MAX_ATTEMPTS
                if nemotron_ultra
                else PERSISTENT_TRANSIENT_MAX_ATTEMPTS
            ),
            stagger_transient_errors=nemotron_ultra,
            **kwargs,
        )
    if _is_zhipu_client(client) and is_glm_53_model(model):
        return _call_until_provider_available(
            messages,
            model,
            temperature,
            client=client,
            provider_name="Zhipu GLM API",
            **kwargs,
        )
    if _is_deepseek_client(client) and _is_deepseek_model(model):
        return _call_until_provider_available(
            messages,
            model,
            temperature,
            client=client,
            provider_name="DeepSeek API",
            **kwargs,
        )
    if _is_minimax_m3_model(model):
        return _call_until_provider_available(
            messages,
            model,
            temperature,
            client=client,
            provider_name="MiniMax M3",
            **kwargs,
        )
    return _call_standard_llm_api(
        messages,
        model,
        temperature,
        client=client,
        **kwargs,
    )


def call_llm(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    use_cache: bool = True,
    reasoning_mode: ReasoningMode = "default",
    max_output_tokens: Optional[int] = None,
    cache_namespace: str = "",
    **kwargs: Any,
) -> Any:
    """Call LLM API with optional caching.

    Args:
        client: Optional pre-created OpenAI client. When provided, bypasses the
            global singleton and its environment-variable dependency — avoids
            race conditions when multiple tasks use different credentials.
            Also bypasses cache (client object is not serializable).
        use_cache: Whether to use the disk LLM cache for global-client calls.
        cache_namespace: Optional pipeline identity appended to global cache keys.
    """
    if reasoning_mode not in {"default", "enabled", "disabled"}:
        raise ValueError(f"Unsupported reasoning_mode: {reasoning_mode}")
    kwargs["_subforge_reasoning_mode"] = reasoning_mode
    if max_output_tokens is not None:
        kwargs["_subforge_max_output_tokens"] = max_output_tokens

    if client is not None:
        # Explicit client: skip cache, call directly
        response = _call_llm_api(messages, model, temperature, client=client, **kwargs)
    elif not use_cache:
        response = _call_llm_api(messages, model, temperature, **kwargs)
    else:
        # Global singleton path: use cache
        provider_namespace = normalize_base_url(os.getenv("OPENAI_BASE_URL", "").strip())
        scoped_namespace = str(cache_namespace or "").strip()
        if scoped_namespace:
            provider_namespace = f"{provider_namespace}|{scoped_namespace}"
        response = _call_llm_cached(
            messages,
            model,
            temperature,
            _subforge_cache_namespace=provider_namespace,
            **kwargs,
        )

    return response


@memoize(get_llm_cache(), expire=3600, typed=True)
def _call_llm_cached(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """Cached LLM call via global singleton."""
    return _call_llm_api(messages, model, temperature, **kwargs)
