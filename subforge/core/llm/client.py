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
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from subforge.core.utils.cache import get_llm_cache, memoize
from subforge.core.utils.logger import setup_logger

from .anthropic_client import MiniMaxAnthropicClient
from .request_logger import create_logging_http_client, log_llm_error, log_llm_response

_global_client: Optional[Any] = None
_client_lock = threading.Lock()

logger = setup_logger("llm_client")

# Timeout for LLM API calls (seconds)
LLM_TIMEOUT = 120.0
PERSISTENT_RATE_LIMIT_MAX_WAIT = 60.0
PERSISTENT_TRANSIENT_MAX_ATTEMPTS = 3
ReasoningMode = Literal["default", "enabled", "disabled"]


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


def create_client(base_url: str, api_key: str) -> Any:
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
            timeout=LLM_TIMEOUT,
            http_client=http_client,
        )
    else:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=LLM_TIMEOUT,
            max_retries=0,
            http_client=http_client,
        )
    setattr(client, "_subforge_log_context", log_context)
    setattr(client, "_subforge_base_url", base_url)
    return client


def set_client_log_context(client: Any, **context: str) -> None:
    """Attach task metadata without changing the public client constructor."""
    target = getattr(client, "_subforge_log_context", None)
    if isinstance(target, dict):
        target.update({key: value for key, value in context.items() if value})


def get_llm_client() -> Any:
    """Get global LLM client instance (thread-safe singleton)."""
    global _global_client

    if _global_client is None:
        with _client_lock:
            if _global_client is None:
                base_url = os.getenv("OPENAI_BASE_URL", "").strip()
                base_url = normalize_base_url(base_url)
                api_key = os.getenv("OPENAI_API_KEY", "").strip()

                if not base_url or not api_key:
                    raise ValueError(
                        "OPENAI_BASE_URL and OPENAI_API_KEY environment variables must be set"
                    )

                _global_client = create_client(base_url, api_key)

    return _global_client


def before_sleep_log(retry_state: RetryCallState) -> None:
    error = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Transient LLM API error (%s), sleeping before retry %d/10",
        type(error).__name__ if error else "unknown",
        retry_state.attempt_number + 1,
    )


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


def _is_deepseek_model(model: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(model or "").lower())
    return normalized.startswith("deepseek")


def prefers_native_reasoning(model: str) -> bool:
    """Return whether a model benefits from SubForge's selective thinking path.

    DeepSeek V4 models expose native thinking controls.  Restricting the policy to
    that family keeps OpenAI-compatible providers from receiving speculative
    parameters and avoids increasing latency for models that do not support them.
    """
    normalized = re.sub(r"[^a-z0-9]+", "", str(model or "").lower())
    return normalized.startswith("deepseek") and "v4" in normalized


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

    reasoning_mode = kwargs.pop("_subforge_reasoning_mode", "default")
    max_output_tokens = kwargs.pop("_subforge_max_output_tokens", None)
    request_kwargs = dict(kwargs)
    deepseek_request = _is_deepseek_client(client) and _is_deepseek_model(model)
    if deepseek_request and reasoning_mode != "default":
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
        extra_body["thinking"] = {"type": reasoning_mode}
        request_kwargs["extra_body"] = extra_body
        if reasoning_mode == "enabled":
            request_kwargs["reasoning_effort"] = "high"
    if deepseek_request and max_output_tokens is not None:
        request_kwargs["max_tokens"] = max(256, int(max_output_tokens))

    # DeepSeek ignores sampling parameters in thinking mode. Omitting them keeps
    # request logs honest and avoids relying on compatibility-only behavior.
    if not (deepseek_request and reasoning_mode == "enabled"):
        request_kwargs["temperature"] = temperature

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            **request_kwargs,
        )
    except Exception as exc:
        log_llm_error(exc)
        raise

    log_llm_response(response)
    return response


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception(_is_retryable_standard_error),
    before_sleep=before_sleep_log,
)
def _call_standard_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Call non-M3 models with the existing bounded retry policy."""
    return _call_llm_once(messages, model, temperature, client=client, **kwargs)


def _call_until_provider_available(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    provider_name: str = "LLM provider",
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
            logger.warning(
                "%s is rate limited; waiting %.1fs before retry %d. "
                "The task will remain active until service recovers.",
                provider_name,
                wait_seconds,
                attempt,
            )
            time.sleep(wait_seconds)
        except (
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as error:
            transient_attempt += 1
            if transient_attempt >= PERSISTENT_TRANSIENT_MAX_ATTEMPTS:
                raise
            wait_seconds = min(30.0, float(2**transient_attempt))
            logger.warning(
                "%s request failed with %s; waiting %.1fs before bounded retry %d/%d.",
                provider_name,
                type(error).__name__,
                wait_seconds,
                transient_attempt + 1,
                PERSISTENT_TRANSIENT_MAX_ATTEMPTS,
            )
            time.sleep(wait_seconds)


def _call_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to the model-specific retry policy."""
    if _is_nvidia_client(client):
        return _call_until_provider_available(
            messages,
            model,
            temperature,
            client=client,
            provider_name="NVIDIA API",
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
    **kwargs: Any,
) -> Any:
    """Call LLM API with optional caching.

    Args:
        client: Optional pre-created OpenAI client. When provided, bypasses the
            global singleton and its environment-variable dependency — avoids
            race conditions when multiple tasks use different credentials.
            Also bypasses cache (client object is not serializable).
        use_cache: Whether to use the disk LLM cache for global-client calls.
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
        response = _call_llm_cached(messages, model, temperature, **kwargs)

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
