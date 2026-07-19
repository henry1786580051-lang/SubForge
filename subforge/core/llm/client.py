"""Unified LLM client for the application."""

import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

import anthropic
import openai
from openai import OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from subforge.core.utils.cache import get_llm_cache, memoize
from subforge.core.utils.logger import setup_logger

from .anthropic_client import MiniMaxAnthropicClient
from .request_logger import create_logging_http_client, log_llm_error, log_llm_response
from .response import get_response_text

_global_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

logger = setup_logger("llm_client")

# Timeout for LLM API calls (seconds)
LLM_TIMEOUT = 120.0
MINIMAX_M3_MAX_RATE_LIMIT_WAIT = 60.0


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
            http_client=http_client,
        )
    setattr(client, "_subforge_log_context", log_context)
    return client


def set_client_log_context(client: Any, **context: str) -> None:
    """Attach task metadata without changing the public client constructor."""
    target = getattr(client, "_subforge_log_context", None)
    if isinstance(target, dict):
        target.update({key: value for key, value in context.items() if value})


def get_llm_client() -> OpenAI:
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

                _global_client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=LLM_TIMEOUT,
                    http_client=create_logging_http_client(),
                )

    return _global_client


def before_sleep_log(retry_state: RetryCallState) -> None:
    logger.warning(
        "Rate Limit Error, sleeping and retrying... Please lower your thread concurrency or use better OpenAI API."
    )


def _is_minimax_m3_model(model: str) -> bool:
    """Return whether the selected model needs MiniMax M3's persistent 429 policy."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(model or "").lower())
    return normalized == "minimaxm3"


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


def _minimax_m3_wait_seconds(error: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return retry_after
    base = min(MINIMAX_M3_MAX_RATE_LIMIT_WAIT, 5.0 * (2 ** min(attempt - 1, 4)))
    return min(MINIMAX_M3_MAX_RATE_LIMIT_WAIT, base + random.uniform(0.0, 1.0))


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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=temperature,
            **kwargs,
        )
    except Exception as exc:
        log_llm_error(exc)
        raise

    log_llm_response(response)
    return response


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
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


def _call_minimax_m3_until_available(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """Wait through MiniMax M3 rate limits until the provider accepts the request."""
    attempt = 0
    while True:
        try:
            return _call_llm_once(messages, model, temperature, client=client, **kwargs)
        except (openai.RateLimitError, anthropic.RateLimitError) as error:
            attempt += 1
            wait_seconds = _minimax_m3_wait_seconds(error, attempt)
            logger.warning(
                "MiniMax M3 is rate limited; waiting %.1fs before retry %d. "
                "The task will remain active until service recovers.",
                wait_seconds,
                attempt,
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
    if _is_minimax_m3_model(model):
        return _call_minimax_m3_until_available(
            messages,
            model,
            temperature,
            client=client,
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
    if client is not None:
        # Explicit client: skip cache, call directly
        response = _call_llm_api(messages, model, temperature, client=client, **kwargs)
    elif not use_cache:
        response = _call_llm_api(messages, model, temperature, **kwargs)
    else:
        # Global singleton path: use cache
        response = _call_llm_cached(messages, model, temperature, **kwargs)

    get_response_text(response)

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
