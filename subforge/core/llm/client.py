"""Unified LLM client for the application."""

import os
import threading
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

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

from .request_logger import create_logging_http_client, log_llm_response

_global_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

logger = setup_logger("llm_client")

# Timeout for LLM API calls (seconds)
LLM_TIMEOUT = 120.0


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


def create_client(base_url: str, api_key: str) -> OpenAI:
    """Create a new OpenAI client with explicit credentials (avoids global state)."""
    base_url = normalize_base_url(base_url)
    if not base_url or not api_key:
        raise ValueError("base_url and api_key are required")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=LLM_TIMEOUT,
        http_client=create_logging_http_client(),
    )


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


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def _call_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    client: Optional[OpenAI] = None,
    **kwargs: Any,
) -> Any:
    """实际调用 LLM API（带重试）"""
    if client is None:
        client = get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        **kwargs,
    )

    # 记录响应内容
    log_llm_response(response)

    return response


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

    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and response.choices[0].message.content
    ):
        raise ValueError("Invalid OpenAI API response: empty choices or content")

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
