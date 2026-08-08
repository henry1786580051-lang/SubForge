"""Shared LLM provider routing and runtime configuration checks."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

LLM_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "deepseek": "https://api.deepseek.com",
    "mimo": "https://token-plan-cn.xiaomimimo.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "baichuan": "https://api.baichuan-ai.com/v1",
    "yi": "https://api.lingyiwanwu.com/v1",
    "minimax": "https://api.minimaxi.com/anthropic",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}

LEGACY_MINIMAX_URLS = {
    "https://api.minimax.chat/v1",
    "https://api.minimaxi.com/v1",
}

_AGGREGATOR_PROVIDERS = {"custom", "nvidia", "openrouter", "siliconflow"}


@dataclass(frozen=True)
class LlmRuntimeConfig:
    """One immutable provider snapshot used for an entire task."""

    provider: str
    base_url: str
    api_key: str
    model: str


def detect_llm_provider(base_url: str) -> str:
    """Infer a direct provider from its endpoint without inspecting credentials."""
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    if hostname in {"api.minimax.chat", "api.minimaxi.com"}:
        return "minimax"
    for provider, default_url in LLM_PROVIDER_URLS.items():
        default_hostname = (urlparse(default_url).hostname or "").lower()
        if default_hostname and hostname == default_hostname:
            return provider
    return "custom"


def model_provider_hint(model: str) -> str | None:
    """Return a provider only for unambiguous direct-provider model names."""
    normalized = str(model or "").strip().lower()
    if normalized.startswith("deepseek-"):
        return "deepseek"
    if normalized.startswith("mimo-"):
        return "mimo"
    if normalized.startswith("minimax-"):
        return "minimax"
    if normalized.startswith(("qwen-", "qwen2", "qwen3")):
        return "qwen"
    if normalized.startswith("glm-"):
        return "zhipu"
    if normalized.startswith(("moonshot-", "kimi-")):
        return "moonshot"
    if normalized.startswith("baichuan"):
        return "baichuan"
    return None


def validate_llm_runtime_config(config: LlmRuntimeConfig) -> None:
    """Reject mixed-provider tuples before any API request is sent."""
    provider = str(config.provider or "custom").strip().lower()
    detected = detect_llm_provider(config.base_url)
    if provider not in _AGGREGATOR_PROVIDERS and detected != provider:
        raise ValueError(
            f"LLM provider '{provider}' does not match BaseURL provider '{detected}'. "
            "Switch providers or save the matching BaseURL before starting the task."
        )

    model_provider = model_provider_hint(config.model)
    if (
        model_provider
        and provider not in _AGGREGATOR_PROVIDERS
        and model_provider != provider
    ):
        raise ValueError(
            f"LLM model '{config.model}' belongs to '{model_provider}', but the active "
            f"provider is '{provider}'. Select the model from the active provider profile."
        )
