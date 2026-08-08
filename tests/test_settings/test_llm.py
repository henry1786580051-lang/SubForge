import pytest

from subforge.settings import (
    LlmRuntimeConfig,
    detect_llm_provider,
    validate_llm_runtime_config,
)


def _runtime(provider: str, base_url: str, model: str) -> LlmRuntimeConfig:
    return LlmRuntimeConfig(
        provider=provider,
        base_url=base_url,
        api_key="test-key",
        model=model,
    )


def test_direct_provider_accepts_matching_model_and_endpoint():
    validate_llm_runtime_config(
        _runtime("deepseek", "https://api.deepseek.com/v1", "deepseek-chat")
    )


def test_direct_provider_rejects_model_from_another_provider():
    with pytest.raises(ValueError, match="belongs to 'deepseek'"):
        validate_llm_runtime_config(
            _runtime(
                "mimo",
                "https://token-plan-cn.xiaomimimo.com/v1",
                "deepseek-chat",
            )
        )


def test_direct_provider_rejects_endpoint_from_another_provider():
    with pytest.raises(ValueError, match="does not match BaseURL"):
        validate_llm_runtime_config(
            _runtime(
                "deepseek",
                "https://token-plan-cn.xiaomimimo.com/v1",
                "deepseek-chat",
            )
        )


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1"),
        ("custom", "https://llm-gateway.example.test/v1"),
    ],
)
def test_aggregators_allow_namespaced_models(provider, base_url):
    validate_llm_runtime_config(
        _runtime(provider, base_url, "deepseek-ai/deepseek-v3.2")
    )


def test_provider_detection_uses_exact_hostname():
    assert detect_llm_provider("https://api.deepseek.com/v1") == "deepseek"
    assert detect_llm_provider("https://api.deepseek.com.example.test/v1") == "custom"
    assert detect_llm_provider("https://api.minimax.chat/v1") == "minimax"
