"""Tests for provider- and model-specific LLM retry policies."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from subforge.core.llm import client as client_module


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.minimax.chat/v1/chat/completions"),
    )
    return openai.RateLimitError("rate limited", response=response, body={})


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("MiniMax-M3", True),
        ("minimax_m3", True),
        ("MiniMax-M2.5", False),
        ("m3", False),
        ("gpt-4o-mini", False),
    ],
)
def test_minimax_m3_model_detection_is_specific(model, expected):
    assert client_module._is_minimax_m3_model(model) is expected


def test_retry_after_seconds_supports_seconds_and_invalid_values():
    assert client_module._retry_after_seconds(_rate_limit_error("17")) == 17.0
    assert client_module._retry_after_seconds(_rate_limit_error("invalid")) is None


def test_minimax_m3_honors_retry_after_beyond_local_backoff_cap():
    assert client_module._persistent_rate_limit_wait_seconds(_rate_limit_error("120"), 1) == 120.0


def test_minimax_m3_waits_until_rate_limit_recovers(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error(), _rate_limit_error(), success]
    sleeps = []

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(
        client_module,
        "_persistent_rate_limit_wait_seconds",
        lambda _error, attempt: float(attempt),
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api([], "MiniMax-M3")

    assert result is success
    assert sleeps == [1.0, 2.0]
    assert outcomes == []


def test_minimax_m3_does_not_retry_non_rate_limit_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise openai.AuthenticationError(
            "bad key",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.minimax.chat/v1/chat/completions"),
            ),
            body={},
        )

    monkeypatch.setattr(client_module, "_call_llm_once", fail)
    monkeypatch.setattr(
        client_module.time,
        "sleep",
        lambda _seconds: pytest.fail("non-rate-limit errors must not sleep"),
    )

    with pytest.raises(openai.AuthenticationError):
        client_module._call_llm_api([], "MiniMax-M3")


def test_other_models_keep_standard_retry_dispatch(monkeypatch):
    sentinel = object()
    calls = []

    def standard(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(client_module, "_call_standard_llm_api", standard)

    assert client_module._call_llm_api([], "MiniMax-M2.5") is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://integrate.api.nvidia.com/v1", True),
        ("https://integrate.api.nvidia.com/v1/", True),
        ("https://INTEGRATE.API.NVIDIA.COM/v1", True),
        ("https://integrate.api.nvidia.com.evil.test/v1", False),
        ("https://openrouter.ai/api/v1", False),
    ],
)
def test_nvidia_client_detection_uses_exact_endpoint_host(base_url, expected):
    client = SimpleNamespace(_subforge_base_url=base_url)

    assert client_module._is_nvidia_client(client) is expected


def test_nvidia_detection_supports_global_environment_client(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

    assert client_module._is_nvidia_client() is True


def test_nvidia_waits_until_rate_limit_recovers_for_third_party_model(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error(), _rate_limit_error(), success]
    sleeps = []
    client = SimpleNamespace(_subforge_base_url="https://integrate.api.nvidia.com/v1")

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(
        client_module,
        "_persistent_rate_limit_wait_seconds",
        lambda _error, attempt: float(attempt),
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api([], "deepseek-ai/deepseek-v4-pro", client=client)

    assert result is success
    assert sleeps == [1.0, 2.0]
    assert outcomes == []


def test_nvidia_does_not_retry_authentication_errors(monkeypatch):
    client = SimpleNamespace(_subforge_base_url="https://integrate.api.nvidia.com/v1")

    def fail(*_args, **_kwargs):
        raise openai.AuthenticationError(
            "bad key",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
            ),
            body={},
        )

    monkeypatch.setattr(client_module, "_call_llm_once", fail)
    monkeypatch.setattr(
        client_module.time,
        "sleep",
        lambda _seconds: pytest.fail("non-rate-limit errors must not sleep"),
    )

    with pytest.raises(openai.AuthenticationError):
        client_module._call_llm_api([], "nvidia/nemotron-3-nano-30b-a3b", client=client)
