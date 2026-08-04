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


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
    )
    error_type = openai.InternalServerError if status >= 500 else openai.APIStatusError
    return error_type("provider error", response=response, body={})


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


def test_standard_retry_only_accepts_transient_provider_failures():
    assert client_module._is_retryable_standard_error(_rate_limit_error()) is True
    assert client_module._is_retryable_standard_error(_status_error(503)) is True
    assert (
        client_module._is_retryable_standard_error(
            openai.APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))
        )
        is True
    )
    assert client_module._is_retryable_standard_error(_status_error(400)) is False


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
                request=httpx.Request(
                    "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
                ),
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


def test_deepseek_waits_until_rate_limit_recovers(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error(), _rate_limit_error(), success]
    sleeps = []
    client = SimpleNamespace(_subforge_base_url="https://api.deepseek.com/v1")

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

    result = client_module._call_llm_api(
        [],
        "deepseek-v4-flash",
        client=client,
    )

    assert result is success
    assert sleeps == [1.0, 2.0]
    assert outcomes == []


def test_deepseek_does_not_wait_for_non_rate_limit_errors(monkeypatch):
    client = SimpleNamespace(_subforge_base_url="https://api.deepseek.com/v1")

    def fail(*_args, **_kwargs):
        raise openai.AuthenticationError(
            "bad key",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions"),
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
        client_module._call_llm_api([], "deepseek-v4-flash", client=client)


def test_deepseek_retries_transient_timeouts_but_not_forever(monkeypatch):
    client = SimpleNamespace(_subforge_base_url="https://api.deepseek.com/v1")
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    outcomes = [
        openai.APITimeoutError(request=request),
        openai.APITimeoutError(request=request),
        object(),
    ]
    sleeps = []

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api([], "deepseek-v4-flash", client=client)

    assert result is not None
    assert sleeps == [2.0, 4.0]


class _CapturingCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def _capturing_client(base_url: str):
    completions = _CapturingCompletions()
    client = SimpleNamespace(
        _subforge_base_url=base_url,
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def test_deepseek_thinking_request_uses_official_controls(monkeypatch):
    client, completions = _capturing_client("https://api.deepseek.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "deepseek-v4-flash",
        temperature=0.2,
        client=client,
        reasoning_mode="enabled",
        max_output_tokens=8192,
    )

    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["max_tokens"] == 8192
    assert "temperature" not in completions.kwargs


def test_deepseek_non_thinking_request_keeps_sampling(monkeypatch):
    client, completions = _capturing_client("https://api.deepseek.com")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "split"}],
        "deepseek-v4-flash",
        temperature=0.1,
        client=client,
        reasoning_mode="disabled",
        max_output_tokens=4096,
    )

    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert completions.kwargs["max_tokens"] == 4096
    assert completions.kwargs["temperature"] == 0.1
    assert "reasoning_effort" not in completions.kwargs


def test_deepseek_controls_are_not_sent_to_other_providers(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "audit"}],
        "deepseek-ai/deepseek-v4-pro",
        temperature=0.3,
        client=client,
        reasoning_mode="enabled",
        max_output_tokens=8192,
    )

    assert completions.kwargs["temperature"] == 0.3
    assert "extra_body" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs
    assert "max_tokens" not in completions.kwargs


def test_call_llm_rejects_unknown_reasoning_mode():
    client, _completions = _capturing_client("https://api.deepseek.com")

    with pytest.raises(ValueError, match="Unsupported reasoning_mode"):
        client_module.call_llm(
            [],
            "deepseek-v4-flash",
            client=client,
            reasoning_mode="turbo",  # type: ignore[arg-type]
        )
