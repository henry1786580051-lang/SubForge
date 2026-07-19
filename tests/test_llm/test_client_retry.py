"""Tests for model-specific LLM retry policies."""

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
    assert client_module._minimax_m3_wait_seconds(_rate_limit_error("120"), 1) == 120.0


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
        "_minimax_m3_wait_seconds",
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
