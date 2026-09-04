"""Tests for provider- and model-specific LLM retry policies."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from subforge.core.llm import client as client_module
from subforge.core.llm.telemetry import LLMTaskTelemetry


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


def test_nvidia_kimi_k3_staggers_rate_limit_retries(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error("60"), success]
    reservations = []
    sleeps = []
    client = SimpleNamespace(_subforge_base_url="https://integrate.api.nvidia.com/v1")

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def reserve(wait_seconds, attempt):
        reservations.append((wait_seconds, attempt))
        return 63.0

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(client_module, "_reserve_kimi_k3_retry_wait_seconds", reserve)
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api([], "moonshotai/kimi-k3", client=client)

    assert result is success
    assert reservations == [(60.0, 1)]
    assert sleeps == [63.0]


def test_nvidia_non_k3_models_keep_existing_rate_limit_wait(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error("60"), success]
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
        "_reserve_kimi_k3_retry_wait_seconds",
        lambda _seconds, _attempt: pytest.fail(
            "non-K3 models must not use the K3 retry gate"
        ),
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api([], "moonshotai/kimi-k2.6", client=client)

    assert result is success
    assert sleeps == [60.0]


@pytest.mark.parametrize(
    ("attempt", "expected_wait"),
    [
        (1, 5.0),
        (5, 60.0),
        (6, 60.0),
        (11, 120.0),
        (16, 240.0),
        (21, 480.0),
        (26, 600.0),
        (50, 600.0),
    ],
)
def test_kimi_k3_sustained_rate_limit_uses_progressive_cooldown(
    monkeypatch,
    attempt,
    expected_wait,
):
    monkeypatch.setattr(client_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(client_module, "_kimi_k3_next_retry_at", 0.0)

    wait_seconds = client_module._reserve_kimi_k3_retry_wait_seconds(
        5.0 if attempt == 1 else 60.0,
        attempt,
    )

    assert wait_seconds == expected_wait


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


def test_retry_policy_records_wait_without_changing_retry_behavior(monkeypatch):
    telemetry = LLMTaskTelemetry()
    client = SimpleNamespace(
        _subforge_base_url="https://api.deepseek.com/v1",
        _subforge_telemetry=telemetry,
    )
    outcomes = [_rate_limit_error("3"), object()]
    sleeps = []

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    client_module._call_llm_api([], "deepseek-v4-flash", client=client)

    snapshot = telemetry.snapshot()
    assert sleeps == [3.0]
    assert snapshot.retry_count == 1
    assert snapshot.rate_limit_retries == 1
    assert snapshot.retry_wait_ms == 3000


def test_persistent_rate_limit_wait_is_interrupted_by_task_cancellation(monkeypatch):
    client = SimpleNamespace(
        _subforge_base_url="https://api.deepseek.com/v1",
        _subforge_cancel_event=client_module.threading.Event(),
    )

    monkeypatch.setattr(
        client_module,
        "_call_llm_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_rate_limit_error()),
    )
    monkeypatch.setattr(
        client_module,
        "_persistent_rate_limit_wait_seconds",
        lambda _error, _attempt: 60.0,
    )
    client._subforge_cancel_event.set()

    with pytest.raises(client_module.LLMRequestCancelled):
        client_module._call_llm_api([], "deepseek-v4-flash", client=client)


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


def test_deepseek_thinking_request_honors_explicit_reasoning_effort(monkeypatch):
    client, completions = _capturing_client("https://api.deepseek.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "deepseek-v4-flash",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="medium",
        max_output_tokens=8192,
    )

    assert completions.kwargs["reasoning_effort"] == "medium"


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


def test_nvidia_deepseek_thinking_uses_documented_controls(monkeypatch):
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

    assert "temperature" not in completions.kwargs
    assert "extra_body" not in completions.kwargs
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["max_tokens"] == 8192


def test_nvidia_deepseek_routine_translation_disables_thinking(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "deepseek-ai/deepseek-v4-flash",
        temperature=0.1,
        client=client,
        reasoning_mode="disabled",
        max_output_tokens=4096,
    )

    assert completions.kwargs["temperature"] == 0.1
    assert "extra_body" not in completions.kwargs
    assert completions.kwargs["reasoning_effort"] == "none"
    assert completions.kwargs["max_tokens"] == 4096


@pytest.mark.parametrize("reasoning_mode", ["default", "disabled"])
def test_lmstudio_qwen_38_routine_work_disables_reasoning(monkeypatch, reasoning_mode):
    client, completions = _capturing_client("http://127.0.0.1:1234/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "qwen/qwen3.8-27b",
        temperature=0.8,
        client=client,
        reasoning_mode=reasoning_mode,
        reasoning_effort="high",
        max_output_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    assert completions.kwargs["reasoning_effort"] == "none"
    assert completions.kwargs["max_tokens"] == 1024
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["timeout"] == client_module.LMSTUDIO_LOCAL_REQUEST_TIMEOUT
    assert "extra_body" not in completions.kwargs


def test_lmstudio_qwen_38_confirmed_repair_uses_low_reasoning(monkeypatch):
    client, completions = _capturing_client("http://localhost:1234/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair confirmed semantic defect"}],
        "qwen/qwen3.8-27b",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="high",
        max_output_tokens=8192,
    )

    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["max_tokens"] == 1536
    assert completions.kwargs["temperature"] == 0.1
    assert completions.kwargs["timeout"] == client_module.LMSTUDIO_LOCAL_REQUEST_TIMEOUT


def test_lmstudio_qwen_38_workload_limits_are_endpoint_specific():
    local_client = SimpleNamespace(_subforge_base_url="http://127.0.0.1:1234/v1")
    cloud_client = SimpleNamespace(
        _subforge_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    assert client_module.constrain_local_llm_workload(
        "qwen/qwen3.8-27b",
        local_client,
        concurrency=20,
        batch_size=20,
    ) == (1, 10)
    assert client_module.constrain_local_llm_workload(
        "qwen/qwen3.8-27b",
        cloud_client,
        concurrency=20,
        batch_size=20,
    ) == (20, 20)
    assert client_module.constrain_local_llm_workload(
        "google/gemma-4-26b-a4b-qat",
        local_client,
        concurrency=20,
        batch_size=20,
    ) == (20, 20)


def test_nvidia_glm_53_keeps_family_budget_without_unsupported_controls(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "z-ai/glm-5.3-flash",
        temperature=0.1,
        client=client,
        reasoning_mode="enabled",
        max_output_tokens=4096,
    )

    assert completions.kwargs["temperature"] == 0.1
    assert completions.kwargs["max_tokens"] == 4096
    assert "extra_body" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs


@pytest.mark.parametrize("reasoning_mode", ["default", "disabled"])
def test_nvidia_kimi_k3_routine_work_uses_low_reasoning(monkeypatch, reasoning_mode):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "moonshotai/kimi-k3",
        temperature=0.2,
        client=client,
        reasoning_mode=reasoning_mode,
        max_output_tokens=4096,
    )

    assert "extra_body" not in completions.kwargs
    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["temperature"] == 1.0
    assert completions.kwargs["top_p"] == 0.95
    assert completions.kwargs["max_tokens"] == 4096
    assert completions.kwargs["timeout"] == client_module.KIMI_K3_REQUEST_TIMEOUT


def test_nvidia_kimi_k3_confirmed_repair_uses_thinking(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair semantic ownership"}],
        "moonshotai/kimi-k3",
        client=client,
        reasoning_mode="enabled",
        max_output_tokens=6144,
    )

    assert "extra_body" not in completions.kwargs
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["temperature"] == 1.0
    assert completions.kwargs["top_p"] == 0.95
    assert completions.kwargs["max_tokens"] == 6144
    assert completions.kwargs["timeout"] == client_module.KIMI_K3_REQUEST_TIMEOUT


def test_nvidia_kimi_k3_preserves_explicit_max_reasoning(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair ambiguous discourse ownership"}],
        "moonshotai/kimi-k3",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="max",
    )

    assert completions.kwargs["reasoning_effort"] == "max"


@pytest.mark.parametrize("reasoning_mode", ["default", "disabled"])
def test_nvidia_nemotron_ultra_disables_thinking_for_routine_work(
    monkeypatch, reasoning_mode
):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.2,
        client=client,
        reasoning_mode=reasoning_mode,
        reasoning_effort="max",
        max_output_tokens=4096,
    )

    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "reasoning_effort" not in completions.kwargs
    assert completions.kwargs["temperature"] == 1.0
    assert completions.kwargs["top_p"] == 0.95
    assert completions.kwargs["max_tokens"] == 4096
    assert completions.kwargs["timeout"] == client_module.NEMOTRON_3_ULTRA_REQUEST_TIMEOUT


def test_nvidia_nemotron_ultra_enables_thinking_for_confirmed_repair(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair semantic ownership"}],
        "nvidia/nemotron-3-ultra-550b-a55b",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="low",
        max_output_tokens=6144,
    )

    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert "reasoning_effort" not in completions.kwargs
    assert completions.kwargs["max_tokens"] == 6144


def test_nvidia_nemotron_ultra_omits_unsupported_effort_controls(
    monkeypatch,
):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair a confirmed complex semantic defect"}],
        "nvidia/nemotron-3-ultra-550b-a55b",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="high",
    )

    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_nvidia_nemotron_ultra_staggers_rate_limit_retries(monkeypatch):
    success = object()
    outcomes = [_rate_limit_error(), success]
    sleeps = []
    reservations = []
    client = SimpleNamespace(_subforge_base_url="https://integrate.api.nvidia.com/v1")

    def fake_call(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client_module, "_call_llm_once", fake_call)
    monkeypatch.setattr(
        client_module,
        "_reserve_nemotron_ultra_retry_wait_seconds",
        lambda seconds: reservations.append(seconds) or 7.0,
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api(
        [], "nvidia/nemotron-3-ultra-550b-a55b", client=client
    )

    assert result is success
    assert len(reservations) == 1
    assert sleeps == [7.0]


def test_nvidia_nemotron_ultra_waits_through_temporary_overload(monkeypatch):
    success = object()
    outcomes = [_status_error(503) for _ in range(4)] + [success]
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
        "_reserve_nemotron_ultra_retry_wait_seconds",
        lambda seconds: seconds,
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = client_module._call_llm_api(
        [], "nvidia/nemotron-3-ultra-550b-a55b", client=client
    )

    assert result is success
    assert sleeps == [2.0, 4.0, 8.0, 16.0]


def test_nvidia_other_models_do_not_receive_family_controls(monkeypatch):
    client, completions = _capturing_client("https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "translate"}],
        "meta/llama-3.3-70b-instruct",
        temperature=0.2,
        client=client,
        reasoning_mode="enabled",
        max_output_tokens=4096,
    )

    assert completions.kwargs["temperature"] == 0.2
    assert "max_tokens" not in completions.kwargs
    assert "extra_body" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs


def test_glm_53_routine_request_uses_low_always_on_thinking(monkeypatch):
    client, completions = _capturing_client("https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "split"}],
        "glm-5.3-flash",
        temperature=0.1,
        client=client,
        reasoning_mode="disabled",
        max_output_tokens=4096,
    )

    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["max_tokens"] == 4096
    assert completions.kwargs["temperature"] == 1.0
    assert completions.kwargs["top_p"] == 0.95


def test_glm_53_confirmed_repair_uses_high_reasoning(monkeypatch):
    client, completions = _capturing_client("https://open.bigmodel.cn/api/paas/v4/")
    monkeypatch.setattr(client_module, "log_llm_response", lambda _response: None)

    client_module.call_llm(
        [{"role": "user", "content": "repair semantic ownership"}],
        "GLM-5.3-Flash",
        client=client,
        reasoning_mode="enabled",
        reasoning_effort="low",
    )

    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert completions.kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("glm-5.3-flash", True),
        ("GLM_5.3", True),
        ("z-ai/glm-5.3-flash", True),
        ("glm-5.2", False),
        ("deepseek-v4-flash", True),
        ("deepseek-ai/deepseek-v4-pro", True),
        ("deepseek-ai/deepseek-v4-flash-0731", True),
        ("nvidia/deepseek-v4-flash", True),
        ("z-ai/glm-5.3-flash/", True),
        ("moonshotai/kimi-k3", True),
        ("Kimi_K3", True),
        ("moonshotai/kimi-k3.1", False),
        ("moonshotai/kimi-k2.6", False),
        ("nvidia/nemotron-3-ultra-550b-a55b", True),
        ("NVIDIA_Nemotron-3-Ultra-550B-A55B", True),
        ("nvidia/nemotron-3-super-120b-a12b", False),
        ("MiniMax-M3", False),
    ],
)
def test_selective_native_reasoning_model_detection(model, expected):
    assert client_module.prefers_native_reasoning(model) is expected


def test_call_llm_rejects_unknown_reasoning_mode():
    client, _completions = _capturing_client("https://api.deepseek.com")

    with pytest.raises(ValueError, match="Unsupported reasoning_mode"):
        client_module.call_llm(
            [],
            "deepseek-v4-flash",
            client=client,
            reasoning_mode="turbo",  # type: ignore[arg-type]
        )
