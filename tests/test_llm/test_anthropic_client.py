"""Tests for the MiniMax Anthropic protocol adapter."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from subforge.core.llm import anthropic_client as client_module


class _FakeMessages:
    def __init__(self, callback):
        self._callback = callback

    def create(self, **kwargs):
        return self._callback(kwargs)


class _FakeAnthropic:
    callback = None

    def __init__(self, **_kwargs):
        self.messages = _FakeMessages(self.callback)
        self.models = SimpleNamespace(list=lambda: [])


def _create_client(monkeypatch, callback):
    _FakeAnthropic.callback = staticmethod(callback)
    monkeypatch.setattr(client_module.anthropic, "Anthropic", _FakeAnthropic)
    return client_module.MiniMaxAnthropicClient(
        base_url="https://api.minimaxi.com/anthropic",
        api_key="test-key",
        timeout=30,
        http_client=SimpleNamespace(),
    )


def test_adapter_marks_static_system_prompt_for_explicit_cache(monkeypatch):
    requests = []
    client = _create_client(monkeypatch, lambda request: requests.append(request) or object())

    client.chat.completions.create(
        model="MiniMax-M2.7",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Translate faithfully."},
            {"role": "user", "content": "First subtitle batch"},
        ],
    )

    assert requests[0]["system"] == [
        {
            "type": "text",
            "text": "Translate faithfully.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert requests[0]["messages"] == [
        {"role": "user", "content": "First subtitle batch"}
    ]
    assert requests[0]["max_tokens"] == client_module.DEFAULT_MAX_OUTPUT_TOKENS


def test_same_prompt_waits_for_cache_creator_before_parallel_requests(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def request_api(request):
        with calls_lock:
            calls.append(request["messages"][0]["content"])
            call_number = len(calls)
        if call_number == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return object()

    client = _create_client(monkeypatch, request_api)

    def send(content):
        return client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "Stable prompt"},
                {"role": "user", "content": content},
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(send, "first")
        assert first_started.wait(timeout=2)
        second = executor.submit(send, "second")
        time.sleep(0.05)
        assert calls == ["first"]
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert calls == ["first", "second"]


def test_failed_cache_creator_allows_waiter_to_take_over(monkeypatch):
    attempts = 0

    def request_api(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return object()

    client = _create_client(monkeypatch, request_api)
    request = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "system", "content": "Stable prompt"},
            {"role": "user", "content": "batch"},
        ],
    }

    try:
        client.chat.completions.create(**request)
    except RuntimeError:
        pass
    client.chat.completions.create(**request)

    assert attempts == 2


def test_m3_skips_unsupported_explicit_cache_and_serial_gate(monkeypatch):
    requests = []
    client = _create_client(monkeypatch, lambda request: requests.append(request) or object())

    client.chat.completions.create(
        model="MiniMax-M3",
        messages=[
            {"role": "system", "content": "Stable prompt"},
            {"role": "user", "content": "batch"},
        ],
    )

    assert "cache_control" not in requests[0]["system"][0]
    assert client._gates == {}
