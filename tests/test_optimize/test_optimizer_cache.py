import json

from subforge.core.optimize import optimize as optimize_module
from subforge.core.optimize.optimize import SubtitleOptimizer


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)


class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]


def test_optimizer_can_bypass_llm_cache(monkeypatch):
    calls = []

    def fake_call_llm(*, messages, model, temperature, use_cache=True, client=None):
        calls.append(use_cache)
        return DummyResponse(json.dumps({"1": "Hello world"}))

    monkeypatch.setattr(optimize_module, "call_llm", fake_call_llm)

    optimizer = SubtitleOptimizer(
        thread_num=1,
        batch_num=10,
        model="test-model",
        custom_prompt="",
        use_cache=False,
    )

    result = optimizer.agent_loop({"1": "Hello world"})

    assert result == {"1": "Hello world"}
    assert calls == [False]


def test_optimizer_preserves_original_batch_after_invalid_retries(monkeypatch):
    calls = []

    def fake_call_llm(*, messages, model, temperature, use_cache=True, client=None):
        calls.append(messages)
        return DummyResponse(
            json.dumps(
                {
                    "1": "Second subtitle.",
                    "2": "",
                }
            )
        )

    monkeypatch.setattr(optimize_module, "call_llm", fake_call_llm)
    original = {
        "1": "First subtitle.",
        "2": "Second subtitle.",
    }
    optimizer = SubtitleOptimizer(
        thread_num=1,
        batch_num=10,
        model="test-model",
        custom_prompt="",
        use_cache=False,
    )

    result = optimizer.agent_loop(original)

    assert result == original
    assert len(calls) == 3
