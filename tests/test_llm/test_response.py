"""Tests for OpenAI-compatible response normalization."""

from types import SimpleNamespace

import pytest

from subforge.core.llm.response import (
    get_response_history_message,
    get_response_text,
    parse_json_object,
    strip_reasoning_blocks,
)


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def test_strip_reasoning_blocks_keeps_only_final_answer():
    content = "<think>private analysis</think>\n最终译文"

    assert strip_reasoning_blocks(content) == "最终译文"
    assert get_response_text(_Response(content)) == "最终译文"


def test_get_response_text_rejects_reasoning_only_response():
    with pytest.raises(ValueError, match="no final answer"):
        get_response_text(_Response("<think>analysis only</think>"))


def test_get_response_text_uses_only_anthropic_text_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="internal reasoning"),
            SimpleNamespace(type="text", text='{"1":"译文"}'),
        ]
    )

    assert get_response_text(response) == '{"1":"译文"}'


def test_get_response_history_message_preserves_kimi_reasoning():
    message = SimpleNamespace(
        content='{"1":"译文"}',
        reasoning_content="private chain",
        reasoning=None,
        tool_calls=None,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    assert get_response_history_message(response) == {
        "role": "assistant",
        "content": '{"1":"译文"}',
        "reasoning_content": "private chain",
    }


def test_parse_json_object_handles_reasoning_fence_and_numeric_keys():
    content = '<think>analysis</think>\n```json\n{1: "你好", "2": "世界"}\n```'

    assert parse_json_object(content) == {"1": "你好", "2": "世界"}


def test_parse_json_object_handles_common_list_wrappers():
    assert parse_json_object('[{"1": "你好"}, {"2": "世界"}]') == {
        "1": "你好",
        "2": "世界",
    }
    assert parse_json_object('[{"key": 1, "translation": "你好"}]') == {"1": "你好"}


def test_parse_json_object_rejects_plain_list():
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object('["你好", "世界"]')
