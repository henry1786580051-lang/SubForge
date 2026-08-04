"""Unit tests for the official Microsoft Azure Translator client."""

from dataclasses import dataclass, field

import pytest
import requests

import subforge.core.translate.bing_translator as bing_module
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.bing_translator import BingTranslator
from subforge.core.translate.factory import TranslatorFactory
from subforge.core.translate.types import TargetLanguage, TranslatorType


@dataclass
class FakeResponse:
    status_code: int
    payload: object
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def make_translator(session, **kwargs):
    return BingTranslator(
        thread_num=1,
        batch_num=10,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        update_callback=None,
        use_cache=False,
        api_key="azure-secret",
        region="eastasia",
        session=session,
        **kwargs,
    )


def test_requires_subscription_key(monkeypatch):
    monkeypatch.delenv("AZURE_TRANSLATOR_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API Key is not configured"):
        BingTranslator(
            thread_num=1,
            batch_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            update_callback=None,
            api_key="",
        )


def test_factory_passes_azure_credentials_to_bing(monkeypatch):
    captured = {}

    class FakeBingTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "subforge.core.translate.factory.BingTranslator", FakeBingTranslator
    )

    TranslatorFactory.create_translator(
        translator_type=TranslatorType.BING,
        azure_translator_key="factory-key",
        azure_translator_region="eastasia",
        azure_translator_endpoint="https://example.test",
    )

    assert captured["api_key"] == "factory-key"
    assert captured["region"] == "eastasia"
    assert captured["endpoint"] == "https://example.test"


def test_uses_official_endpoint_key_and_region_headers():
    session = FakeSession(
        [FakeResponse(200, [{"translations": [{"text": "你好"}]}])]
    )
    translator = make_translator(session)
    data = [SubtitleProcessData(index=1, original_text="Hello")]

    result = translator._translate_chunk(data)

    assert result[0].translated_text == "你好"
    url, request = session.calls[0]
    assert url == "https://api.cognitive.microsofttranslator.com/translate"
    assert request["params"]["api-version"] == "3.0"
    assert request["params"]["to"] == "zh-Hans"
    assert request["headers"]["Ocp-Apim-Subscription-Key"] == "azure-secret"
    assert request["headers"]["Ocp-Apim-Subscription-Region"] == "eastasia"
    assert "X-ClientTraceId" in request["headers"]
    translator.stop()
    assert session.closed is False


def test_custom_resource_endpoint_appends_translate_path():
    session = FakeSession(
        [FakeResponse(200, [{"translations": [{"text": "你好"}]}])]
    )
    translator = make_translator(
        session,
        endpoint="https://example.cognitiveservices.azure.com/translator/text/v3.0/",
    )

    assert translator.test_connection() == "你好"
    assert session.calls[0][0] == (
        "https://example.cognitiveservices.azure.com/translator/text/v3.0/translate"
    )
    translator.stop()


def test_retries_throttled_request_using_retry_after(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(429, {"error": {"message": "Rate limit"}}, {"Retry-After": "2"}),
            FakeResponse(200, [{"translations": [{"text": "你好"}]}]),
        ]
    )
    sleeps = []
    monkeypatch.setattr(bing_module.time, "sleep", sleeps.append)
    translator = make_translator(session, max_retries=2)

    assert translator.test_connection() == "你好"
    assert sleeps == [2.0]
    assert len(session.calls) == 2
    translator.stop()


def test_authentication_failure_is_not_silently_swallowed():
    session = FakeSession(
        [FakeResponse(401, {"error": {"message": "Invalid subscription key"}})]
    )
    translator = make_translator(session)
    data = [SubtitleProcessData(index=1, original_text="Hello")]

    with pytest.raises(RuntimeError, match="HTTP 401: Invalid subscription key"):
        translator._translate_chunk(data)

    assert data[0].translated_text == ""
    translator.stop()


def test_network_failure_is_retried_then_propagated(monkeypatch):
    session = FakeSession(
        [requests.ConnectionError("offline"), requests.ConnectionError("offline")]
    )
    monkeypatch.setattr(bing_module.time, "sleep", lambda _delay: None)
    translator = make_translator(session, max_retries=1)

    with pytest.raises(RuntimeError, match="network request failed after 2 attempts"):
        translator.test_connection()

    translator.stop()


def test_rejects_partial_batch_response():
    session = FakeSession(
        [FakeResponse(200, [{"translations": [{"text": "第一条"}]}])]
    )
    translator = make_translator(session)
    data = [
        SubtitleProcessData(index=1, original_text="First"),
        SubtitleProcessData(index=2, original_text="Second"),
    ]

    with pytest.raises(RuntimeError, match="different number of results"):
        translator._translate_chunk(data)

    translator.stop()
