import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate import base as translate_base
from subforge.core.translate.base import BaseTranslator
from subforge.core.translate.types import TargetLanguage


class FakeCache:
    def __init__(self):
        self.get_calls = 0
        self.set_calls = 0
        self.delete_calls = 0
        self.values = {}

    def get(self, key, default=None):
        self.get_calls += 1
        return self.values.get(key, default)

    def set(self, key, value, expire=None):
        self.set_calls += 1
        self.values[key] = value

    def delete(self, key):
        self.delete_calls += 1
        self.values.pop(key, None)


class DummyTranslator(BaseTranslator):
    def __init__(self, fake_cache: FakeCache, use_cache: bool):
        self.fake_cache = fake_cache
        super().__init__(
            thread_num=1,
            batch_num=10,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            update_callback=None,
            use_cache=use_cache,
        )

    def _translate_chunk(self, subtitle_chunk):
        for item in subtitle_chunk:
            item.translated_text = f"fresh:{item.original_text}"
        return subtitle_chunk


class FailingTranslator(DummyTranslator):
    def _translate_chunk(self, subtitle_chunk):
        raise RuntimeError("temporary service failure")


class EmptyTranslator(DummyTranslator):
    def _translate_chunk(self, subtitle_chunk):
        return subtitle_chunk


def test_translator_does_not_read_or_write_cache_when_disabled(monkeypatch):
    fake_cache = FakeCache()
    fake_cache.values["DummyTranslator:any:简体中文"] = [
        SubtitleProcessData(index=1, original_text="Hello", translated_text="old")
    ]
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: True)

    translator = DummyTranslator(fake_cache, use_cache=False)
    translator._get_cache_key = lambda _chunk: "DummyTranslator:any:简体中文"

    result = translator._safe_translate_chunk(
        [SubtitleProcessData(index=1, original_text="Hello")]
    )

    assert result[0].translated_text == "fresh:Hello"
    assert fake_cache.get_calls == 0
    assert fake_cache.set_calls == 0


def test_translator_uses_cache_when_enabled(monkeypatch):
    fake_cache = FakeCache()
    fake_cache.values["DummyTranslator:any:简体中文"] = [
        SubtitleProcessData(index=1, original_text="Hello", translated_text="old")
    ]
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: True)

    translator = DummyTranslator(fake_cache, use_cache=True)
    translator._get_cache_key = lambda _chunk: "DummyTranslator:any:简体中文"

    result = translator._safe_translate_chunk(
        [SubtitleProcessData(index=1, original_text="Hello")]
    )

    assert result[0].translated_text == "old"
    assert fake_cache.get_calls == 1
    assert fake_cache.set_calls == 0


def test_translate_subtitle_rejects_failed_chunks(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: False)
    translator = FailingTranslator(fake_cache, use_cache=False)
    asr_data = ASRData([ASRDataSeg("Hello", 0, 1000)])

    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate_subtitle(asr_data)


def test_translate_subtitle_rejects_empty_translations(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: False)
    translator = EmptyTranslator(fake_cache, use_cache=False)
    asr_data = ASRData([ASRDataSeg("Hello", 0, 1000)])

    with pytest.raises(RuntimeError, match="Translation incomplete"):
        translator.translate_subtitle(asr_data)
