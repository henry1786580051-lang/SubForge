import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate import base as translate_base
from subforge.core.translate.base import BaseTranslator, PartialTranslationError
from subforge.core.translate.types import TargetLanguage, get_language_code


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
            item.translated_text = f"新译文:{item.original_text}"
        return subtitle_chunk


class FailingTranslator(DummyTranslator):
    def _translate_chunk(self, subtitle_chunk):
        raise RuntimeError("temporary service failure")


class EmptyTranslator(DummyTranslator):
    def _translate_chunk(self, subtitle_chunk):
        return subtitle_chunk


class PartiallyFailingTranslator(DummyTranslator):
    def _translate_chunk(self, subtitle_chunk):
        completed = [
            SubtitleProcessData(
                index=item.index,
                original_text=item.original_text,
                translated_text=f"译文{item.index}",
            )
            for item in subtitle_chunk[:-1]
        ]
        raise PartialTranslationError(
            "Single item translation failed for one entry",
            completed=completed,
            failed_indices=[subtitle_chunk[-1].index],
        )


def test_translator_does_not_read_or_write_cache_when_disabled(monkeypatch):
    fake_cache = FakeCache()
    fake_cache.values["DummyTranslator:any:简体中文"] = [
        SubtitleProcessData(index=1, original_text="Hello", translated_text="旧译文")
    ]
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: True)

    translator = DummyTranslator(fake_cache, use_cache=False)
    translator._get_cache_key = lambda _chunk: "DummyTranslator:any:简体中文"

    result = translator._safe_translate_chunk(
        [SubtitleProcessData(index=1, original_text="Hello")]
    )

    assert result[0].translated_text == "新译文:Hello"
    assert fake_cache.get_calls == 0
    assert fake_cache.set_calls == 0


def test_translator_uses_cache_when_enabled(monkeypatch):
    fake_cache = FakeCache()
    fake_cache.values["DummyTranslator:any:简体中文"] = [
        SubtitleProcessData(index=1, original_text="Hello", translated_text="旧译文")
    ]
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: True)

    translator = DummyTranslator(fake_cache, use_cache=True)
    translator._get_cache_key = lambda _chunk: "DummyTranslator:any:简体中文"

    result = translator._safe_translate_chunk(
        [SubtitleProcessData(index=1, original_text="Hello")]
    )

    assert result[0].translated_text == "旧译文"
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


def test_partial_chunk_failure_preserves_completed_items_and_counts_exact_failure(
    monkeypatch,
):
    fake_cache = FakeCache()
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: False)
    progress = []
    translator = PartiallyFailingTranslator(fake_cache, use_cache=False)
    translator.update_callback = progress.extend
    chunk = [
        SubtitleProcessData(index=index, original_text=f"source {index}")
        for index in range(1, 11)
    ]

    with pytest.raises(
        RuntimeError,
        match=r"1/10 segments failed.*provider responded.*quality validation",
    ):
        translator._parallel_translate([chunk])

    assert [item.index for item in progress] == list(range(1, 10))


@pytest.mark.parametrize(
    "target_language",
    [
        TargetLanguage.TRADITIONAL_CHINESE,
        TargetLanguage.KOREAN,
        TargetLanguage.CANTONESE,
    ],
)
def test_asian_target_languages_reject_untranslated_english(target_language):
    translator = object.__new__(DummyTranslator)
    translator.target_language = target_language

    assert translator._is_untranslated_output(
        "This output was not translated",
        "This output was not translated",
    )


def test_unsupported_provider_language_does_not_fallback_to_chinese():
    with pytest.raises(ValueError, match="does not support"):
        get_language_code(TargetLanguage.CANTONESE, "deeplx")
