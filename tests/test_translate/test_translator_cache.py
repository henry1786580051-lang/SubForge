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
    def __init__(
        self,
        fake_cache: FakeCache,
        use_cache: bool,
        cache_namespace: str = "",
    ):
        self.fake_cache = fake_cache
        super().__init__(
            thread_num=1,
            batch_num=10,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            update_callback=None,
            use_cache=use_cache,
            cache_namespace=cache_namespace,
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
            provisional=[
                SubtitleProcessData(
                    index=subtitle_chunk[-1].index,
                    original_text=subtitle_chunk[-1].original_text,
                    translated_text="最后一次可恢复的候选译文",
                )
            ],
        )


class FinalizingPartiallyFailingTranslator(PartiallyFailingTranslator):
    def __init__(self, fake_cache: FakeCache, use_cache: bool):
        super().__init__(fake_cache, use_cache)
        self.finalize_calls = 0

    def _finalize_translated_list(self, source_list, translated_list):
        self.finalize_calls += 1
        assert [item.index for item in source_list] == list(range(1, 11))
        for item in translated_list:
            if item.index == 10:
                item.translated_text = "收尾审计后的恢复译文"
        return translated_list


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


def test_candidate_translator_cache_cannot_read_legacy_entries(monkeypatch):
    fake_cache = FakeCache()
    legacy_key = "DummyTranslator:any:简体中文"
    candidate_key = f"translation-quality:candidate:phase8-r1:{legacy_key}"
    fake_cache.values[legacy_key] = [
        SubtitleProcessData(index=1, original_text="Hello", translated_text="旧译文")
    ]
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: True)

    translator = DummyTranslator(
        fake_cache,
        use_cache=True,
        cache_namespace="translation-quality:candidate:phase8-r1",
    )
    translator._get_cache_key = lambda _chunk: legacy_key

    result = translator._safe_translate_chunk(
        [SubtitleProcessData(index=1, original_text="Hello")]
    )

    assert result[0].translated_text == "新译文:Hello"
    assert candidate_key in fake_cache.values
    assert fake_cache.values[legacy_key][0].translated_text == "旧译文"


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

    assert [item.index for item in progress] == list(range(1, 11))
    assert progress[-1].translated_text == "最后一次可恢复的候选译文"


def test_complete_recovery_becomes_success_after_finalization_and_validation(
    monkeypatch,
):
    fake_cache = FakeCache()
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: False)
    progress = []
    translator = FinalizingPartiallyFailingTranslator(fake_cache, use_cache=False)
    translator.update_callback = lambda items: progress.append(list(items))
    asr_data = ASRData(
        [ASRDataSeg(f"source {index}", index * 1000, (index + 1) * 1000) for index in range(10)]
    )

    result = translator.translate_subtitle(asr_data)

    assert translator.finalize_calls == 1
    assert [item.index for item in progress[-1]] == list(range(1, 11))
    assert progress[-1][-1].translated_text == "收尾审计后的恢复译文"
    assert result.segments[-1].translated_text == "收尾审计后的恢复译文"


@pytest.mark.parametrize("failure", ["request", "validation"])
def test_failed_recovery_finalization_runs_once_and_preserves_the_checkpoint(
    monkeypatch, failure
):
    fake_cache = FakeCache()
    monkeypatch.setattr(translate_base, "get_translate_cache", lambda: fake_cache)
    monkeypatch.setattr(translate_base, "is_cache_enabled", lambda: False)
    translator = PartiallyFailingTranslator(fake_cache, use_cache=False)
    progress = []
    calls = []
    translator.update_callback = lambda items: progress.append(
        [(item.index, item.translated_text) for item in items]
    )
    data = ASRData([
        ASRDataSeg(f"source {index}", index * 1000, (index + 1) * 1000)
        for index in range(10)
    ])

    def reject_finalization(source, translated):
        calls.append([item.index for item in source])
        translated[0].translated_text = ""
        if failure == "request":
            raise RuntimeError("provider unavailable after an intermediate rewrite")
        return translated

    monkeypatch.setattr(translator, "_finalize_translated_list", reject_finalization)
    with pytest.raises(RuntimeError, match="Single item translation failed"):
        translator.translate_subtitle(data)

    assert len(calls) == 1
    assert progress[-1] == [
        *[(index, f"译文{index}") for index in range(1, 10)],
        (10, "最后一次可恢复的候选译文"),
    ]
    assert all(not segment.translated_text for segment in data.segments)


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
