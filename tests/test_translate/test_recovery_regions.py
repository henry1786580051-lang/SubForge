from dataclasses import replace

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.llm.client import LLMRequestCancelled
from subforge.core.translate.base import BaseTranslator, PartialTranslationError
from subforge.core.translate.types import TargetLanguage


class IslandTranslator(BaseTranslator):
    def __init__(self, failure=""):
        super().__init__(1, 5, TargetLanguage.SIMPLIFIED_CHINESE, None, use_cache=False)
        self.failure, self.regions, self.published = failure, [], []
        self.update_callback = lambda items: self.published.append(
            [(s.index, s.translated_text) for s in items]
        )

    def _translate_chunk(self, chunk):
        raise PartialTranslationError(
            "missing key 3",
            [replace(s, translated_text=f"原译文{s.index}") for s in chunk if s.index != 3],
            [3],
        )

    def _finalize_translated_list(self, source, translated):
        self.regions.append([s.index for s in source])
        if source[0].index == 4:
            if self.failure == "cancel":
                raise LLMRequestCancelled("cancelled")
            if self.failure == "request":
                translated[0].translated_text = "不要泄漏中间修改"
                raise RuntimeError("offline")
            if self.failure == "ownership":
                return [replace(translated[0], index=1)]
        return [replace(s, translated_text=f"复核后{s.index}") for s in translated]


@pytest.mark.parametrize("failure", ["", "request", "ownership"])
def test_missing_key_does_not_skip_other_islands_or_become_success(failure):
    t = IslandTranslator(failure)
    data = ASRData([ASRDataSeg(f"source {i}", i * 1000, (i + 1) * 1000) for i in range(1, 6)])
    with pytest.raises(RuntimeError, match="missing key 3"):
        t.translate_subtitle(data)
    assert t.regions == [[1, 2], [4, 5]]
    assert t.recovery_review["missing_indices"] == [3]
    assert t.recovery_review["reviewed_indices"] == ([1, 2] if failure else [1, 2, 4, 5])
    assert t.recovery_review["unchecked_indices"] == ([4, 5] if failure else [])
    assert t.published[-1] == [
        (1, "复核后1"),
        (2, "复核后2"),
        (4, "原译文4" if failure else "复核后4"),
        (5, "原译文5" if failure else "复核后5"),
    ]
    assert not any(s.translated_text for s in data.segments)


def test_cancellation_is_not_swallowed_as_recovery_success():
    t = IslandTranslator("cancel")
    data = ASRData([ASRDataSeg(f"source {i}", i * 1000, (i + 1) * 1000) for i in range(1, 6)])
    with pytest.raises(LLMRequestCancelled):
        t.translate_subtitle(data)
    assert t.published == [[(1, "原译文1"), (2, "原译文2"), (4, "原译文4"), (5, "原译文5")]]


def test_provider_failure_checkpoints_completed_items_without_new_review_calls(monkeypatch):
    t = IslandTranslator()
    t.batch_num = 2

    def translate(chunk):
        if chunk[0].index > 2:
            raise RuntimeError("provider unavailable")
        return [replace(s, translated_text=f"原译文{s.index}") for s in chunk]

    monkeypatch.setattr(t, "_translate_chunk", translate)
    data = ASRData([ASRDataSeg(f"source {i}", i * 1000, (i + 1) * 1000) for i in range(1, 5)])
    with pytest.raises(RuntimeError, match="provider unavailable"):
        t.translate_subtitle(data)
    assert t.regions == []
    assert t.recovery_review == {
        "status": "partial",
        "missing_indices": [3, 4],
        "reviewed_indices": [],
        "unchecked_indices": [1, 2],
    }


@pytest.mark.parametrize(
    "error", [RuntimeError("finalizer unavailable"), ValueError("invalid finalizer output")]
)
def test_failed_finalizer_retains_snapshot_and_marks_all_unchecked(monkeypatch, error):
    t = IslandTranslator()
    monkeypatch.setattr(
        t,
        "_translate_chunk",
        lambda chunk: [replace(s, translated_text=f"原译文{s.index}") for s in chunk],
    )
    calls = []

    def finalize(source, translated):
        calls.append(True)
        translated[0].translated_text = "不应写入的中间修改"
        raise error

    monkeypatch.setattr(t, "_finalize_translated_list", finalize)
    data = ASRData([ASRDataSeg("source", 0, 1000)])
    with pytest.raises(RuntimeError):
        t.translate_subtitle(data)
    assert len(calls) == 1
    assert t.published[-1] == [(1, "原译文1")]
    assert t.recovery_review == {
        "status": "partial",
        "missing_indices": [],
        "reviewed_indices": [],
        "unchecked_indices": [1],
    }
