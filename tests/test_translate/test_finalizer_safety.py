"""Regression coverage for semantic preservation and overlapping polish passes."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    value = LLMTranslator(
        thread_num=2,
        batch_num=20,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=True,
        update_callback=None,
        use_cache=False,
    )
    yield value
    value.stop()


@pytest.mark.parametrize(
    "target,translations",
    [
        (
            TargetLanguage.SIMPLIFIED_CHINESE,
            [
                "所以我认为这很重要",
                "我在报道中问过这个问题",
                "但我认为答案很简单",
                "促成友谊的并不是文字独有的特质",
            ],
        ),
        (
            TargetLanguage.FRENCH,
            [
                "Je pense donc que c'est important.",
                "J'ai pose cette question dans mon reportage.",
                "Mais je pense que la reponse etait simple.",
                "L'amitie ne dependait pas du texte.",
            ],
        ),
    ],
)
def test_finalizer_never_substitutes_canned_dialogue(translator, monkeypatch, target, translations):
    translator.target_language = target
    source = [
        SubtitleProcessData(index=i, original_text=text)
        for i, text in enumerate(
            [
                "So I think that this is important.",
                "I asked that exact question in my reporting.",
                "But I think that the answer was simple.",
                "It wasn't something specific to text that enabled friendship.",
            ],
            1,
        )
    ]
    translated = [replace(item, translated_text=text) for item, text in zip(source, translations)]
    translator._all_source_by_index = {item.index: item.original_text for item in source}
    translator._all_speaker_by_index = {1: "A", 2: "A", 3: "B", 4: "B"}
    monkeypatch.setattr(translator, "_validate_cross_key_boundaries", lambda *args: (True, ""))
    monkeypatch.setattr(translator, "_has_repetitive_dependent_boundary", lambda *args: False)
    monkeypatch.setattr(translator, "_strong_asr_semantic_candidates", lambda *args: [])
    monkeypatch.setattr(translator, "_strong_chinese_prose_candidates", lambda *args: [])
    audit = Mock()
    monkeypatch.setattr(translator, "_repair_chinese_boundary_fluency", audit)

    result = translator._finalize_translated_list(source, translated)

    assert [item.translated_text for item in result] == translations
    assert [item.original_text for item in result] == [item.original_text for item in source]
    audit.assert_called_once()  # The general semantic/fluency route remains active.


@pytest.mark.parametrize("reject_overlap", [False, True])
def test_overlapping_windows_use_latest_text_and_revalidate(
    translator, monkeypatch, reject_overlap
):
    source = [SubtitleProcessData(index=i, original_text=f"Source {i}.") for i in range(1, 9)]
    current = {
        item.index: replace(item, translated_text=f"Original {item.index}") for item in source
    }
    windows = [source[:6], source[5:]]
    seen = []

    def repair(window, snapshot):
        seen.append([item.translated_text for item in snapshot])
        label = "First" if len(seen) == 1 else "Second"
        return (
            window,
            [replace(item, translated_text=f"{label} {item.index}") for item in snapshot],
            None,
        )

    def validate(window, before, after):
        assert window == windows[0]
        assert before[-1].translated_text == "First 6"
        assert after[-1].translated_text == "Second 6"
        if reject_overlap:
            raise ValueError("Shared cue no longer completes its previous sentence")

    validation = Mock(side_effect=validate)
    monkeypatch.setattr(translator, "_repair_chinese_fluency_window_with_retries", repair)
    monkeypatch.setattr(translator, "_validate_chinese_fluency_repair", validation)
    result = translator._repair_chinese_fluency_group(windows, current)

    assert seen[1][0] == "First 6"
    validation.assert_called_once()
    assert result[6].translated_text == ("First 6" if reject_overlap else "Second 6")
    assert result[7].translated_text == ("Original 7" if reject_overlap else "Second 7")
    assert [result[i].translated_text for i in range(1, 6)] == [f"First {i}" for i in range(1, 6)]


def test_failed_or_invalid_repair_does_not_change_ownership(translator, monkeypatch):
    source = [SubtitleProcessData(index=i, original_text=f"Source {i}.") for i in range(1, 5)]
    current = {
        item.index: replace(item, translated_text=f"Original {item.index}") for item in source
    }
    original = dict(current)
    repair = Mock(
        side_effect=[
            (source[:2], None, ValueError("unavailable")),
            (source[2:], [replace(source[0], translated_text="Wrong cue")], None),
        ]
    )
    monkeypatch.setattr(translator, "_repair_chinese_fluency_window_with_retries", repair)
    assert translator._repair_chinese_fluency_group([source[:2], source[2:]], current) == original


def test_disjoint_repairs_do_not_trigger_extra_overlap_audits(translator, monkeypatch):
    source = [SubtitleProcessData(index=i, original_text=f"Source {i}.") for i in range(1, 5)]
    current = {
        item.index: replace(item, translated_text=f"Original {item.index}") for item in source
    }
    monkeypatch.setattr(
        translator,
        "_repair_chinese_fluency_window_with_retries",
        lambda window, snapshot: (
            window,
            [replace(item, translated_text=f"Repaired {item.index}") for item in snapshot],
            None,
        ),
    )
    validation = Mock()
    monkeypatch.setattr(translator, "_validate_chinese_fluency_repair", validation)
    result = translator._repair_chinese_fluency_group([source[:2], source[2:]], current)
    assert all(item.translated_text == f"Repaired {item.index}" for item in result.values())
    validation.assert_not_called()


@pytest.mark.parametrize("retry_fails", [True, False])
def test_adjacent_repair_retry_never_commits_a_rejected_rewrite(
    translator, monkeypatch, retry_fails
):
    source = [
        SubtitleProcessData(index=1, original_text="First sentence."),
        SubtitleProcessData(index=2, original_text="Second sentence."),
    ]
    previous = [replace(item, translated_text=f"保留译文{item.index}") for item in source]
    rejected = [replace(item, translated_text="不应写入的重复内容") for item in source]
    accepted = [replace(item, translated_text=f"通过校验的译文{item.index}") for item in source]
    verdict = Mock(side_effect=[
        (False, "Repeated endings: 1-2"),
        (False, "Repeated endings: 1-2"),
        (True, ""),
    ])
    retry = Mock(side_effect=[
        rejected,
        RuntimeError("provider unavailable") if retry_fails else accepted,
    ])
    monkeypatch.setattr(translator, "_validate_cross_key_boundaries", verdict)
    monkeypatch.setattr(translator, "_translate_locked_batch", retry)
    monkeypatch.setattr(translator, "_has_repetitive_dependent_boundary", lambda *args: False)
    monkeypatch.setattr(translator, "_strong_asr_semantic_candidates", lambda *args: [])
    monkeypatch.setattr(translator, "_strong_chinese_prose_candidates", lambda *args: [])
    monkeypatch.setattr(translator, "_repair_chinese_boundary_fluency", Mock())
    result = translator._finalize_translated_list(source, previous)

    assert result == (previous if retry_fails else accepted)
    assert [item.translated_text for item in previous] == ["保留译文1", "保留译文2"]
    assert retry.call_count == 2
