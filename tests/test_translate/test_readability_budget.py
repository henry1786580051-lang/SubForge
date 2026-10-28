from dataclasses import replace
from types import SimpleNamespace

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


def translator(limit=16, language=TargetLanguage.SIMPLIFIED_CHINESE):
    return LLMTranslator(
        1,
        10,
        language,
        "offline",
        "Keep names.",
        False,
        None,
        use_cache=False,
        max_word_count_cjk=limit,
    )


def test_budget_is_in_actual_translation_prompt_without_overriding_fidelity(monkeypatch):
    instance = translator()
    captured = []

    def loop(prompt, source):
        captured.append(prompt)
        return {"1": "2026款雷克萨斯ES350H"}

    monkeypatch.setattr(instance, "_agent_loop", loop)
    monkeypatch.setattr(instance, "_needs_alignment_audit", lambda: False)
    result = instance._translate_chunk([SubtitleProcessData(1, "2026 Lexus ES350H")])
    assert "at most 16" in captured[0]
    assert "Never shorten a model identifier" in captured[0]
    assert "Keep names." in captured[0]
    assert result[0].translated_text == "2026款雷克萨斯ES350H"


def test_budget_is_language_scoped_and_changes_cache_identity():
    assert not translator(language=TargetLanguage.ENGLISH)._readability_requirements()
    assert not translator(limit=None)._readability_requirements()
    chunk = [SubtitleProcessData(1, "The Lexus ES350H")]
    assert translator(16)._get_cache_key(chunk) != translator(25)._get_cache_key(chunk)


def test_reading_budget_counts_identifiers_without_counting_punctuation():
    from subforge.core.utils.text_utils import count_reading_units

    assert count_reading_units("全新第八代，") == 5
    assert count_reading_units("本周体验2026款雷克萨斯 ES 350h Premium 全驱版，") == 16
    assert count_reading_units("2.0 F-150 / 2026，") == 3
    assert count_reading_units("……！？") == 0


def test_actual_compaction_prompt_receives_counted_overflow_and_exact_keys(monkeypatch):
    import json

    import subforge.core.translate.llm_translator as module

    instance = translator()
    instance.translation_context = replace(
        instance.translation_context, terminology="living with -> 长期体验 (non-literal)"
    )
    items = opening_items()
    captured = []
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: captured.append(kwargs) or object())
    monkeypatch.setattr(
        module,
        "get_response_text",
        lambda _: '{"2":"本周体验2026款雷克萨斯 ES350H Premium 全驱版"}',
    )
    result = instance._rewrite_presentation_readability(items, items, {2})
    assert len(captured) == 1 and captured[0]["reasoning_mode"] == "disabled"
    assert "at most 16" in captured[0]["messages"][0]["content"]
    payload = json.loads(captured[0]["messages"][1]["content"])
    assert set(payload) == {"2"} and payload["2"]["current_units"] > 16
    assert payload["2"]["required_terms"] == ["长期体验"]
    assert result[1].translated_text.startswith("本周体验2026")
    assert result[0] == items[0] and result[2:] == items[2:]


def test_actual_fidelity_gate_rejects_lost_confirmed_term(monkeypatch):
    instance = translator()
    instance._presentation_indices = (1, 2, 3, 4)
    instance.translation_context = replace(
        instance.translation_context, terminology="living with -> 长期试驾 (non-literal)"
    )
    items = opening_items()
    candidate = [
        replace(s, translated_text="本周体验2026款雷克萨斯ES350H Premium全驱版")
        if s.index == 2
        else s
        for s in items
    ]
    monkeypatch.setattr(instance, "_rewrite_presentation_readability", lambda *a: candidate)
    assert instance._compact_opening_presentation(items, items) is items


def test_unavoidable_overflow_retains_facts_and_reports_review(monkeypatch):
    from subforge.core.asr.asr_data import ASRData, ASRDataSeg
    from subforge.core.translate.base import BaseTranslator

    data = ASRData(
        [
            ASRDataSeg(
                "2026 Lexus ES350H Premium All-Wheel Drive",
                0,
                6000,
                translated_text="2026款雷克萨斯ES350H Premium全轮驱动版",
            )
        ]
    )
    original = data.segments[0].translated_text
    monkeypatch.setattr(BaseTranslator, "translate_subtitle", lambda self, supplied: supplied)
    instance = translator(limit=5)
    result = instance.translate_subtitle(data)
    assert result.segments[0].translated_text == original
    assert instance.readability_issues[0]["limit"] == 5
    assert instance.readability_issues[0]["units"] > 5


def test_factory_forwards_budget_only_to_llm(monkeypatch):
    from subforge.core.translate import factory
    from subforge.core.translate.types import TranslatorType

    captured = []
    monkeypatch.setattr(
        factory, "LLMTranslator", lambda **kw: captured.append(kw) or SimpleNamespace()
    )
    factory.TranslatorFactory.create_translator(TranslatorType.OPENAI, max_word_count_cjk=16)
    assert captured[0]["max_word_count_cjk"] == 16


@pytest.mark.parametrize("ending", [".", "...", "!", "?"])
def test_sentence_final_year_is_owned_by_every_source_key_that_speaks_it(ending):
    instance = translator()
    source = {
        "2": "The 2026 Lexus ES350H Premium All-Wheel Drive,",
        "4": f"and current Lexus flagship sedan in the lineup in the year 2026{ending}",
    }
    response = {"2": "2026款雷克萨斯ES350H Premium全驱版", "4": "也是雷克萨斯2026年的旗舰轿车"}
    assert instance._validate_cross_key_boundaries(response, source, str) == (True, "")
    assert "2026" in instance._boundary_tokens(source["4"])


def test_year_fix_still_rejects_borrowing_and_does_not_split_decimal():
    instance = translator()
    valid, error = instance._validate_cross_key_boundaries(
        {"2": "2026款雷克萨斯ES350H", "4": "也是雷克萨斯2026年的旗舰轿车"},
        {"2": "The 2026 Lexus ES350H,", "4": "and its current flagship sedan."},
        str,
    )
    assert not valid and "Cross-key duplicates" in error
    assert not instance._boundary_tokens("The length is 15.60.")


def opening_items():
    return [
        SubtitleProcessData(
            1, "Hey everyone, welcome back to Topher Drives,", "大家好 欢迎回到Topher Drives"
        ),
        SubtitleProcessData(
            2,
            "where this week I've been living with the 2026 Lexus ES350H Premium All-Wheel Drive,",
            "这周我一直在体验2026款雷克萨斯ES350H Premium全轮驱动版",
        ),
        SubtitleProcessData(3, "the new 8th generation", "全新第八代"),
        SubtitleProcessData(
            4,
            "and current Lexus flagship sedan in the lineup in the year 2026.",
            "也是雷克萨斯2026年的旗舰轿车",
        ),
    ]


def test_compact_opening_is_bounded_and_validated_before_writeback(monkeypatch):
    instance = translator()
    instance._presentation_indices = (1, 2, 3, 4)
    items = opening_items()
    captured = []
    validation = []

    def rewrite(source, current, overlong):
        captured.append(overlong)
        return [
            replace(s, translated_text="本周体验2026款雷克萨斯ES350H Premium全驱版")
            if s.index == 2
            else s
            for s in current
        ]

    monkeypatch.setattr(instance, "_rewrite_presentation_readability", rewrite)
    monkeypatch.setattr(
        instance, "_validate_chinese_fluency_repair", lambda *args: validation.append(args)
    )
    result = instance._compact_opening_presentation(items, items)
    assert len(captured) == 1 and len(validation) == 1
    assert captured == [{2}]
    assert result[0] == items[0] and result[2:] == items[2:]
    assert result[1].translated_text == "本周体验2026款雷克萨斯ES350H Premium全驱版"
    assert [s.original_text for s in result] == [s.original_text for s in items]
    assert items[1].translated_text.startswith("这周我一直")


@pytest.mark.parametrize("failure", ["keys", "other_cue", "not_shorter", "fidelity", "cancel"])
def test_rejected_compact_rewrite_retains_validated_text(monkeypatch, failure):
    from subforge.core.llm.client import LLMRequestCancelled

    instance = translator()
    instance._presentation_indices = (1, 2, 3, 4)
    items = opening_items()
    candidate = [
        replace(s, translated_text="本周体验2026款雷克萨斯ES350H Premium全驱版")
        if s.index == 2
        else s
        for s in items
    ]
    if failure == "keys":
        candidate.pop()
    if failure == "other_cue":
        candidate[0] = replace(candidate[0], translated_text="大家好")
    if failure == "not_shorter":
        candidate = items
    calls = []

    def rewrite(*args, **kwargs):
        calls.append(1)
        if failure == "cancel":
            raise LLMRequestCancelled("cancelled")
        return candidate

    def validate(*args):
        if failure == "fidelity":
            raise ValueError("missing Premium trim")

    monkeypatch.setattr(instance, "_rewrite_presentation_readability", rewrite)
    monkeypatch.setattr(instance, "_validate_chinese_fluency_repair", validate)
    if failure == "cancel":
        with pytest.raises(LLMRequestCancelled):
            instance._compact_opening_presentation(items, items)
    else:
        assert instance._compact_opening_presentation(items, items) is items
    assert len(calls) == 1
