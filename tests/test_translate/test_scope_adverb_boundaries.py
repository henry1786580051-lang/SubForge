"""Regression for a focus marker left behind by source-aligned translation."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import (
    ChineseBoundaryFeatures,
    detect_discourse_bridge_boundary,
)
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    value = LLMTranslator(
        thread_num=1,
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


@pytest.fixture
def glc_window(translator):
    source = [
        SubtitleProcessData(
            index=124,
            original_text="We've got some Alcantara in the seams, some nice detailing, and even",
        ),
        SubtitleProcessData(
            index=125,
            original_text="top of the door panels here, finished in leather, some decent red stitching.",
        ),
    ]
    current = [
        replace(source[0], translated_text="缝线处用了Alcantara材质 细节也很精致 甚至"),
        replace(source[1], translated_text="车门板上沿包了皮革 配着不错的红色缝线"),
    ]
    translator._all_source_by_index = {item.index: item.original_text for item in source}
    translator._gap_after_index = {124: 301}
    return source, current


@pytest.mark.parametrize("marker", ["甚至", "甚至是", "乃至", "尤其", "尤其是", "特别是", "特別是"])
@pytest.mark.parametrize("punctuation", ["", "。", "，"])
def test_scope_marker_requires_its_following_complement(marker, punctuation):
    left = f"做工很精致 {marker}{punctuation}"
    signal = LLMTranslator._chinese_boundary_signal(left, "车门板也包了皮革")
    assert signal == "scope adverb is separated from its complement"


def test_standalone_marker_is_detected_but_not_emptied_by_relocation(glc_window):
    source, current = glc_window
    current[0] = replace(current[0], translated_text="甚至")
    assert LLMTranslator._chinese_boundary_signal("甚至", "车门板也包了皮革") == (
        "scope adverb is separated from its complement"
    )
    assert LLMTranslator._deterministic_chinese_fluency_fallback(source, current) is None


@pytest.mark.parametrize(
    "left,right",
    [
        ("做工很精致 甚至车门板也包了皮革", "缝线是红色的"),
        ("这个设计很特别", "我很喜欢"),
        ("我说的是‘甚至’", "它是个副词"),
        ('这个词叫"尤其是"', "它表示强调"),
        ("做工很精致 甚至", ""),
    ],
)
def test_complete_sentences_and_quoted_word_mentions_are_not_scope_breaks(left, right):
    match = detect_discourse_bridge_boundary(ChineseBoundaryFeatures.from_text(left, right))
    assert match is None or match.message != "scope adverb is separated from its complement"


@pytest.mark.parametrize("gap", [80, 301, 4700])
def test_actual_glc_break_is_mandatory_even_when_auditor_misses_it(translator, glc_window, gap):
    source, current = glc_window
    translator._gap_after_index = {124: gap}
    assert translator._mandatory_chinese_fluency_candidates(
        source, {item.index: item for item in current}
    ) == [124]


def test_adding_punctuation_does_not_hide_the_break(translator, glc_window):
    source, current = glc_window
    source[0] = replace(source[0], original_text=source[0].original_text + ".")
    current[0] = replace(current[0], translated_text=current[0].translated_text + "。")
    assert translator._mandatory_chinese_fluency_candidates(
        source, {item.index: item for item in current}
    ) == [124]


def test_unchanged_or_punctuated_rewrite_cannot_pass_fidelity_audit(
    translator, glc_window, monkeypatch
):
    source, current = glc_window
    monkeypatch.setattr(translator, "_validate_llm_response", lambda *a, **kw: (True, ""))
    audit = Mock()
    monkeypatch.setattr(translator, "_validate_chinese_window_fidelity", audit)
    with pytest.raises(ValueError, match="structural boundary signals"):
        translator._validate_chinese_fluency_repair(source, current, current)
    audit.assert_not_called()


def test_safe_relocation_preserves_all_words_and_is_idempotent(glc_window):
    source, current = glc_window
    repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)
    assert repaired is not None
    assert repaired[0].translated_text == "缝线处用了Alcantara材质 细节也很精致"
    assert repaired[1].translated_text == "甚至车门板上沿包了皮革 配着不错的红色缝线"
    assert "".join(x.translated_text for x in current).replace(" ", "") == "".join(
        x.translated_text for x in repaired
    ).replace(" ", "")
    assert LLMTranslator._deterministic_chinese_fluency_fallback(source, repaired) is None


def test_relocation_does_not_duplicate_existing_marker(glc_window):
    source, current = glc_window
    current[1] = replace(current[1], translated_text="甚至" + current[1].translated_text)
    repaired = LLMTranslator._deterministic_chinese_fluency_fallback(source, current)
    assert repaired is not None
    assert sum(item.translated_text.count("甚至") for item in repaired) == 1


@pytest.mark.parametrize("protection", ["pause", "speaker", "language", "nonadjacent"])
def test_relocation_respects_existing_boundary_protections(
    translator, glc_window, monkeypatch, protection
):
    source, current = glc_window
    translator._gap_after_index = {124: 80}
    if protection == "pause":
        translator._gap_after_index = {124: 301}
    elif protection == "speaker":
        translator._all_speaker_by_index = {124: "A", 125: "B"}
    elif protection == "language":
        source = [replace(source[0], source_language="en"), replace(source[1], source_language="de")]
    else:
        source[1] = replace(source[1], index=126)
        current[1] = replace(current[1], index=126)
    rewrite = Mock(side_effect=ValueError("No model in unit tests"))
    monkeypatch.setattr(translator, "_should_reason_about_chinese_fluency_window", lambda *a, **k: False)
    monkeypatch.setattr(translator, "_rewrite_chinese_fluency_window", rewrite)
    monkeypatch.setattr(translator, "_rewrite_chinese_fluency_window_fresh", rewrite)
    monkeypatch.setattr(
        translator, "_validate_chinese_fluency_repair", Mock(side_effect=ValueError("still broken"))
    )
    _, repaired, _ = translator._repair_chinese_fluency_window_with_retries(source, current)
    assert repaired is None
    assert rewrite.called


def test_flat_surface_even_is_not_source_support_for_relocation(glc_window):
    source, current = glc_window
    source[0] = replace(source[0], original_text="The surface is even")
    assert LLMTranslator._deterministic_chinese_fluency_fallback(source, current) is None


def test_unchanged_window_is_not_logged_as_corrected(translator, glc_window, monkeypatch):
    source, current = glc_window
    monkeypatch.setattr(
        translator, "_repair_chinese_fluency_window_with_retries",
        lambda window, snapshot: (window, snapshot, None),
    )
    info = Mock()
    monkeypatch.setattr("subforge.core.translate.llm_translator.logger.info", info)
    before = {item.index: item for item in current}
    assert translator._repair_chinese_fluency_group([source], dict(before)) == before
    info.assert_not_called()


def test_unchanged_valid_window_still_protects_overlapping_repair(translator, monkeypatch):
    source = [SubtitleProcessData(index=i, original_text=f"Source {i}.") for i in range(1, 4)]
    current = {item.index: replace(item, translated_text=f"Original {item.index}") for item in source}
    rewrite = Mock(side_effect=[
        (source[:2], list(current.values())[:2], None),
        (source[1:], [replace(item, translated_text="Changed") for item in source[1:]], None),
    ])
    monkeypatch.setattr(translator, "_repair_chinese_fluency_window_with_retries", rewrite)
    validate = Mock(side_effect=ValueError("breaks previously accepted boundary"))
    monkeypatch.setattr(translator, "_validate_chinese_fluency_repair", validate)
    assert translator._repair_chinese_fluency_group([source[:2], source[1:]], dict(current)) == current
    validate.assert_called_once()
