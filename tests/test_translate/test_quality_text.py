import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.quality import (
    contains_reasoning_leak,
    inspect_translation_batch,
    is_placeholder_translation,
    is_source_copy,
    is_untranslated_output,
)
from subforge.core.translate.types import TargetLanguage


@pytest.mark.parametrize(
    "text",
    ["（此句合并至上一句）", "内容同上", "merged into the previous", "omitted"],
)
def test_placeholder_detection_is_provider_independent(text):
    assert is_placeholder_translation(text)


def test_target_script_detection_preserves_model_only_caption():
    assert not is_untranslated_output(
        "BMW M3",
        "BMW M3",
        TargetLanguage.SIMPLIFIED_CHINESE,
    )


def test_chinese_target_rejects_unchanged_japanese_source():
    assert is_untranslated_output(
        "これは伝統的な工法です",
        "これは伝統的な工法です",
        TargetLanguage.SIMPLIFIED_CHINESE,
        "ja",
    )
    assert not is_untranslated_output(
        "这是一种传统工法",
        "これは伝統的な工法です",
        TargetLanguage.SIMPLIFIED_CHINESE,
        "ja",
    )
    assert is_untranslated_output(
        "This is still English",
        "This is still English",
        TargetLanguage.SIMPLIFIED_CHINESE,
    )


@pytest.mark.parametrize(
    "source,output",
    [
        ("Talkiatry.com slash grayarea.", "Talkiatry.com/grayarea"),
        ("patreon dot com slash vox", "patreon.com/vox"),
    ],
)
def test_target_script_detection_allows_standalone_url(source, output):
    assert not is_untranslated_output(
        output,
        source,
        TargetLanguage.SIMPLIFIED_CHINESE,
    )


def test_target_script_detection_still_translates_url_call_to_action():
    assert is_untranslated_output(
        "Visit example.com",
        "Visit example.com",
        TargetLanguage.SIMPLIFIED_CHINESE,
    )


@pytest.mark.parametrize(
    "text",
    [
        "<think>private chain</think>最终译文",
        "```json\n{\"1\": \"译文\"}\n```",
        "Reasoning: first compare the clauses",
        "作为一名 AI 我无法翻译",
    ],
)
def test_reasoning_leak_detection_uses_high_confidence_markers(text):
    assert contains_reasoning_leak(text)


@pytest.mark.parametrize("text", ["以下是结果", "分析结果如下", "这是一段正常译文"])
def test_reasoning_leak_detection_does_not_reject_normal_narration(text):
    assert not contains_reasoning_leak(text)


def test_source_copy_detection_matches_substantial_normalized_text():
    assert is_source_copy("The same source sentence.", "The same source sentence!")
    assert not is_source_copy("BMW", "BMW")
    assert not is_source_copy("这是译文", "This is the source")


def test_batch_report_collects_every_completeness_failure():
    source = [
        SubtitleProcessData(1, "One"),
        SubtitleProcessData(2, "Two"),
        SubtitleProcessData(3, "Three"),
        SubtitleProcessData(4, "Four"),
    ]
    translated = [
        SubtitleProcessData(1, "One", "一"),
        SubtitleProcessData(1, "One", "一"),
        SubtitleProcessData(2, "Two", ""),
        SubtitleProcessData(3, "Three", "（此句合并至上一句）"),
    ]

    report = inspect_translation_batch(
        source,
        translated,
        TargetLanguage.SIMPLIFIED_CHINESE,
    )

    assert report.duplicates == ["1"]
    assert report.empty == ["2"]
    assert report.placeholders == ["3"]
    assert report.missing == ["4"]
    assert not report.valid


def test_batch_report_rejects_reasoning_residue():
    source = [SubtitleProcessData(1, "One")]
    translated = [SubtitleProcessData(1, "One", "<think>分析</think>第一条")]

    report = inspect_translation_batch(
        source,
        translated,
        TargetLanguage.SIMPLIFIED_CHINESE,
    )

    assert report.reasoning_leaks == ["1"]
    assert "reasoning leaks: ['1']" in report.error_detail()
