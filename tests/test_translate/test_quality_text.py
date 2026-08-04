import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.quality import (
    inspect_translation_batch,
    is_placeholder_translation,
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
    assert is_untranslated_output(
        "This is still English",
        "This is still English",
        TargetLanguage.SIMPLIFIED_CHINESE,
    )


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
