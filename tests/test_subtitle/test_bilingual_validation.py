import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.subtitle.validation import (
    lock_source_segments,
    validate_bilingual_result,
)


def _data(translation: str = "这是一条自然译文") -> ASRData:
    return ASRData(
        [
            ASRDataSeg(
                text="This is a source subtitle with a dependent clause.",
                start_time=1000,
                end_time=4000,
                translated_text=translation,
            )
        ]
    )


def test_final_validation_allows_natural_non_literal_translation():
    data = _data("这句话采用了自然意译")
    source_lock = lock_source_segments(data)

    validate_bilingual_result(data, source_lock)


def test_final_validation_rejects_empty_translation_after_post_processing():
    data = _data()
    source_lock = lock_source_segments(data)
    data.segments[0].translated_text = ""

    with pytest.raises(RuntimeError, match="empty translation at index 1"):
        validate_bilingual_result(data, source_lock)


def test_final_validation_rejects_source_boundary_changes():
    data = _data()
    source_lock = lock_source_segments(data)
    data.segments[0].end_time += 100

    with pytest.raises(RuntimeError, match="source timing changed at index 1"):
        validate_bilingual_result(data, source_lock)


def test_final_validation_rejects_segment_count_changes():
    data = ASRData(
        [
            ASRDataSeg("First clause", 0, 1000, "第一部分"),
            ASRDataSeg("second clause", 1000, 2000, "第二部分"),
        ]
    )
    source_lock = lock_source_segments(data)
    data.segments.pop()

    with pytest.raises(RuntimeError, match="segment count changed from 2 to 1"):
        validate_bilingual_result(data, source_lock)
