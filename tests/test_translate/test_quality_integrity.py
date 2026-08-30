from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.translate.quality import (
    capture_segment_integrity,
    inspect_segment_integrity,
)


def _segment() -> ASRDataSeg:
    return ASRDataSeg(
        text="Original source",
        start_time=100,
        end_time=900,
        speaker_id="SPEAKER_00",
        words=[
            ASRWord(
                text="Original",
                start_time=100,
                end_time=500,
                speaker_id="SPEAKER_00",
                timing_source="forced_alignment",
                language_code="en",
            ),
            ASRWord(
                text="source",
                start_time=520,
                end_time=900,
                speaker_id="SPEAKER_00",
                timing_source="forced_alignment",
                language_code="en",
            ),
        ],
        timestamp_granularity="sentence",
        timing_source="forced_alignment",
        language_code="en",
    )


def test_segment_integrity_allows_only_translation_text_to_change():
    segment = _segment()
    snapshot = capture_segment_integrity((segment,))

    segment.translated_text = "原文"

    assert inspect_segment_integrity(snapshot, (segment,)) == ()


def test_segment_integrity_reports_source_timeline_and_word_mutations():
    segment = _segment()
    snapshot = capture_segment_integrity((segment,))

    segment.text = "Changed source"
    segment.end_time = 950
    segment.words[0].start_time = 80

    diagnostics = inspect_segment_integrity(snapshot, (segment,))

    assert [item.rule_id for item in diagnostics] == [
        "source.text_changed",
        "timeline.cue_timestamp_changed",
        "timeline.word_alignment_changed",
    ]


def test_segment_integrity_reports_segment_count_changes():
    segment = _segment()
    snapshot = capture_segment_integrity((segment,))

    diagnostics = inspect_segment_integrity(snapshot, ())

    assert [item.rule_id for item in diagnostics] == ["timeline.segment_count_changed"]
