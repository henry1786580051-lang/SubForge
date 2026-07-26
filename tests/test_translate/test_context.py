from subforge.core.translate.context import _compact_transcript


def test_compact_transcript_samples_the_middle_of_long_transcript():
    segments = [f"section-{index} " + (chr(65 + index) * 80) for index in range(9)]

    compact = _compact_transcript(segments, limit=500)

    assert len(compact) <= 500
    assert "section-0" in compact
    assert "section-4" in compact
    assert "section-8" in compact
    assert compact.count("\n...\n") == 4


def test_compact_transcript_keeps_short_transcript_unchanged():
    assert _compact_transcript([" first ", "second"], limit=100) == "first second"
