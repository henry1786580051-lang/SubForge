import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import assess_english_boundary, normalize_boundaries
from subforge.core.split.boundary_detectors.grammar import morphological_attributive_modifier
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    "left,right",
    [
        ("We read the manual,", "though Alice would prefer a video."),
        ("They reached the hospital,", "because it was nearby."),
        ("It is available to the public,", "while they are waiting."),
        ("The enclosure is flexible,", "although it can withstand impacts."),
    ],
)
def test_explicit_clause_is_not_a_missing_adjective_head(left, right):
    assert not morphological_attributive_modifier(extract_english_boundary_features(left, right))


@pytest.mark.parametrize(
    "left,right",
    [
        ("It uses a flexible", "panel."),
        ("It uses a flexible,", "though fragile, panel."),
        ("It uses a flexible", "although it can bend"),
        ("It is a typical,", "if expensive, solution."),
        ("It has an electric", "motor."),
        ("We read the manual,", "though Alice"),
    ],
)
def test_missing_or_parenthetical_clause_evidence_keeps_existing_risk(left, right):
    assert morphological_attributive_modifier(extract_english_boundary_features(left, right))


def _segments(speaker_change=False, gap=20, language="en", translated=False):
    cursor = 0
    segments = []
    for index, text in enumerate(
        [
            "This special edition has a six-speed",
            "manual, though Alice would also consider an automatic gearbox.",
        ]
    ):
        speaker = "S2" if index and speaker_change else "S1"
        words = []
        for token in text.split():
            words.append(
                ASRWord(
                    token,
                    cursor,
                    cursor + 550,
                    speaker_id=speaker,
                    timing_source="forced_alignment",
                    language_code=language,
                )
            )
            cursor += 600
        segments.append(
            ASRDataSeg(
                text,
                words[0].start_time,
                words[-1].end_time,
                words=words,
                speaker_id=speaker,
                language_code=language,
                timestamp_granularity="sentence",
                timing_source="forced_alignment",
                translated_text="已翻译" if translated else "",
            )
        )
        cursor += gap
    return segments


def _word_snapshot(segments):
    return [
        (w.text, w.start_time, w.end_time, w.speaker_id, w.timing_source)
        for segment in segments
        for w in segment.words
    ]


def _cue_snapshot(segments):
    return [
        (s.text, s.start_time, s.end_time, s.speaker_id, s.language_code, s.translated_text)
        for s in segments
    ]


def test_normalizer_keeps_compound_and_uses_following_complete_clause():
    segments = _segments()
    expected_words = _word_snapshot(segments)
    result = normalize_boundaries(segments)
    assert [s.text for s in result] == [
        "This special edition has a six-speed manual,",
        "though Alice would also consider an automatic gearbox.",
    ]
    assert _word_snapshot(result) == expected_words
    assert not assess_english_boundary(result[0].text, result[1].text).unstable
    assert _cue_snapshot(normalize_boundaries(result)) == _cue_snapshot(result)


@pytest.mark.parametrize("translated_index", [0, 1])
def test_partially_translated_compact_pair_is_not_rebuilt(translated_index):
    segments = _segments()
    # Compact enough to enter the merge pass before the old, late guard.
    for segment in segments:
        segment.text = segment.text.replace("six-speed", "six speed")
        for word in segment.words:
            word.text = word.text.replace("six-speed", "six speed")
            word.start_time //= 4
            word.end_time //= 4
        segment.start_time = segment.words[0].start_time
        segment.end_time = segment.words[-1].end_time
    segments[translated_index].translated_text = "已有译文"
    before = _cue_snapshot(segments)
    result = normalize_boundaries(segments)
    assert _cue_snapshot(result) == before
    assert all(a is b for a, b in zip(result, segments))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaker_change": True, "gap": 2000},
        {"gap": 2500},
        {"translated": True},
    ],
)
def test_normalizer_preserves_hard_boundaries_and_translated_cues(kwargs):
    segments = _segments(**kwargs)
    before = _cue_snapshot(segments)
    words = _word_snapshot(segments)
    result = normalize_boundaries(segments)
    assert _cue_snapshot(result) == before
    assert _word_snapshot(result) == words
