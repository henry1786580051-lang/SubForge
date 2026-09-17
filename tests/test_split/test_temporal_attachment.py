import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import _repair_stranded_temporal_clauses, normalize_boundaries
from subforge.core.split.temporal_attachment import temporal_clause_split

LEFT = "by the end of 2024, and that number could have increased to about half"
RIGHT = "by the end of last year, 2025."


def make_cues():
    cursor = 0
    cues = []
    for text in [LEFT, RIGHT]:
        words = []
        for token in text.split():
            words.append(
                ASRWord(text=token, start_time=cursor, end_time=cursor + 280, speaker_id="S1")
            )
            cursor += 300
        cues.append(
            ASRDataSeg(
                text=text,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
                words=words,
                speaker_id="S1",
            )
        )
    return cues


def test_repair_keeps_every_word_and_time_and_is_idempotent():
    cues = make_cues()
    result = normalize_boundaries(cues)
    assert [cue.text for cue in result] == [
        "by the end of 2024,",
        "and that number could have increased to about half by the end of last year, 2025.",
    ]
    assert [w for c in result for w in c.words] == [w for c in cues for w in c.words]
    assert [c.text for c in normalize_boundaries(result)] == [c.text for c in result]


@pytest.mark.parametrize(
    "left,right",
    [
        ("The share increased to half", RIGHT),
        ("It was last year", "actually, 2025."),
        ("by the end of 2024, and that number was incorrect", RIGHT),
        (LEFT, "by the end of last year, demand had fallen."),
    ],
)
def test_normal_supplements_and_corrections_are_not_rewritten(left, right):
    assert temporal_clause_split(left, right) is None


@pytest.mark.parametrize(
    "condition", ["speaker", "missing_words", "duration", "word_limit", "mismatch", "overlap"]
)
def test_unsafe_retiming_is_rejected(condition):
    cues = make_cues()
    limit = 22
    if condition == "speaker":
        cues[1].speaker_id = "S2"
    if condition == "missing_words":
        cues[0].words = []
    if condition == "duration":
        cues[1].words[-1].end_time = 30000
    if condition == "word_limit":
        limit = 10
    if condition == "mismatch":
        cues[0].words[0].text = "until"
    if condition == "overlap":
        cues[1].words[0].start_time = 0
    assert _repair_stranded_temporal_clauses(cues, hard_max_words=limit) == cues


def test_translated_cues_are_never_retimed():
    cues = make_cues()
    cues[0].translated_text = "译文"
    assert normalize_boundaries(cues) == cues


def test_real_word_timing_window_preserves_three_cues():
    import json
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/internet_power_temporal.json").read_text()
    )
    cues = [
        ASRDataSeg(
            text=c["text"],
            start_time=c["start"],
            end_time=c["end"],
            speaker_id="S3",
            words=[
                ASRWord(text=w["text"], start_time=w["start"], end_time=w["end"], speaker_id="S3")
                for w in c["words"]
            ],
        )
        for c in fixture
    ]
    result = normalize_boundaries(cues)
    assert len(result) == 3
    assert result[0].text == cues[0].text
    assert (result[1].start_time, result[1].end_time) == (292338, 294718)
    assert (result[2].start_time, result[2].end_time) == (295218, 302590)
    assert [w for c in result for w in c.words] == [w for c in cues for w in c.words]
    assert [(c.text, c.start_time, c.end_time) for c in normalize_boundaries(result)] == [
        (c.text, c.start_time, c.end_time) for c in result
    ]


def test_temporal_guidance_is_conditional_and_chinese_only():
    from subforge.core.translate.guidance import target_language_style_rules

    assert "Keep each deadline" in target_language_style_rules("简体中文", [LEFT, RIGHT])
    assert "Keep each deadline" not in target_language_style_rules("简体中文", ["This is a car."])
    assert "Keep each deadline" not in target_language_style_rules("English", [LEFT, RIGHT])
