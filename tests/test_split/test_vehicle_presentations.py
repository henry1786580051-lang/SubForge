import copy
import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRWord
from subforge.core.split.boundary import (
    _is_hard_boundary,
    _join_words,
    _make_cue,
    assess_english_boundary,
    normalize_boundaries,
)
from subforge.core.split.presentation import opening_presentation_indices, plan_presentations
from subforge.core.split.vehicle_names import introduction_seam, splits_vehicle_name


def fixture(cuts):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/lexus_es_presentation.json").read_text()
    )
    words = [
        ASRWord(text, start, end, timing_source="imported", language_code="en")
        for text, start, end in data["words"]
    ]
    result, start = [], 0
    for stop in cuts:
        result.append(_make_cue(words[start:stop], ""))
        start = stop
    return result


@pytest.mark.parametrize("cuts", [[14, 21, 37], [17, 25, 37], [7, 21, 25, 37], [37]])
def test_lexus_intro_is_separate_and_full_vehicle_is_one_cue(cuts):
    original = fixture(cuts)
    words = [w for c in original for w in c.words]
    result = normalize_boundaries(original)
    assert [len(c.words) for c in result] == [7, 14, 4, 12]
    assert [w for c in result for w in c.words] == words
    assert "Topher Drives" in result[0].text and "Lexus" not in result[0].text
    assert "2026 Lexus ES350H Premium All-Wheel Drive," in result[1].text
    assert [(c.start_time, c.end_time) for c in result] == [
        (60, 2305),
        (2345, 8620),
        (8880, 10404),
        (10806, 15960),
    ]
    assert [c.__dict__ for c in normalize_boundaries(result)] == [c.__dict__ for c in result]
    assert opening_presentation_indices(result) == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "left,right",
    [
        ("The 2026 Lexus", "ES350H Premium All-Wheel Drive,"),
        ("This is the 2026", "Lexus ES350H"),
        ("A Lexus ES", "350h Premium"),
        ("The Mercedes-AMG GLC", "53 is here"),
        ("A BMW", "iX5 is parked"),
        ("The Hyundai IONIQ", "5 XRT is here"),
        ("A Ford", "F-150 Lightning"),
        ("The GMC Sierra", "1500 AT4X"),
        ("A Porsche", "911 Turbo"),
        ("A Tesla Model", "3 Performance"),
        ("All-Wheel", "Drive"),
    ],
)
def test_identifier_is_a_hard_boundary_constraint(left, right):
    assert splits_vehicle_name(left, right)
    assessment = assess_english_boundary(left, right)
    assert "vehicle name or drive identifier split across cues" in assessment.reasons


@pytest.mark.parametrize(
    "left,right",
    [
        ("We visited Lexus.", "This is the next topic."),
        ("Lexus", "announced 350 sales today"),
        ("The Toyota", "Motor company opened a branch"),
        ("The showroom has BMW,", "Lexus and Toyota vehicles"),
        ("Here is a car.", "BMW iX5 is written on it."),
        ("The GMC Sierra,", "1500 people were there."),
    ],
)
def test_vehicle_detector_does_not_join_unrelated_brand_mentions(left, right):
    assert not splits_vehicle_name(left, right)


@pytest.mark.parametrize(
    "brand_model",
    [
        "2027 BMW iX5 xDrive",
        "2026 Mercedes-AMG GLC 53",
        "2026 Ford F-150",
        "2027 GMC Sierra 1500 Denali Ultimate",
    ],
)
def test_channel_intro_exception_is_not_specific_to_reported_model(brand_model):
    assert introduction_seam(
        "Hi everyone, welcome back to Daily Drives,",
        f"where today we're driving the {brand_model}.",
    )
    assert not introduction_seam(
        "We visited the showroom,", f"where today we're driving the {brand_model}."
    )


@pytest.mark.parametrize(
    "failure",
    [
        "speaker",
        "language",
        "pause",
        "estimated",
        "overlap",
        "missing",
        "text",
        "limit",
        "late",
        "translated",
    ],
)
def test_presentation_planner_declines_unsafe_retiming(failure):
    original = fixture([14, 21, 37])
    if failure == "speaker":
        original[1].speaker_id = "other"
    if failure == "language":
        original[1].language_code = "fr"
    if failure == "pause":
        original[1].words[0].start_time += 3000
    if failure == "estimated":
        original[1].words[0].timing_source = "estimated"
    if failure == "overlap":
        original[1].words[0].start_time = 3000
    if failure == "missing":
        original[1].words = []
    if failure == "text":
        original[1].text = "A different car was mentioned."
    if failure == "late":
        original[0].start_time = 65000
    if failure == "translated":
        original[1].translated_text = "保留已存在的译文"
    before = copy.deepcopy([c.__dict__ for c in original])
    result = plan_presentations(
        original,
        hard_max_words=5 if failure == "limit" else 22,
        join_words=_join_words,
        make_cue=_make_cue,
        hard_boundary=_is_hard_boundary,
    )
    assert [c.__dict__ for c in result] == before


def test_existing_bilingual_subtitles_are_never_rebuilt():
    cues = fixture([17, 25, 37])
    cues[0].translated_text = "用户保留的译文"
    assert normalize_boundaries(cues) == cues


@pytest.mark.parametrize("failure", ["speaker", "language", "pause", "late", "overlap"])
def test_translation_polish_window_preserves_turns_pauses_and_languages(failure):
    cues = fixture([7, 21, 25, 37])
    if failure == "speaker":
        cues[1].speaker_id = "other"
    if failure == "language":
        cues[1].language_code = "fr"
    if failure == "pause":
        cues[1].start_time += 3000
    if failure == "late":
        cues[0].start_time = 65000
    if failure == "overlap":
        cues[1].start_time = 1000
    assert opening_presentation_indices(cues) == ()
