import copy
import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import (
    _is_hard_boundary,
    _join_words,
    _make_cue,
    assess_english_boundary,
)
from subforge.core.split.mapped_boundary import map_cleaned_words, normalize_mapped_boundaries
from subforge.core.split.syntax_units import is_atomic_seam
from subforge.core.split.unit_planner import plan_syntax_units
from subforge.core.split.vehicle_names import splits_vehicle_name


def source_window(first):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/porsche_structural_boundaries.json").read_text()
    )
    window = next(w for w in data["windows"] if w["start_id"] == first)
    return [
        ASRDataSeg(
            c["source"],
            c["start"],
            c["end"],
            words=[
                ASRWord(t, a, b, timing_source="imported", language_code="en")
                for t, a, b in c["words"]
            ],
            timing_source="imported",
            language_code="en",
        )
        for c in window["cues"]
    ]


@pytest.mark.parametrize("first", [9, 155, 221, 226, 234, 415, 419, 552, 600])
def test_reported_windows_retain_every_word_and_remove_atomic_splits(first):
    original = source_window(first)
    result = normalize_mapped_boundaries(original)
    assert [w for c in result for w in c.words] == [w for c in original for w in c.words]
    assert all(
        not is_atomic_seam(a.text, b.text) and not splits_vehicle_name(a.text, b.text)
        for a, b in zip(result, result[1:])
    )
    assert all(a.end_time <= b.start_time for a, b in zip(result, result[1:]))
    assert [c.__dict__ for c in normalize_mapped_boundaries(result)] == [c.__dict__ for c in result]
    if first == 226:
        assert len(result) == 4
        assert any(
            "the car will already preemptively start lifting the front" in c.text.lower()
            for c in result
        )
    if first == 221:
        assert any(c.text.startswith("If you want this to go down") for c in result)


@pytest.mark.parametrize(
    "unsafe",
    [
        "speaker",
        "language",
        "estimated",
        "overlap",
        "missing",
        "replacement",
        "pause",
        "translated",
    ],
)
def test_variable_planner_declines_unsafe_word_evidence(unsafe):
    cues = source_window(226)
    if unsafe == "speaker":
        cues[1].speaker_id = "other"
    if unsafe == "language":
        cues[1].language_code = "fr"
    if unsafe == "estimated":
        cues[1].words[0].timing_source = "estimated"
    if unsafe == "overlap":
        cues[1].words[0].start_time = cues[0].words[-1].start_time
    if unsafe == "missing":
        cues[1].words = []
    if unsafe == "replacement":
        cues[1].text = "Unrelated replacement."
    if unsafe == "pause":
        for c in cues[1:]:
            c.start_time += 3000
            c.end_time += 3000
            for w in c.words:
                w.start_time += 3000
                w.end_time += 3000
    if unsafe == "translated":
        cues[0].translated_text = "保留用户译文"
    before = copy.deepcopy([c.__dict__ for c in cues])
    result = plan_syntax_units(
        cues,
        soft_max=18,
        hard_max=22,
        join_words=_join_words,
        make_cue=_make_cue,
        hard_boundary=_is_hard_boundary,
        assess=assess_english_boundary,
    )
    if unsafe == "pause":
        assert result[0].__dict__ == before[0]
        assert all(c.start_time >= cues[1].start_time for c in result[1:])
        assert [w for c in result for w in c.words] == [w for c in cues for w in c.words]
    else:
        assert [c.__dict__ for c in result] == before


@pytest.mark.parametrize(
    "left,right",
    [
        ("If you want,", "we can leave."),
        ("You can do what you want.", "This is next."),
        ("The car already stopped.", "Start the engine."),
        ("We visited the dealer.", "Cars are parked outside."),
        ("I like this hybrid.", "The setup is easy."),
    ],
)
def test_complete_clauses_are_not_treated_as_source_atoms(left, right):
    assert not is_atomic_seam(left, right)


@pytest.mark.parametrize(
    "left,right",
    [
        ("The Porsche 911", "Turbo S Cabriolet is here."),
        ("The Porsche 911 Turbo", "S Cabriolet is here."),
        ("The Turbo S", "Cabriolet is here."),
    ],
)
def test_full_observed_model_and_body_style_are_atomic(left, right):
    assert splits_vehicle_name(left, right)


def test_named_suffix_does_not_match_s_shaped_adjective():
    assert not splits_vehicle_name("a turbo", "S-shaped hose")


@pytest.mark.parametrize("name", ["grand_old_time", "entry_point"])
def test_additional_full_file_phrase_breaks_keep_raw_words_and_times(name):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/asr/porsche_additional_units.json").read_text()
    )
    window = next(w for w in data["windows"] if w["name"] == name)
    cues = [
        ASRDataSeg(
            c["source"],
            c["start"],
            c["end"],
            words=[
                ASRWord(t, a, b, timing_source="imported", language_code="en")
                for t, a, b in c["words"]
            ],
            timing_source="imported",
            language_code="en",
        )
        for c in window["cues"]
    ]
    result = normalize_mapped_boundaries(cues)
    assert [w for c in result for w in c.words] == [w for c in cues for w in c.words]
    assert all(not is_atomic_seam(a.text, b.text) for a, b in zip(result, result[1:]))
    assert any(name.replace("_", " ") in c.text.lower() for c in result)


@pytest.mark.parametrize(
    "left,right",
    [
        ("I've never seen that. I wish", "I would have had the camera on."),
        ("Sounds from the 3.6 liter twin turbo", "flat six in this car."),
        ("it ignitions on, and then", "it's your job to do the final click."),
        ("we've evaluated most modern", "911s at this point, the Carrera S."),
        ("the entry point for a base", "911, and especially a Turbo S."),
        ("as far as tire", "technology right now."),
        ("But maybe your partner has something", "like a Macan or a Cayenne,"),
        ("the most genius integration of a wind deflector", "I've ever seen in any convertible."),
    ],
)
def test_live_observed_predicates_and_engine_units_stay_together(left, right):
    assert is_atomic_seam(left, right)


@pytest.mark.parametrize("tail", ["something", "anything", "everything", "nothing"])
def test_indefinite_pronoun_is_not_a_participle(tail):
    assessment = assess_english_boundary(
        f"because if you're pulling into your driveway or {tail},",
        "the car will start lifting the front.",
    )
    assert "participle separated from its complement" not in assessment.reasons


@pytest.mark.parametrize(
    "left,right",
    [
        (
            "It's unacceptable at the $130,000 entry point for a base 911,",
            "and it's especially unacceptable for the Turbo S.",
        ),
        ("It costs 130,000,", "but it has a different engine."),
    ],
)
def test_complete_numeric_fact_before_a_new_finite_clause_is_not_a_missing_unit(left, right):
    assert (
        "numeric value separated from its unit or noun"
        not in assess_english_boundary(left, right).reasons
    )


@pytest.mark.parametrize(
    "right", ["dollars for this car.", "miles per hour.", "people in the room.", "and a half feet."]
)
def test_real_numeric_complements_remain_protected(right):
    assert assess_english_boundary("130", right).unstable


@pytest.mark.parametrize(
    "raw,clean,allowed",
    [
        ("the flat sixth engine", "the flat six engine", True),
        ("the sixth gear", "the six gear", False),
        ("a flat sixth gear", "a flat six gear", False),
    ],
)
def test_engine_spelling_alias_preserves_raw_word_provenance(raw, clean, allowed):
    words = [
        ASRWord(t, i * 100, (i + 1) * 100, timing_source="imported")
        for i, t in enumerate(raw.split())
    ]
    cue = ASRDataSeg(clean, 0, len(words) * 100, words=words, timing_source="imported")
    before = copy.deepcopy([w.__dict__ for w in words])
    mapping = map_cleaned_words(cue)
    assert bool(not mapping.error) == allowed
    assert [w.__dict__ for w in words] == before
