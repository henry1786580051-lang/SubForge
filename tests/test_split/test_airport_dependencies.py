"""Development-triplet regressions and unrelated grammatical counterexamples."""

import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import assess_english_boundary, normalize_boundaries
from subforge.core.split.boundary_detectors.grammar import preposition_object_pronoun
from subforge.core.split.boundary_detectors.numeric import standalone_magnitude_named_population
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    "left,right",
    [
        ("Almost a million", "Berliners voted to keep it open."),
        ("Nearly two thousand", "Americans attended."),
        ("At least 20 million", "Europeans use the service."),
        ("One hundred", "Australians joined us."),
    ],
)
def test_quantity_fragment_keeps_capitalized_population(left, right):
    assert standalone_magnitude_named_population(extract_english_boundary_features(left, right))
    assert assess_english_boundary(left, right).unstable


@pytest.mark.parametrize(
    "left,right",
    [
        ("We raised a million", "Berliners voted the next day."),
        ("It cost nearly two million", "Americans wanted a refund."),
        ("Almost a million.", "Berliners voted."),
        ("Almost a million,", "Berliners said."),
        ("Almost a million", "This is unexpected."),
        ("Almost a million", "Thanks for asking."),
        ("Almost a million", "were counted."),
        ("Almost a million", "and counting."),
        ("Almost a million", ""),
        ("", "Berliners voted."),
    ],
)
def test_quantity_rule_does_not_reopen_complete_clauses(left, right):
    assert not standalone_magnitude_named_population(extract_english_boundary_features(left, right))


@pytest.mark.parametrize(
    "preposition,pronoun",
    [
        ("around", "me"),
        ("behind", "him"),
        ("beside", "us"),
        ("alongside", "them"),
    ],
)
def test_object_pronoun_disambiguates_preposition(preposition, pronoun):
    left, right = f"There is room {preposition}", f"{pronoun} for a table."
    assert preposition_object_pronoun(extract_english_boundary_features(left, right))
    assert assess_english_boundary(left, right).unstable


@pytest.mark.parametrize(
    "left,right",
    [
        ("Take a look around", "and enjoy the view."),
        ("They stayed behind", "while we left."),
        ("He turned around", "I was already gone."),
        ("I looked around", "Me, I prefer the old place."),
        ("He stayed behind,", "him being tired and all."),
        ("I built it around.", "Me too."),
        ("I looked around", "her team was already gone."),
        ("I looked around", "it was empty."),
        ("I looked around", "you were gone."),
        ("They went beyond", "them; the next group followed."),
        ("They went beyond", ""),
        ("", "me"),
    ],
)
def test_adverbs_and_new_clauses_are_not_missing_objects(left, right):
    assert not preposition_object_pronoun(extract_english_boundary_features(left, right))


FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/split/airport_dependencies.json").read_text()
)


def make_segments(case):
    result = []
    for cue in case["cues"]:
        words = [ASRWord(**w, language_code="en", timing_source="asr") for w in cue["words"]]
        result.append(
            ASRDataSeg(
                " ".join(w.text for w in words),
                cue["start_time"],
                cue["end_time"],
                words=words,
                speaker_id=words[0].speaker_id,
                language_code="en",
                timestamp_granularity="sentence",
                timing_source="asr",
            )
        )
    return result


def words_snapshot(segments):
    return [
        (w.text, w.start_time, w.end_time, w.speaker_id, w.language_code, w.timing_source)
        for s in segments
        for w in s.words
    ]


def cues_snapshot(segments):
    return [(s.text, s.start_time, s.end_time, s.speaker_id, s.translated_text) for s in segments]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: str(c["index"]))
def test_actual_word_timing_replay_repairs_dependency(case):
    segments = make_segments(case)
    before = words_snapshot(segments)
    assert assess_english_boundary(segments[0].text, segments[1].text).unstable
    result = normalize_boundaries(segments)
    assert any(case["protected_phrase"] in s.text for s in result)
    assert words_snapshot(result) == before
    assert cues_snapshot(normalize_boundaries(result)) == cues_snapshot(result)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: str(c["index"]))
@pytest.mark.parametrize(
    "protection", ["long_pause", "speaker_pause", "left_translated", "right_translated"]
)
def test_actual_dependencies_preserve_protected_boundaries(case, protection):
    segments = make_segments(case)
    if protection.endswith("translated"):
        segments[int(protection == "right_translated")].translated_text = "已有译文"
    else:
        # A speaker change must stop repair before the general 1800 ms pause
        # limit; exercise that separate guard rather than only a long silence.
        shift = 500 if protection == "speaker_pause" else 3000
        segments[1].start_time += shift
        segments[1].end_time += shift
        for w in segments[1].words:
            w.start_time += shift
            w.end_time += shift
        if protection == "speaker_pause":
            segments[1].speaker_id = "different-speaker"
            for w in segments[1].words:
                w.speaker_id = "different-speaker"
    before, words = cues_snapshot(segments), words_snapshot(segments)
    result = normalize_boundaries(segments)
    assert cues_snapshot(result) == before
    assert words_snapshot(result) == words
