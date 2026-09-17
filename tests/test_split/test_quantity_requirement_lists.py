import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import _repair_quantified_requirement_lists, normalize_boundaries
from subforge.core.split.boundary_detectors.numeric import spelled_quantity_unit
from subforge.core.split.boundary_features import extract_english_boundary_features


@pytest.mark.parametrize(
    "left,right",
    [
        ("They consume several hundred", "megawatts on their own,"),
        ("It needs two", "gigawatts."),
        ("It uses a few million", "litres of water."),
        ("It weighs five", "kilograms."),
    ],
)
def test_spelled_physical_quantity_is_protected(left, right):
    assert spelled_quantity_unit(extract_english_boundary_features(left, right))


@pytest.mark.parametrize(
    "left,right",
    [
        ("We counted several hundred.", "Megawatts are a unit."),
        ("It opened in 2024,", "megawatts were discussed."),
        ("There were several hundred", "people outside."),
        ("It uses a model five", "and works well."),
    ],
)
def test_no_join_for_new_sentences_or_ordinary_nouns(left, right):
    assert not spelled_quantity_unit(extract_english_boundary_features(left, right))


def cues():
    data = json.loads((Path(__file__).parent / "fixtures/internet_power_quantity.json").read_text())
    return [
        ASRDataSeg(
            text=c["text"],
            start_time=c["start"],
            end_time=c["end"],
            speaker_id="S1",
            words=[
                ASRWord(text=w["text"], start_time=w["start"], end_time=w["end"], speaker_id="S1")
                for w in c["words"]
            ],
        )
        for c in data
    ]


def test_real_requirement_list_keeps_quantity_relative_clause_and_word_times():
    original = cues()
    result = normalize_boundaries(original)
    assert len(result) == 3
    assert result[0].text.endswith("connections, water sources")
    assert (
        result[1].text
        == "and, for the really large ones that handle several hundred megawatts on their own,"
    )
    assert result[2].text == "high capacity power grids."
    assert (result[0].end_time, result[1].start_time, result[1].end_time, result[2].start_time) == (
        334462,
        334706,
        339278,
        339755,
    )
    assert [w for c in result for w in c.words] == [w for c in original for w in c.words]
    assert [(c.text, c.start_time, c.end_time) for c in normalize_boundaries(result)] == [
        (c.text, c.start_time, c.end_time) for c in result
    ]


@pytest.mark.parametrize("case", ["speaker", "missing", "long", "limit", "mismatch"])
def test_unsafe_requirement_retiming_is_rejected(case):
    original = cues()
    if case == "speaker":
        original[0].speaker_id = "S2"
    if case == "missing":
        original[1].words = []
    if case == "long":
        original[1].words[0].start_time -= 9000
    if case == "mismatch":
        original[1].words[0].text = "changed"
    assert (
        _repair_quantified_requirement_lists(original, hard_max_words=8 if case == "limit" else 22)
        == original
    )


def test_on_their_own_idiom_does_not_hide_own_head_dependency():
    assert extract_english_boundary_features(
        "It works on their own,", "high capacity grids."
    ).complete_own_idiom
    assert not extract_english_boundary_features(
        "It works on their own", "power grid."
    ).complete_own_idiom
