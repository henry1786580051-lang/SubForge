from pathlib import Path

import pytest

from subforge.core.asr.diarization_benchmark import (
    ReferenceWord,
    boundary_f1,
    diagnose_turns,
    load_ami_words,
    load_rttm,
    word_overlaps_multiple_speakers,
    word_speaker_error_rate,
    write_rttm,
)
from subforge.core.asr.speaker_diarization import SpeakerTurn


def test_rttm_round_trip(tmp_path: Path):
    output = tmp_path / "sample.rttm"
    expected = {
        "meeting-1": [
            SpeakerTurn(100, 700, "speaker-a"),
            SpeakerTurn(650, 1_200, "speaker-b"),
        ]
    }

    write_rttm(output, expected)

    assert load_rttm(output) == expected


def test_write_rttm_normalizes_internal_speaker_label(tmp_path: Path):
    output = tmp_path / "sample.rttm"

    write_rttm(output, {"meeting": [SpeakerTurn(0, 500, "Speaker 1")]})

    assert load_rttm(output) == {"meeting": [SpeakerTurn(0, 500, "Speaker_1")]}


def test_load_rttm_rejects_invalid_timing(tmp_path: Path):
    path = tmp_path / "invalid.rttm"
    path.write_text(
        "SPEAKER test 1 1.0 -0.5 <NA> <NA> speaker-a <NA> <NA>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="interval"):
        load_rttm(path)


def test_diagnostics_report_overlap_and_short_island():
    turns = [
        SpeakerTurn(0, 1_000, "A"),
        SpeakerTurn(900, 1_300, "B"),
        SpeakerTurn(1_300, 2_000, "A"),
    ]

    report = diagnose_turns(turns)

    assert report.speakers == 2
    assert report.overlap_ms == 100
    assert report.short_turns_500ms == 1
    assert report.short_islands_1500ms == 1


def test_boundary_f1_uses_one_to_one_matching():
    reference = [
        SpeakerTurn(0, 1_000, "A"),
        SpeakerTurn(1_000, 2_000, "B"),
        SpeakerTurn(2_000, 3_000, "A"),
    ]
    hypothesis = [
        SpeakerTurn(0, 950, "X"),
        SpeakerTurn(950, 1_100, "Y"),
        SpeakerTurn(1_100, 2_100, "X"),
        SpeakerTurn(2_100, 3_000, "Y"),
    ]

    report = boundary_f1(reference, hypothesis, tolerance_ms=150)

    assert report["matched"] == 2
    assert report["reference_boundaries"] == 2
    assert report["hypothesis_boundaries"] == 3
    assert report["recall"] == 1.0
    assert report["precision"] == pytest.approx(2 / 3)


def test_load_ami_words_keeps_timing_speaker_and_punctuation(tmp_path: Path):
    (tmp_path / "IB0001.A.words.xml").write_text(
        """<?xml version="1.0"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="w0" starttime="1.0" endtime="1.2">Hello</w>
  <w nite:id="w1" starttime="1.2" endtime="1.2" punc="true">.</w>
</nite:root>
""",
        encoding="utf-8",
    )

    assert load_ami_words(tmp_path, "IB0001") == [
        ReferenceWord("Hello.", 1_000, 1_200, "A")
    ]


def test_word_speaker_error_rate_maps_anonymous_labels_optimally():
    words = [
        ReferenceWord("one", 0, 100, "A"),
        ReferenceWord("two", 100, 200, "A"),
        ReferenceWord("three", 200, 300, "B"),
    ]

    report = word_speaker_error_rate(words, ["Speaker 2", "Speaker 2", "Speaker 1"])

    assert report["error_rate"] == 0.0
    assert report["mapping"] == {"Speaker 1": "B", "Speaker 2": "A"}


def test_word_overlap_uses_reference_speaker_activity():
    word = ReferenceWord("yes", 900, 1_100, "A")
    turns = [
        SpeakerTurn(0, 2_000, "A"),
        SpeakerTurn(1_000, 1_500, "B"),
    ]

    assert word_overlaps_multiple_speakers(word, turns)
