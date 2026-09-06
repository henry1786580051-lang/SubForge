"""Regression for Elantra cue 25: a compact size spans eight spoken words."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.asr.compound_numbers import parse_tire_size
from subforge.core.asr.whisperx_asr import (
    WhisperXASR,
    _compound_timing_candidates,
    _prepare_spoken_alignment,
    _restore_display_alignment,
    _spoken_token,
)


@pytest.mark.parametrize("text", ["245-35ZR19.", "245/35ZR19", "245/35zr19", "(245/35ZR19)"])
def test_compact_tire_size_expands_to_complete_speech(text):
    assert _spoken_token(text).rstrip(".") == "two forty five thirty five Z R nineteen"


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("225/45R18", "two twenty five forty five R eighteen"),
        ("205/55R16", "two oh five fifty five R sixteen"),
        ("200/50R20", "two hundred fifty R twenty"),
    ],
)
def test_other_tire_sizes(text, spoken):
    assert _spoken_token(text) == spoken


@pytest.mark.parametrize(
    "text",
    [
        "0-60",
        "2026",
        "M4",
        "245-35",
        "245/35",
        "245/35XYZ19",
        "999/99R99",
        "245/35R190",
        "abc245/35R19",
    ],
)
def test_non_sizes_are_not_given_tire_duration_allowances(text):
    assert parse_tire_size(text) is None


def test_spaced_size_preserves_display_token_boundaries_and_non_english():
    original = [{"text": "Tires are 245/35 ZR19.", "start": 0, "end": 5}]
    normalized, plans = _prepare_spoken_alignment(original, "en")
    assert normalized[0]["text"] == "Tires are two forty five thirty five Z R nineteen."
    assert [token.display_text for token in plans[0].tokens] == original[0]["text"].split()
    assert _prepare_spoken_alignment(original, "zh") == (original, None)
    assert _spoken_token("245-35") == "two hundred forty five to thirty five"


def test_elantra_spoken_alignment_survives_display_mapping_cap_and_sentence_merge():
    source = (
        "And we have a squared tire setup, a 245-35ZR19. I do think these wheels look excellent."
    )
    normalized, plans = _prepare_spoken_alignment(
        [{"text": source, "start": 105.7, "end": 112.15}], "en"
    )
    assert "two forty five thirty five Z R nineteen" in normalized[0]["text"]
    fixture = Path(__file__).parents[1] / "fixtures/asr/elantra_tire_alignment.json"
    words = json.loads(fixture.read_text())["words"]
    restored = _restore_display_alignment({"segments": [{"words": words}]}, plans)
    asr = object.__new__(WhisperXASR)
    asr.need_word_time_stamp = True
    data = ASRData(asr._make_segments(restored))
    data.cap_abnormal_word_durations()
    tire = next(seg for seg in data.segments if seg.text == "245-35ZR19.")
    assert tire.start_time == 107603
    assert tire.end_time == 110007
    boundary = data.segments.index(tire) + 1
    next_start = data.segments[boundary].start_time
    cues = ASRData(
        [
            ASRDataSeg.from_segments(data.segments[:boundary], text=source.split(" I do")[0]),
            ASRDataSeg.from_segments(
                data.segments[boundary:], text="I do think these wheels look excellent."
            ),
        ]
    )
    cues.extend_sentence_tails_conservatively([(105800, 110500)])
    assert cues.segments[0].end_time == 110007
    assert cues.segments[1].start_time == next_start == 110087
    assert [seg.text for seg in data.segments] == source.split()


def test_partial_numeric_suffix_cannot_be_accepted_as_complete_alignment():
    _, plans = _prepare_spoken_alignment([{"text": "245-35ZR19.", "start": 1, "end": 4}], "en")
    words = [
        {"word": word, "start": 1 + i * 0.2, "end": 1.2 + i * 0.2}
        for i, word in enumerate("two forty five thirty five Z R".split())
    ]
    assert _restore_display_alignment({"segments": [{"words": words}]}, plans) is None


def test_untimed_numeric_suffix_does_not_inherit_the_preceding_words_end():
    _, plans = _prepare_spoken_alignment([{"text": "245-35ZR19.", "start": 1, "end": 4}], "en")
    words = [
        {"word": word, "start": 1 + i * 0.2, "end": 1.2 + i * 0.2}
        for i, word in enumerate("two forty five thirty five Z R".split())
    ] + [{"word": "nineteen."}]
    assert _restore_display_alignment({"segments": [{"words": words}]}, plans) is None


def test_tire_duration_protection_is_bounded_and_keeps_other_words_unchanged():
    data = ASRData(
        [
            ASRDataSeg("245/35ZR19", 1000, 9000, timestamp_granularity="word"),
            ASRDataSeg("200,000", 10000, 18000, timestamp_granularity="word"),
            ASRDataSeg("next", 18100, 18400, timestamp_granularity="word"),
        ]
    )
    data.cap_abnormal_word_durations()
    assert 3400 <= data.segments[0].end_time - 1000 <= 4200
    assert data.segments[0].words[-1].end_time == data.segments[0].end_time
    assert data.segments[1].end_time == 11400
    assert data.segments[2].start_time == 18100
    assert data.segments[2].end_time == 18400


def test_spaced_tire_size_preserves_a_slowly_spoken_numeric_prefix():
    data = ASRData([ASRDataSeg("245/35", 1000, 3000), ASRDataSeg("ZR19", 3050, 4150)])
    data.cap_abnormal_word_durations()
    assert [(seg.start_time, seg.end_time) for seg in data.segments] == [(1000, 3000), (3050, 4150)]


@pytest.mark.parametrize(
    "ending, next_onset, score, expected",
    [
        (3.9, 4.0, 0.9, 3900),
        (2.0, 4.0, 0.9, 2000),
        (3.9, 4.0, 0.4, 2000),
        (3.9, 4.3, 0.9, 2000),
        (4.5, 4.0, 0.9, 2000),
    ],
)
def test_local_realign_requires_acoustic_evidence_and_preserves_pauses(
    monkeypatch, ending, next_onset, score, expected
):
    import numpy as np

    package = ModuleType("whisperx")
    alignment = ModuleType("whisperx.alignment")
    audio = ModuleType("whisperx.audio")
    audio.load_audio = lambda _: np.zeros(80000, dtype=np.float32)
    package.alignment = alignment
    monkeypatch.setitem(sys.modules, "whisperx", package)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", alignment)
    monkeypatch.setitem(sys.modules, "whisperx.audio", audio)
    data = ASRData(
        [
            ASRDataSeg("setup,", 1000, 1300, timestamp_granularity="word"),
            ASRDataSeg("245-35ZR19.", 1400, 2000, timestamp_granularity="word"),
            ASRDataSeg("I", 4000, 4100, timestamp_granularity="word"),
            ASRDataSeg("do", 4140, 4300, timestamp_granularity="word"),
        ]
    )
    offset = 0.65
    words = [
        {
            "word": s.text,
            "start": s.start_time / 1000 - offset,
            "end": s.end_time / 1000 - offset,
            "score": 0.9,
        }
        for s in data.segments
    ]
    words[1].update(end=ending - offset, score=score)
    words[2]["start"] = next_onset - offset
    asr = object.__new__(WhisperXASR)
    asr._align_result = lambda *_: {"word_segments": words}
    repaired = asr.realign_compound_word_gaps(data, "original.wav")
    assert data.segments[1].end_time == expected
    assert data.segments[1].words[-1].end_time == expected
    assert repaired == int(expected != 2000)
    assert data.segments[2].start_time == 4000
    assert data.segments[3].end_time == 4300


def test_local_realign_ignores_ordinary_words_and_already_tight_sizes():
    for text in ("2026", "ordinary", "245-35ZR19."):
        data = ASRData([ASRDataSeg(text, 1000, 2000), ASRDataSeg("next", 2080, 2300)])
        assert _compound_timing_candidates(data) == []


def test_local_realign_honors_cancellation_before_loading_audio(monkeypatch):
    import sys
    import threading

    # A cancelled operation must not require optional inference dependencies,
    # including on a minimal CI install without the WhisperX extra.
    monkeypatch.setitem(sys.modules, "whisperx.alignment", None)
    monkeypatch.setitem(sys.modules, "whisperx.audio", None)
    asr = object.__new__(WhisperXASR)
    asr.cancel_event = threading.Event()
    asr.cancel_event.set()
    data = ASRData([ASRDataSeg("245-35ZR19.", 1000, 2000), ASRDataSeg("next", 4000, 4300)])
    with pytest.raises(RuntimeError, match="cancelled"):
        asr.realign_compound_word_gaps(data, "unread.wav")
