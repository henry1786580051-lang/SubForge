"""Backend subtitle conversion helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.api.subtitles import (
    _normalize_segment_timing,
    segments_to_ass,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
)


def test_normalize_segment_timing_removes_overlap():
    segments = [
        {"id": 1, "start": "00:00:00.000", "end": "00:00:02.000", "text": "First"},
        {"id": 2, "start": "00:00:01.500", "end": "00:00:03.000", "text": "Second"},
    ]

    normalized = _normalize_segment_timing(segments)

    assert normalized[0]["end"] == "00:00:01.750"
    assert normalized[1]["start"] == "00:00:01.750"


def test_segments_to_srt_uses_normalized_timing_shape():
    segments = _normalize_segment_timing([
        {"id": 1, "start": "00:00:00,000", "end": "00:00:02,000", "text": "First"},
        {"id": 2, "start": "00:00:01,500", "end": "00:00:03,000", "text": "Second"},
    ])

    srt = segments_to_srt(segments)

    assert "00:00:00,000 --> 00:00:01,750" in srt
    assert "00:00:01,750 --> 00:00:03,000" in srt


def test_bilingual_exports_put_translation_above_source():
    segments = [
        {
            "id": 1,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "This chart shows jet fuel costs.",
            "translated": "这张图展示的是航空燃油成本。",
        }
    ]

    srt = segments_to_srt(segments)
    vtt = segments_to_vtt(segments)
    txt = segments_to_txt(segments)
    ass = segments_to_ass(segments)

    assert "这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs." in srt
    assert "这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs." in vtt
    assert txt.startswith("这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs.")
    assert "这张图展示的是航空燃油成本。\\NThis chart shows jet fuel costs." in ass
