"""Backend subtitle conversion helpers."""

import asyncio
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.api.subtitles import (
    ExportRequest,
    SegmentData,
    _normalize_segment_timing,
    _timestamp_to_ms,
    export_subtitle,
    export_subtitle_post,
    parse_ass,
    parse_srt,
    parse_vtt,
    segments_to_ass,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
)

from subforge.core.asr.asr_data import ASRData
from subforge.core.utils.atomic_write import encode_srt_text


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


def test_srt_download_encoding_is_word_compatible():
    payload = encode_srt_text("1\n00:00:00,000 --> 00:00:01,000\n中文\n")

    assert payload.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in payload
    assert payload.decode("utf-8-sig").splitlines()[-1] == "中文"


@pytest.mark.parametrize("value", ["", "not-a-time", "00:61:00.000", -1, float("nan")])
def test_strict_timestamp_parser_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _timestamp_to_ms(value, strict=True)


def test_strict_timestamp_parser_keeps_supported_formats():
    assert _timestamp_to_ms("00:01:02,345", strict=True) == 62_345
    assert _timestamp_to_ms("1:02.34", strict=True) == 62_340


@pytest.mark.parametrize("filename", ["字幕.srt", "Europe’s Airport.srt", "clip.srt", "line\r\nname.srt"])
def test_export_post_supports_unicode_and_safe_download_headers(filename):
    request = ExportRequest(filename=filename, segments=[SegmentData(text="Hello", translated="你好")])
    response = asyncio.run(export_subtitle_post(request))
    header = response.headers["content-disposition"]
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == request.filename
    assert "\r" not in header and "\n" not in header
    assert header.isascii()
    assert bytes(response.body).startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("filename", ["中文.srt", "Europe’s Airport.srt"])
def test_export_get_supports_unicode_filename(tmp_path, filename):
    path = tmp_path / filename
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\nHello\n", encoding="utf-8")
    response = asyncio.run(export_subtitle(str(path), format="srt", mode="bilingual"))
    header = response.headers["content-disposition"]
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == filename
    assert "你好\r\nHello" in bytes(response.body).decode("utf-8-sig")


@pytest.mark.parametrize("fmt,exporter,parser,core_parser", [
    ("srt", segments_to_srt, parse_srt, ASRData.from_srt),
    ("vtt", segments_to_vtt, parse_vtt, ASRData.from_vtt),
    ("ass", segments_to_ass, parse_ass, ASRData.from_ass),
])
@pytest.mark.parametrize("source,target", [
    ("Hello world.\nWelcome back.", "你好 世界\n欢迎回来"),
    ("오늘 날씨가 좋아요.", "今天天气很好"),
    ("今日はいい天気ですね。", "今天天气很好"),
    ("First source line\nSecond source line", ""),
])
def test_bilingual_format_roundtrip_preserves_columns(fmt, exporter, parser, core_parser, source, target):
    text = exporter([{"id": 1, "start": "00:00:01.000", "end": "00:00:02.000", "text": source, "translated": target}])
    parsed = parser(text)
    assert len(parsed) == 1
    assert (parsed[0]["text"], parsed[0]["translated"]) == (source, target)
    core = core_parser(text)
    assert len(core.segments) == 1
    assert (core.segments[0].text, core.segments[0].translated_text) == (source, target)
    assert (core.segments[0].start_time, core.segments[0].end_time) == (1000, 2000)
