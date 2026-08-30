from pathlib import Path

import pytest

from scripts.translation_quality.srt import SrtParseError, parse_srt


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_target_above_preserves_multiline_source(tmp_path: Path):
    path = _write(
        tmp_path / "sample.srt",
        """1
00:00:01,000 --> 00:00:03,000
这是一条自然的中文
This is one English
source split over two lines.

2
00:00:03,100 --> 00:00:04,000
第二条
Second cue.
""",
    )

    document = parse_srt(path, layout="target_above")

    assert len(document.cues) == 2
    assert document.cues[0].target == "这是一条自然的中文"
    assert document.cues[0].source == "This is one English\nsource split over two lines."
    assert document.cues[0].start_ms == 1000
    assert document.cues[0].end_ms == 3000


def test_parse_source_only_keeps_all_text_lines(tmp_path: Path):
    path = _write(
        tmp_path / "source.srt",
        """7
00:01:02.003 --> 00:01:04.005
First source line
second source line
""",
    )

    cue = parse_srt(path, layout="source_only").cues[0]

    assert cue.index == 7
    assert cue.source == "First source line\nsecond source line"
    assert cue.target == ""
    assert cue.start_ms == 62_003
    assert cue.end_ms == 64_005


def test_parse_rejects_duplicate_indices(tmp_path: Path):
    path = _write(
        tmp_path / "bad.srt",
        """1
00:00:00,000 --> 00:00:01,000
One

1
00:00:01,000 --> 00:00:02,000
Two
""",
    )

    with pytest.raises(SrtParseError, match="Duplicate cue index 1"):
        parse_srt(path, layout="source_only")


def test_parse_utf16_bom_file(tmp_path: Path):
    path = tmp_path / "utf16.srt"
    path.write_text(
        "1\r\n00:00:00,000 --> 00:00:01,000\r\n中文译文\r\nSource.\r\n",
        encoding="utf-16",
    )

    document = parse_srt(path, layout="target_above")

    assert document.encoding == "utf-16"
    assert document.has_bom is True
    assert document.newline == "crlf"
    assert document.cues[0].target == "中文译文"
