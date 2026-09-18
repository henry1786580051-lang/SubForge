from pathlib import Path

import pytest

from scripts.repair_subtitle_boundaries import load_evidence, write_srt
from scripts.translation_quality.srt import parse_srt


def test_historical_evidence_roundtrip_keeps_bilingual_order_and_word_times(tmp_path: Path):
    source = tmp_path / "words.srt"
    bilingual = tmp_path / "bilingual.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:01,400\n[Speaker 1] Hello\n\n"
        "2\n00:00:01,500 --> 00:00:02,000\n[Speaker 1] world.\n"
    )
    bilingual.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好世界\nHello world.\n")
    cues = load_evidence(source, bilingual)
    assert cues[0].speaker_id == "1"
    assert [(w.start_time, w.end_time) for w in cues[0].words] == [(1000, 1400), (1500, 2000)]
    output = tmp_path / "result.srt"
    write_srt(output, cues)
    assert (
        parse_srt(output, layout="target_above").cues
        == parse_srt(bilingual, layout="target_above").cues
    )


def test_historical_reader_rejects_sentence_timing_as_word_evidence(tmp_path: Path):
    source = tmp_path / "words.srt"
    bilingual = tmp_path / "bilingual.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello world.\n")
    bilingual.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好世界\nHello world.\n")
    with pytest.raises(ValueError, match="one timed word"):
        load_evidence(source, bilingual)
