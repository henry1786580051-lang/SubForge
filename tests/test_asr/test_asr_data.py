"""ASRData 核心功能测试 - 严格边缘用例"""

import tempfile
import unicodedata
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg, ASRWord, handle_long_path
from subforge.core.utils.cache import get_subtitle_language_cache


def test_language_metadata_survives_srt_task_boundary(tmp_path):
    subtitle_path = tmp_path / "mixed.srt"
    data = ASRData(
        [
            ASRDataSeg("Welcome", 0, 500, language_code="en"),
            ASRDataSeg("こんにちは", 600, 1200, language_code="ja"),
        ]
    )
    data.save(str(subtitle_path))
    data.save_language_metadata(str(subtitle_path))

    restored = ASRData.from_subtitle_file(str(subtitle_path))

    assert [segment.language_code for segment in restored.segments] == ["en", "ja"]
    get_subtitle_language_cache().delete(data._language_metadata_key(str(subtitle_path)))


def test_timing_metadata_survives_exact_srt_and_rejects_edited_file(tmp_path):
    subtitle_path = tmp_path / "timed.srt"
    data = ASRData([ASRDataSeg("Welcome", 0, 500)])
    data.save(str(subtitle_path))
    data.save_timing_metadata(str(subtitle_path), [(0, 620)], 1_000)

    assert ASRData.load_timing_metadata(str(subtitle_path)) == ([(0, 620)], 1_000)

    subtitle_path.write_text(
        subtitle_path.read_text(encoding="utf-8").replace("Welcome", "Edited"),
        encoding="utf-8",
    )
    assert ASRData.load_timing_metadata(str(subtitle_path)) == ([], None)


def test_merged_segment_records_mixed_source_languages():
    merged = ASRDataSeg.from_segments(
        [
            ASRDataSeg("This is", 0, 500, language_code="en"),
            ASRDataSeg("木組み", 500, 1000, language_code="ja"),
        ],
        text="This is 木組み",
    )

    assert merged.language_code == "mixed"


class TestASRDataSegEdgeCases:
    """测试 ASRDataSeg 边缘情况"""

    def test_zero_duration_segment(self):
        """测试零时长字幕段"""
        seg = ASRDataSeg("Instant", 1000, 1000)
        assert seg.start_time == seg.end_time
        timestamp = seg.to_srt_ts()
        assert timestamp == "00:00:01,000 --> 00:00:01,000"

    def test_negative_duration(self):
        """测试倒序时间戳(start > end)"""
        seg = ASRDataSeg("Reversed", 2000, 1000)
        assert seg.start_time > seg.end_time  # 不应自动修正

    def test_very_long_timestamp(self):
        """测试超长时间戳(超过24小时)"""
        seg = ASRDataSeg("Long", 90000000, 90001000)  # 25小时
        timestamp = seg.to_srt_ts()
        assert "25:00:00,000" in timestamp

    def test_unicode_text_extreme(self):
        """测试极端Unicode文本"""
        # Emoji + 中文 + 日文 + 韩文 + 阿拉伯文
        text = "😀你好こんにちは안녕مرحبا"
        seg = ASRDataSeg(text, 0, 1000)
        assert seg.text == text

    def test_empty_translation(self):
        """测试空翻译与无翻译的区别"""
        seg1 = ASRDataSeg("Test", 0, 1000)
        seg2 = ASRDataSeg("Test", 0, 1000, translated_text="")
        assert seg1.translated_text == seg2.translated_text == ""

    def test_multiline_text(self):
        """测试多行文本"""
        text = "Line 1\nLine 2\nLine 3"
        seg = ASRDataSeg(text, 0, 1000)
        assert "\n" in seg.text
        assert seg.text.count("\n") == 2


class TestASRDataEdgeCases:
    """测试 ASRData 边缘情况"""

    def test_mixed_empty_and_whitespace(self):
        """测试混合空字符串和纯空格"""
        segments = [
            ASRDataSeg("Valid", 0, 1000),
            ASRDataSeg("", 1000, 2000),
            ASRDataSeg("   ", 2000, 3000),
            ASRDataSeg("\t\n", 3000, 4000),
            ASRDataSeg("  Valid  ", 4000, 5000),  # 前后空格应保留
        ]
        asr_data = ASRData(segments)
        assert len(asr_data) == 2
        assert asr_data.segments[1].text == "  Valid  "

    def test_overlapping_timestamps(self):
        """测试重叠的时间戳"""
        segments = [
            ASRDataSeg("First", 0, 2000),
            ASRDataSeg("Overlap", 1000, 3000),  # 重叠
            ASRDataSeg("Third", 2500, 4000),
        ]
        asr_data = ASRData(segments)
        # 应按start_time排序，但不修正重叠
        assert asr_data.segments[0].text == "First"
        assert asr_data.segments[1].text == "Overlap"

    def test_unsorted_large_dataset(self):
        """测试大量乱序数据"""
        segments = [ASRDataSeg(f"Text{i}", i * 1000, (i + 1) * 1000) for i in range(1000, 0, -1)]
        asr_data = ASRData(segments)
        # 应该正确排序
        for i in range(len(asr_data) - 1):
            assert asr_data.segments[i].start_time <= asr_data.segments[i + 1].start_time

    def test_duplicate_timestamps(self):
        """测试完全相同的时间戳"""
        segments = [
            ASRDataSeg("First", 1000, 2000),
            ASRDataSeg("Second", 1000, 2000),
            ASRDataSeg("Third", 1000, 2000),
        ]
        asr_data = ASRData(segments)
        assert len(asr_data) == 3  # 都应保留

    def test_single_segment(self):
        """测试单个字幕段的边界情况"""
        segments = [ASRDataSeg("Only", 0, 1000)]
        asr_data = ASRData(segments)
        # 各种操作不应崩溃
        asr_data.optimize_timing()
        assert len(asr_data) == 1

    def test_clip_to_media_duration_drops_overflow_and_clamps_tail(self):
        data = ASRData(
            [
                ASRDataSeg("Valid", 8_000, 9_000),
                ASRDataSeg("Tail", 9_500, 10_500),
                ASRDataSeg("Hallucination", 12_000, 13_000),
            ]
        )

        data.clip_to_media_duration(10_000)

        assert [(segment.text, segment.start_time, segment.end_time) for segment in data] == [
            ("Valid", 8_000, 9_000),
            ("Tail", 9_500, 10_000),
        ]


class TestWordTimestampEdgeCases:
    """测试词级时间戳检测边缘情况"""

    def test_exactly_80_percent_threshold(self):
        """测试恰好80%阈值"""
        # 10个片段，8个词级，2个句子级
        segments = [ASRDataSeg(f"word{i}", i * 100, (i + 1) * 100) for i in range(8)]
        segments.extend(
            [
                ASRDataSeg("This is sentence", 800, 900),
                ASRDataSeg("Another sentence", 900, 1000),
            ]
        )
        asr_data = ASRData(segments)
        assert asr_data.is_word_timestamp()  # 80% 应该通过

    def test_79_percent_below_threshold(self):
        """测试略低于80%阈值"""
        # 10个片段，7个词级，3个句子级
        segments = [ASRDataSeg(f"word{i}", i * 100, (i + 1) * 100) for i in range(7)]
        segments.extend(
            [
                ASRDataSeg("This is sentence", 700, 800),
                ASRDataSeg("Another sentence", 800, 900),
                ASRDataSeg("Third sentence", 900, 1000),
            ]
        )
        asr_data = ASRData(segments)
        assert not asr_data.is_word_timestamp()  # 70% 不应通过

    def test_mixed_cjk_latin_single_chars(self):
        """测试混合CJK和拉丁单字符"""
        segments = [
            ASRDataSeg("你", 0, 100),  # CJK单字
            ASRDataSeg("好", 100, 200),
            ASRDataSeg("a", 200, 300),  # 拉丁单字符
            ASRDataSeg("b", 300, 400),
        ]
        asr_data = ASRData(segments)
        assert asr_data.is_word_timestamp()

    def test_three_char_cjk(self):
        """测试3字符CJK(边界情况)"""
        segments = [ASRDataSeg("你好吗", 0, 1000)]  # 3个字符，不是词级
        asr_data = ASRData(segments)
        assert not asr_data.is_word_timestamp()

    def test_explicit_sentence_granularity_overrides_single_word_heuristic(self):
        segments = [
            ASRDataSeg(
                "Yes.",
                0,
                500,
                timestamp_granularity="sentence",
                timing_source="native",
            ),
            ASRDataSeg(
                "No.",
                700,
                1100,
                timestamp_granularity="sentence",
                timing_source="native",
            ),
        ]

        asr_data = ASRData(segments)

        assert asr_data.granularity == "sentence"
        assert asr_data.timing_source == "native"
        assert not asr_data.is_word_timestamp()

    def test_explicit_word_granularity_retains_atomic_timing(self):
        segment = ASRDataSeg(
            "New York",
            100,
            700,
            words=[ASRWord("New York", 100, 700, timing_source="forced_alignment")],
            timestamp_granularity="word",
            timing_source="forced_alignment",
        )

        asr_data = ASRData([segment])

        assert asr_data.is_word_timestamp()
        assert asr_data.granularity == "word"
        assert asr_data.timing_source == "forced_alignment"
        assert asr_data.segments[0].words[0].start_time == 100

    def test_imported_word_srt_rebuilds_atomic_metadata(self):
        source = """1
00:00:00,100 --> 00:00:00,300
Hello

2
00:00:00,350 --> 00:00:00,700
world
"""

        asr_data = ASRData.from_srt(source)

        assert asr_data.granularity == "word"
        assert asr_data.timing_source == "imported"
        assert [word.text for seg in asr_data.segments for word in seg.words] == [
            "Hello",
            "world",
        ]

    def test_imported_sentence_srt_is_explicitly_sentence_level(self):
        source = """1
00:00:00,100 --> 00:00:01,300
Hello world from SubForge.

2
00:00:01,500 --> 00:00:02,700
This remains a sentence.
"""

        asr_data = ASRData.from_srt(source)

        assert asr_data.granularity == "sentence"
        assert asr_data.timing_source == "imported"
        assert not any(seg.words for seg in asr_data.segments)


class TestSplitToWordsEdgeCases:
    """测试分词边缘情况"""

    def test_split_empty_text(self):
        """测试空文本分词"""
        segments = [ASRDataSeg("", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        assert len(asr_data.segments) == 0

    def test_split_only_punctuation(self):
        """测试纯标点分词"""
        segments = [ASRDataSeg("..., !!!", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        assert len(asr_data.segments) == 0  # 标点不应匹配

    def test_split_very_long_word(self):
        """测试超长单词"""
        long_word = "a" * 1000
        segments = [ASRDataSeg(long_word, 0, 10000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == long_word

    def test_split_mixed_scripts(self):
        """测试混合多种文字系统"""
        # 拉丁+中文+日文+韩文+阿拉伯文+俄文
        text = "Hello你好こんにちは안녕مرحباПривет"
        segments = [ASRDataSeg(text, 0, 7000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        # 应该正确分割各种文字
        assert len(asr_data.segments) > 5
        texts = [seg.text for seg in asr_data.segments]
        assert "Hello" in texts
        assert "Привет" in texts

    def test_split_numbers_and_words(self):
        """测试数字和单词混合"""
        segments = [ASRDataSeg("version 3.14 build 2024", 0, 3000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        texts = [seg.text for seg in asr_data.segments]
        assert "version" in texts
        assert "3" in texts or "14" in texts  # 数字应被分开
        assert "build" in texts
        assert "2024" in texts

    def test_split_thai_with_combining_chars(self):
        """测试泰文带组合字符"""
        thai_text = "สวัสดี"  # 泰文 "你好"
        segments = [ASRDataSeg(thai_text, 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()
        assert len(asr_data.segments) > 0  # 应该能匹配泰文

    def test_split_zero_duration_distribution(self):
        """测试零时长的时间分配"""
        segments = [ASRDataSeg("Hello world", 1000, 1000)]
        asr_data = ASRData(segments)
        asr_data.split_to_word_segments()

        assert asr_data.granularity == "word"
        assert asr_data.timing_source == "estimated"
        assert all(seg.words for seg in asr_data.segments)
        assert all(seg.words[0].timing_source == "estimated" for seg in asr_data.segments)
        # 零时长应该不崩溃
        assert all(seg.start_time == 1000 for seg in asr_data.segments)
        assert all(seg.end_time == 1000 for seg in asr_data.segments)

    def test_split_preserves_speaker_metadata(self):
        asr_data = ASRData([ASRDataSeg("Hello there", 0, 1000, speaker_id="Speaker 2")])

        asr_data.split_to_word_segments()

        assert [segment.speaker_id for segment in asr_data.segments] == [
            "Speaker 2",
            "Speaker 2",
        ]


class TestMergeEdgeCases:
    """测试合并边缘情况"""

    def test_merge_single_segment(self):
        """测试合并单个片段(自己和自己)"""
        segments = [ASRDataSeg("Only", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.merge_segments(0, 0)
        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "Only"

    def test_merge_all_segments(self):
        """测试合并所有片段"""
        segments = [ASRDataSeg(f"T{i}", i * 100, (i + 1) * 100) for i in range(10)]
        asr_data = ASRData(segments)
        asr_data.merge_segments(0, 9)
        assert len(asr_data.segments) == 1
        assert "T0" in asr_data.segments[0].text
        assert "T9" in asr_data.segments[0].text

    def test_merge_invalid_indices(self):
        """测试无效的合并索引"""
        segments = [ASRDataSeg("A", 0, 1000), ASRDataSeg("B", 1000, 2000)]
        asr_data = ASRData(segments)

        with pytest.raises(IndexError):
            asr_data.merge_segments(-1, 1)  # 负索引
        with pytest.raises(IndexError):
            asr_data.merge_segments(0, 5)  # 超出范围
        with pytest.raises(IndexError):
            asr_data.merge_segments(1, 0)  # start > end

    def test_merge_with_next_at_boundary(self):
        """测试在边界位置合并"""
        segments = [ASRDataSeg("Only", 0, 1000)]
        asr_data = ASRData(segments)

        with pytest.raises(IndexError):
            asr_data.merge_with_next_segment(0)  # 没有下一个

    def test_merge_with_unicode(self):
        """测试合并Unicode文本"""
        segments = [
            ASRDataSeg("😀你好", 0, 1000),
            ASRDataSeg("🌍world", 1000, 2000),
        ]
        asr_data = ASRData(segments)
        asr_data.merge_with_next_segment(0)
        assert "😀" in asr_data.segments[0].text
        assert "🌍" in asr_data.segments[0].text


class TestOptimizeTimingEdgeCases:
    """测试时间优化边缘情况"""

    def test_optimize_negative_gap(self):
        """测试负间隔(重叠)"""
        segments = [
            ASRDataSeg("First", 0, 2000),
            ASRDataSeg("Overlap", 1500, 3000),  # 重叠500ms
        ]
        asr_data = ASRData(segments)
        asr_data.optimize_timing()
        # 负间隔不应优化(或根据实现调整)
        assert asr_data.segments[0].end_time == 2000

    def test_optimize_exact_threshold(self):
        """测试恰好在阈值边界"""
        segments = [
            ASRDataSeg("First sentence", 0, 1000),
            ASRDataSeg("Second sentence", 2000, 3000),  # 恰好1000ms gap
        ]
        asr_data = ASRData(segments)
        asr_data.optimize_timing(threshold_ms=1000)
        # 恰好等于阈值不优化(需要 < threshold)
        gap = asr_data.segments[1].start_time - asr_data.segments[0].end_time
        assert gap == 1000  # 应该保持不变

    def test_optimize_word_level_no_change(self):
        """测试词级时间戳不优化"""
        segments = [
            ASRDataSeg("Word1", 0, 500),
            ASRDataSeg("Word2", 1000, 1500),
        ]
        asr_data = ASRData(segments)
        original_end = asr_data.segments[0].end_time

        asr_data.optimize_timing()
        # 词级应该跳过优化
        assert asr_data.segments[0].end_time == original_end


class TestFixBoundaryOverlaps:
    """测试字幕时间轴重叠修复"""

    def test_splits_adjacent_overlap_at_midpoint(self):
        segments = [
            ASRDataSeg("First", 0, 2000),
            ASRDataSeg("Second", 1500, 3000),
        ]
        asr_data = ASRData(segments)

        asr_data.fix_boundary_overlaps()

        assert asr_data.segments[0].end_time == 1750
        assert asr_data.segments[1].start_time == 1750
        assert asr_data.segments[0].end_time <= asr_data.segments[1].start_time

    def test_handles_contained_segment_without_leaving_overlap(self):
        segments = [
            ASRDataSeg("Long segment", 0, 10000),
            ASRDataSeg("Contained", 100, 200),
            ASRDataSeg("After", 300, 1000),
        ]
        asr_data = ASRData(segments)

        asr_data.fix_boundary_overlaps()

        for current, next_seg in zip(asr_data.segments, asr_data.segments[1:]):
            assert current.end_time <= next_seg.start_time
            assert current.start_time <= current.end_time
            assert next_seg.start_time <= next_seg.end_time

    def test_to_srt_normalizes_overlaps_before_output(self):
        segments = [
            ASRDataSeg("First", 0, 2000),
            ASRDataSeg("Second", 1500, 3000),
        ]
        asr_data = ASRData(segments)

        srt = asr_data.to_srt()

        assert "00:00:01,750 --> 00:00:01,750" not in srt
        assert "00:00:00,000 --> 00:00:01,750" in srt
        assert "00:00:01,750 --> 00:00:03,000" in srt

    def test_save_normalizes_json_timestamps(self, tmp_path):
        segments = [
            ASRDataSeg("First", 0, 2000),
            ASRDataSeg("Second", 1500, 3000),
        ]
        asr_data = ASRData(segments)
        output_path = tmp_path / "result.json"

        asr_data.save(str(output_path))
        loaded = ASRData.from_subtitle_file(str(output_path))

        assert loaded.segments[0].end_time <= loaded.segments[1].start_time


class TestDeduplicateAdjacentText:
    """测试 VAD/ASR 边界处的相邻重复文本清理"""

    def test_removes_short_prefix_fragment_contained_in_next_segment(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Hey everyone, welcome back to", 5200, 6480),
                ASRDataSeg(
                    "Hey everyone, welcome back to Topher Drives where today",
                    6800,
                    11380,
                ),
            ]
        )

        asr_data.deduplicate_adjacent_text()

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == (
            "Hey everyone, welcome back to Topher Drives where today"
        )
        assert asr_data.segments[0].start_time == 6800

    def test_removes_exact_adjacent_duplicate(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Stability something.", 702300, 703550),
                ASRDataSeg("Stability something.", 703878, 704428),
                ASRDataSeg("Next thought.", 705000, 706000),
            ]
        )

        asr_data.deduplicate_adjacent_text()

        assert [seg.text for seg in asr_data.segments] == [
            "Stability something.",
            "Next thought.",
        ]

    def test_trims_duplicate_prefix_from_following_segment(self):
        asr_data = ASRData(
            [
                ASRDataSeg("the steering is honestly it's not as quick", 590100, 594200),
                ASRDataSeg(
                    "it's not as quick as some newer cars",
                    594650,
                    598000,
                ),
            ]
        )

        asr_data.deduplicate_adjacent_text()

        assert len(asr_data.segments) == 2
        assert asr_data.segments[1].text == "as some newer cars"
        assert asr_data.segments[1].start_time > 594650

    def test_removes_compact_duplicate_with_spacing_difference(self):
        asr_data = ASRData(
            [
                ASRDataSeg("And up front we have a moon roof as well.", 346450, 348110),
                ASRDataSeg("And up front we have a moonroof as well.", 348214, 349450),
            ]
        )

        asr_data.deduplicate_adjacent_text()

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "And up front we have a moonroof as well."

    def test_keeps_distant_repeated_phrase(self):
        asr_data = ASRData(
            [
                ASRDataSeg("How about that?", 0, 1000),
                ASRDataSeg("How about that?", 5000, 6000),
            ]
        )

        asr_data.deduplicate_adjacent_text()

        assert len(asr_data.segments) == 2


class TestDeduplicateAlignmentEchoes:
    @staticmethod
    def _word(text: str, start: int, end: int, speaker: str = "Speaker 1") -> ASRDataSeg:
        return ASRDataSeg(
            text,
            start,
            end,
            speaker_id=speaker,
            timestamp_granularity="word",
            timing_source="forced_alignment",
        )

    def test_removes_suffix_and_exact_word_echoes(self):
        asr_data = ASRData(
            [
                self._word("become.", 0, 200),
                self._word("come.", 220, 500),
                self._word("building.", 1000, 1250),
                self._word("building.", 1250, 1500),
                self._word("off-plan.", 2000, 2250),
                self._word("plan,", 2250, 2500),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert [segment.text for segment in asr_data.segments] == [
            "become.",
            "building.",
            "off-plan.",
        ]
        assert [segment.end_time for segment in asr_data.segments] == [500, 1500, 2500]
        assert [segment.words[-1].end_time for segment in asr_data.segments] == [
            500,
            1500,
            2500,
        ]

    def test_preserves_tail_of_short_exact_alignment_echo(self):
        asr_data = ASRData(
            [
                self._word("that.", 0, 100),
                self._word("that.", 120, 720),
                self._word("Next", 1500, 1750),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert [segment.text for segment in asr_data.segments] == ["that.", "Next"]
        assert asr_data.segments[0].end_time == 720
        assert asr_data.segments[0].words[-1].end_time == 720

    def test_keeps_normal_suffix_across_a_sentence_boundary(self):
        asr_data = ASRData(
            [
                self._word("thin.", 0, 200),
                self._word("In", 260, 400),
                self._word("profile,", 420, 700),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert [segment.text for segment in asr_data.segments] == ["thin.", "In", "profile,"]

    def test_keeps_echo_like_words_across_speakers_or_a_real_gap(self):
        asr_data = ASRData(
            [
                self._word("basis.", 0, 200, "Speaker 1"),
                self._word("basis.", 200, 500, "Speaker 2"),
                self._word("become.", 1000, 1200, "Speaker 1"),
                self._word("come.", 1500, 1800, "Speaker 1"),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert len(asr_data.segments) == 4

    def test_removes_lowercase_two_homophone_echo_after_number(self):
        asr_data = ASRData(
            [
                self._word("432.", 0, 300),
                self._word("too.", 380, 700),
                self._word("These", 1200, 1450),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert [segment.text for segment in asr_data.segments] == ["432.", "These"]

    def test_keeps_capitalized_two_as_a_new_counted_item(self):
        asr_data = ASRData(
            [
                self._word("432.", 0, 300),
                self._word("Two.", 380, 700),
            ],
            granularity="word",
        )

        asr_data.deduplicate_alignment_echoes()

        assert [segment.text for segment in asr_data.segments] == ["432.", "Two."]


class TestMergeSentenceFragments:
    """测试 whisper.cpp 句子中途切分的保守合并"""

    def test_merges_short_sentence_tail_fragment(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "the hot version of the third-generation TL, and this is a car that I've always wanted",
                    18600,
                    23690,
                ),
                ASRDataSeg("to drive.", 23690, 25400),
                ASRDataSeg(
                    "It's always been interesting to me because I was always a Lexus guy.",
                    25890,
                    28110,
                ),
            ]
        )

        asr_data.merge_sentence_fragments()

        assert [seg.text for seg in asr_data.segments] == [
            "the hot version of the third-generation TL, and this is a car that I've always wanted to drive.",
            "It's always been interesting to me because I was always a Lexus guy.",
        ]
        assert asr_data.segments[0].start_time == 18600
        assert asr_data.segments[0].end_time == 25400
        assert asr_data.segments[1].start_time == 25890

    def test_merges_short_predicate_fragment(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "one of these when I was shopping around, but I kind of wish I would have because this thing",
                    41250,
                    44250,
                ),
                ASRDataSeg("seems pretty compelling.", 44650, 45400),
            ]
        )

        asr_data.merge_sentence_fragments()

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text.endswith("this thing seems pretty compelling.")
        assert asr_data.segments[0].end_time == 45400

    def test_does_not_merge_after_terminal_punctuation(self):
        asr_data = ASRData(
            [
                ASRDataSeg("This is complete.", 0, 1200),
                ASRDataSeg("The next sentence starts here.", 1400, 2600),
            ]
        )

        asr_data.merge_sentence_fragments()

        assert len(asr_data.segments) == 2

    def test_does_not_merge_across_long_pause(self):
        asr_data = ASRData(
            [
                ASRDataSeg("This phrase is still", 0, 1200),
                ASRDataSeg("going after a pause.", 2500, 3900),
            ]
        )

        asr_data.merge_sentence_fragments()

        assert len(asr_data.segments) == 2

    def test_does_not_merge_when_combined_caption_is_too_long(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "When I was in college, I drove a 2003 Lexus IS300 with a five-speed manual, and I still",
                    28110,
                    34050,
                ),
                ASRDataSeg(
                    "have that car, and the Acura TL's prime competitor is the Lexus IS, and I never really considered",
                    34410,
                    40850,
                ),
            ]
        )

        asr_data.merge_sentence_fragments()

        assert len(asr_data.segments) == 2


class TestSpeechVadTimingRefinement:
    """测试使用语音 VAD 修剪字幕尾部无语音覆盖"""

    def test_trims_tail_when_vad_and_text_rate_agree(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Let's pop the hood and show you this J35A8.", 213400, 219400),
            ]
        )

        asr_data.refine_timing_with_speech_segments([(213420, 216560)])

        assert asr_data.segments[0].end_time < 218000
        assert asr_data.segments[0].end_time >= 216500

    def test_trims_leading_silence_when_vad_has_clear_speech_start(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Hey everyone, welcome back to Topher Drives.", 0, 9240),
            ]
        )

        asr_data.refine_timing_with_speech_segments([(7450, 9233)])

        assert asr_data.segments[0].start_time >= 7000
        assert asr_data.segments[0].end_time == 9240

    def test_trims_short_fragment_tail(self):
        asr_data = ASRData([ASRDataSeg("to pop the trunk.", 179060, 182450)])

        asr_data.refine_timing_with_speech_segments([(179100, 180350)])

        assert asr_data.segments[0].end_time <= 180700

    def test_does_not_trim_when_speech_reaches_segment_end(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "This line is long enough and speech reaches the end",
                    1000,
                    5000,
                ),
            ]
        )

        asr_data.refine_timing_with_speech_segments([(1000, 4700)])

        assert asr_data.segments[0].end_time == 5000

    def test_does_not_trim_without_vad_overlap(self):
        asr_data = ASRData([ASRDataSeg("Possible missed speech", 1000, 5000)])

        asr_data.refine_timing_with_speech_segments([])

        assert asr_data.segments[0].end_time == 5000


class TestAudioEnergyPauseRestore:
    """测试从音频静音恢复字幕间隔"""

    @staticmethod
    def _tone(duration_ms=1000):
        from pydub.generators import Sine

        return Sine(440).to_audio_segment(duration=duration_ms).apply_gain(-3)

    def test_filter_hallucinations_keeps_word_level_intro_tokens(self, tmp_path):
        from pydub import AudioSegment

        audio_path = tmp_path / "quiet_intro.wav"
        AudioSegment.silent(duration=3000).export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg("Hey", 10, 580),
                ASRDataSeg("everyone,", 780, 3210),
                ASRDataSeg("welcome", 3210, 4940),
                ASRDataSeg("back", 5370, 5930),
                ASRDataSeg("to", 5930, 6080),
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        assert [seg.text for seg in asr_data.segments] == [
            "Hey",
            "everyone,",
            "welcome",
            "back",
            "to",
        ]

    def test_filter_hallucinations_removes_isolated_word_run_rejected_by_vad(self):
        asr_data = ASRData(
            [
                ASRDataSeg("!", 6042, 6062),
                ASRDataSeg("Today,", 28600, 28783),
                ASRDataSeg("I", 28803, 28823),
                ASRDataSeg("will", 28864, 28945),
                ASRDataSeg("introduce", 28966, 29229),
                ASRDataSeg("a", 29250, 29270),
                ASRDataSeg("recipe", 29290, 29412),
                ASRDataSeg("for", 29452, 29513),
                ASRDataSeg("a", 29594, 29655),
                ASRDataSeg("delicious", 29716, 29980),
                ASRDataSeg("Hey", 68581, 68721),
                ASRDataSeg("everyone,", 68762, 69143),
            ],
            granularity="word",
        )

        asr_data.filter_hallucinations(
            speech_segments=[(27840, 28880), (68656, 69856)],
            strict_speech_segments=[(68672, 69824)],
            corroborating_speech_segments=[(69216, 69920)],
            media_duration_ms=90000,
        )

        assert [(seg.text, seg.start_time, seg.end_time) for seg in asr_data.segments] == [
            ("Hey", 68581, 68721),
            ("everyone,", 68762, 69143),
        ]

    def test_filter_hallucinations_keeps_isolated_words_with_corroborating_speech(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Quiet", 10000, 10300),
                ASRDataSeg("but", 10320, 10480),
                ASRDataSeg("real", 10500, 10800),
            ],
            granularity="word",
        )

        asr_data.filter_hallucinations(
            speech_segments=[(10000, 10100)],
            strict_speech_segments=[],
            corroborating_speech_segments=[(10100, 10600)],
            media_duration_ms=20000,
        )

        assert [(seg.text, seg.start_time, seg.end_time) for seg in asr_data.segments] == [
            ("Quiet", 10000, 10300),
            ("but", 10320, 10480),
            ("real", 10500, 10800),
        ]

    def test_filter_hallucinations_removes_short_run_in_confirmed_long_nonspeech(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Previous", 0, 400),
                ASRDataSeg("speech", 420, 900),
                ASRDataSeg("Thank", 12000, 12200),
                ASRDataSeg("you.", 12220, 12480),
                ASRDataSeg("Next", 24000, 24300),
                ASRDataSeg("sentence", 24320, 24800),
            ],
            granularity="word",
        )

        asr_data.filter_hallucinations(
            speech_segments=[(0, 900), (24000, 24800)],
            strict_speech_segments=[(0, 900), (24000, 24800)],
            corroborating_speech_segments=[(0, 900), (24000, 24800)],
            media_duration_ms=30000,
        )

        assert [segment.text for segment in asr_data.segments] == [
            "Previous",
            "speech",
            "Next",
            "sentence",
        ]

    def test_filter_hallucinations_keeps_short_isolated_real_utterance(self):
        asr_data = ASRData(
            [
                ASRDataSeg("Previous", 0, 400),
                ASRDataSeg("speech", 420, 900),
                ASRDataSeg("Thank", 12000, 12200),
                ASRDataSeg("you.", 12220, 12480),
                ASRDataSeg("Next", 24000, 24300),
                ASRDataSeg("sentence", 24320, 24800),
            ],
            granularity="word",
        )

        asr_data.filter_hallucinations(
            speech_segments=[(0, 900), (12080, 12320), (24000, 24800)],
            strict_speech_segments=[(0, 900), (24000, 24800)],
            corroborating_speech_segments=[(0, 900), (24000, 24800)],
            media_duration_ms=30000,
        )

        assert [segment.text for segment in asr_data.segments] == [
            "Previous",
            "speech",
            "Thank",
            "you.",
            "Next",
            "sentence",
        ]

    def test_filter_hallucinations_keeps_punctuation_attached_to_real_words(self):
        asr_data = ASRData(
            [
                ASRDataSeg("That", 10000, 10300),
                ASRDataSeg("works", 10320, 10600),
                ASRDataSeg("!", 10610, 10630),
            ],
            granularity="word",
        )

        asr_data.filter_hallucinations(
            speech_segments=[(10000, 10600)],
            strict_speech_segments=[(10000, 10600)],
            corroborating_speech_segments=[(10000, 10600)],
            media_duration_ms=20000,
        )

        assert [segment.text for segment in asr_data.segments] == ["That", "works", "!"]

    def test_filter_hallucinations_restores_zero_gap_pause_from_audio(self, tmp_path):
        from pydub import AudioSegment

        tone = self._tone()
        silence = AudioSegment.silent(duration=800)
        audio = tone + silence + tone
        audio_path = tmp_path / "pause.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg("Before pause", 0, 1400),
                ASRDataSeg("After pause", 1400, 2800),
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        gap = asr_data.segments[1].start_time - asr_data.segments[0].end_time
        assert gap >= 250
        assert 900 <= asr_data.segments[0].end_time <= 1100
        assert 1700 <= asr_data.segments[1].start_time <= 1900

    def test_filter_hallucinations_splits_segment_on_internal_silence(self, tmp_path):
        from pydub import AudioSegment

        audio = self._tone() + AudioSegment.silent(duration=800) + self._tone()
        audio_path = tmp_path / "internal_pause.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg(
                    "This phrase should split across the internal pause",
                    0,
                    2800,
                )
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 2
        assert asr_data.segments[0].end_time <= 1100
        assert asr_data.segments[1].start_time >= 1700
        assert asr_data.segments[1].start_time - asr_data.segments[0].end_time >= 600
        assert asr_data.segments[0].text
        assert asr_data.segments[1].text

    def test_filter_hallucinations_splits_multiple_active_clusters_before_trimming(self, tmp_path):
        from pydub import AudioSegment

        audio = (
            self._tone(duration_ms=3500)
            + AudioSegment.silent(duration=3500)
            + self._tone(duration_ms=1200)
        )
        audio_path = tmp_path / "two_active_clusters.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData([ASRDataSeg("Yeah. That is zesty.", 0, 8200)])

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 2
        assert asr_data.segments[0].text == "Yeah."
        assert "That is zesty" in asr_data.segments[1].text
        assert asr_data.segments[1].start_time >= 6500

    def test_filter_hallucinations_removes_overlong_short_segment(self, tmp_path):
        audio = self._tone(duration_ms=12000)
        audio_path = tmp_path / "active_noise.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData([ASRDataSeg("Still very clear.", 0, 12000)])

        asr_data.filter_hallucinations(str(audio_path))

        assert asr_data.segments == []

    def test_filter_hallucinations_trims_overlong_segment_to_active_cluster(self, tmp_path):
        from pydub import AudioSegment

        audio = self._tone() + AudioSegment.silent(duration=6000)
        audio_path = tmp_path / "trailing_silence.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData([ASRDataSeg("First driving impressions today now", 0, 7000)])

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].end_time <= 1200

    def test_filter_hallucinations_caps_overlong_segment_in_active_noise(self, tmp_path):
        audio = self._tone(duration_ms=12000)
        audio_path = tmp_path / "active_long.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg(
                    "Volume controls here to the left and we are going left now",
                    0,
                    12000,
                )
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].end_time <= 7500

    def test_filter_hallucinations_trims_sentence_tail_in_active_noise(self, tmp_path):
        audio = self._tone(duration_ms=6000)
        audio_path = tmp_path / "active_sentence_tail.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg(
                    "have to look up into a head-up display to see what I'm pressing.",
                    0,
                    5300,
                )
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].end_time < 5300
        assert asr_data.segments[0].end_time >= 3000

    def test_filter_hallucinations_does_not_trim_non_terminal_active_noise(self, tmp_path):
        audio = self._tone(duration_ms=6000)
        audio_path = tmp_path / "active_non_terminal.wav"
        audio.export(audio_path, format="wav").close()

        asr_data = ASRData(
            [
                ASRDataSeg(
                    "to place on the road it is still narrow enough to fit through",
                    0,
                    5300,
                )
            ]
        )

        asr_data.filter_hallucinations(str(audio_path))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].end_time == 5300


class TestRemovePunctuationEdgeCases:
    """测试移除标点边缘情况"""

    def test_remove_multiple_punctuation(self):
        """测试连续多个标点"""
        segments = [ASRDataSeg("你好，，，。。。", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.remove_punctuation()
        assert asr_data.segments[0].text == "你好"

    def test_remove_punctuation_only(self):
        """测试纯标点文本"""
        segments = [ASRDataSeg("，。，。", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.remove_punctuation()
        assert asr_data.segments[0].text == ""

    def test_remove_punctuation_middle(self):
        """测试中间的标点不移除"""
        segments = [ASRDataSeg("你好，世界。", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.remove_punctuation()
        assert asr_data.segments[0].text == "你好，世界"  # 只删尾部

    def test_remove_non_chinese_punctuation(self):
        """测试非中文标点不移除"""
        segments = [ASRDataSeg("Hello, world!", 0, 1000)]
        asr_data = ASRData(segments)
        asr_data.remove_punctuation()
        assert asr_data.segments[0].text == "Hello, world!"  # 不变


class TestChineseTranslationPunctuationFinalization:
    def test_replaces_chinese_punctuation_only_in_translated_lines(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "Hello, world.",
                    0,
                    1000,
                    translated_text="你好，世界。 这是测试，完成。",
                ),
                ASRDataSeg(
                    "中文原文，保留。",
                    1000,
                    2000,
                    translated_text="English, translation.",
                ),
            ]
        )

        asr_data.replace_chinese_translation_punctuation()

        assert asr_data.segments[0].text == "Hello, world."
        assert asr_data.segments[0].translated_text == "你好 世界 这是测试 完成"
        assert asr_data.segments[1].text == "中文原文，保留。"
        assert asr_data.segments[1].translated_text == "English, translation."

    def test_replaces_ascii_sentence_punctuation_but_preserves_identifiers(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "Technical terms.",
                    0,
                    1000,
                    translated_text="REM睡眠,很常见。 版本3.5见himss.com。",
                ),
                ASRDataSeg(
                    "Nightmare Obscura.",
                    1000,
                    2000,
                    translated_text="《Nightmare Obscura》。",
                ),
            ]
        )

        asr_data.replace_chinese_translation_punctuation()

        assert asr_data.segments[0].translated_text == "REM睡眠 很常见 版本3.5见himss.com"
        assert asr_data.segments[1].translated_text == "《Nightmare Obscura》"

    def test_round_trip_latin_only_translated_title(self):
        parsed = ASRData.from_srt(
            "1\n00:00:00,000 --> 00:00:02,000\n《Nightmare Obscura》\nNightmare Obscura.\n"
        )

        assert parsed.segments[0].text == "Nightmare Obscura."
        assert parsed.segments[0].translated_text == "《Nightmare Obscura》"

    def test_removes_only_trailing_enumeration_comma(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "A, B, and C.",
                    0,
                    1000,
                    translated_text="甲、乙、丙、",
                )
            ]
        )

        asr_data.replace_chinese_translation_punctuation()

        assert asr_data.segments[0].translated_text == "甲、乙、丙"

    def test_finalization_is_idempotent(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "Technical terms.",
                    0,
                    1000,
                    translated_text="版本3.5，详见himss.com。",
                )
            ]
        )

        asr_data.replace_chinese_translation_punctuation()
        finalized = asr_data.segments[0].translated_text
        asr_data.replace_chinese_translation_punctuation()

        assert finalized == "版本3.5 详见himss.com"
        assert asr_data.segments[0].translated_text == finalized


class TestFormatConversionEdgeCases:
    """测试格式转换边缘情况"""

    def test_srt_layout_modes_all(self):
        """测试所有SRT布局模式"""
        from subforge.core.entities import SubtitleLayoutEnum

        segments = [ASRDataSeg("Hello", 0, 1000, translated_text="你好")]
        asr_data = ASRData(segments)

        srt1 = asr_data.to_srt(layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP)
        assert "Hello\n你好" in srt1

        srt2 = asr_data.to_srt(layout=SubtitleLayoutEnum.TRANSLATE_ON_TOP)
        assert "你好\nHello" in srt2

        srt3 = asr_data.to_srt(layout=SubtitleLayoutEnum.ONLY_ORIGINAL)
        assert "Hello" in srt3
        assert "你好" not in srt3

        srt4 = asr_data.to_srt(layout=SubtitleLayoutEnum.ONLY_TRANSLATE)
        assert "你好" in srt4

    def test_srt_no_translation_all_layouts(self):
        """测试无翻译时的所有布局"""
        segments = [ASRDataSeg("Hello", 0, 1000)]
        asr_data = ASRData(segments)

        for layout in ["原文在上", "译文在上", "仅原文", "仅译文"]:
            srt = asr_data.to_srt(layout=layout)
            assert "Hello" in srt  # 所有模式都应显示原文

    def test_srt_speaker_styles_keep_labels_internal(self):
        asr_data = ASRData(
            [
                ASRDataSeg(
                    "Hello",
                    0,
                    1000,
                    translated_text="你好",
                    speaker_id="Speaker 2",
                )
            ]
        )

        labeled = asr_data.to_srt(speaker_style="label")
        dialogue = asr_data.to_srt(speaker_style="dash")
        hidden = asr_data.to_srt(speaker_style="none")

        assert "[Speaker 2] Hello" in labeled
        assert "- Hello" in dialogue
        assert "- 你好" in dialogue
        assert "Speaker 2" not in dialogue
        assert "Speaker 2" not in hidden
        assert "- Hello" not in hidden

    def test_srt_dash_style_does_not_duplicate_existing_marker(self):
        asr_data = ASRData([ASRDataSeg("- Already marked", 0, 1000, speaker_id="Speaker 1")])

        srt = asr_data.to_srt(speaker_style="dash")

        assert "- Already marked" in srt
        assert "- - Already marked" not in srt

    def test_srt_dash_style_marks_unassigned_speech_consistently(self):
        asr_data = ASRData([ASRDataSeg("Unassigned", 0, 1000)])

        srt = asr_data.to_srt(speaker_style="dash")

        assert "- Unassigned" in srt

    def test_srt_dash_bilingual_amounts_round_trip(self):
        source = """1
00:00:00,000 --> 00:00:01,000
- 十七万刀
- $170,000

2
00:00:01,000 --> 00:00:02,000
- 100
- 100
"""

        parsed = ASRData.from_srt(source)

        assert parsed.segments[0].text == "$170,000"
        assert parsed.segments[0].translated_text == "十七万刀"
        assert parsed.segments[1].text == "100"
        assert parsed.segments[1].translated_text == "100"

    def test_srt_unmarked_bilingual_amounts_round_trip(self):
        source = """1
00:00:00,000 --> 00:00:01,000
十七万刀
$170,000

2
00:00:01,000 --> 00:00:02,000
100
100
"""

        parsed = ASRData.from_srt(source)

        assert parsed.segments[0].text == "$170,000"
        assert parsed.segments[0].translated_text == "十七万刀"
        assert parsed.segments[1].text == "100"
        assert parsed.segments[1].translated_text == "100"

    def test_json_large_dataset(self):
        """测试大数据集JSON转换"""
        segments = [ASRDataSeg(f"Text{i}", i * 1000, (i + 1) * 1000) for i in range(1000)]
        asr_data = ASRData(segments)
        json_data = asr_data.to_json()
        assert len(json_data) == 1000
        assert "1" in json_data
        assert "1000" in json_data

    def test_txt_multiline_segments(self):
        """测试多行文本转换"""
        segments = [
            ASRDataSeg("Line1\nLine2", 0, 1000),
            ASRDataSeg("Line3", 1000, 2000),
        ]
        asr_data = ASRData(segments)
        txt = asr_data.to_txt()
        assert "Line1\nLine2" in txt


class TestFileIOEdgeCases:
    """测试文件读写边缘情况"""

    def test_save_unsupported_format(self):
        """测试不支持的格式"""
        segments = [ASRDataSeg("Test", 0, 1000)]
        asr_data = ASRData(segments)

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                asr_data.save(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            ASRData.from_subtitle_file("/nonexistent/path/file.srt")

    def test_save_load_unicode_path(self):
        """测试Unicode文件路径"""
        segments = [ASRDataSeg("测试", 0, 1000)]
        asr_data = ASRData(segments)

        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_path = Path(tmpdir) / "测试文件名.srt"
            asr_data.save(str(unicode_path))
            loaded = ASRData.from_subtitle_file(str(unicode_path))
            assert loaded.segments[0].text == "测试"


class TestParseEdgeCases:
    """测试解析边缘情况"""

    def test_parse_malformed_srt(self):
        """测试畸形SRT"""
        malformed = """1
00:00:00,000 --> INVALID
Hello

2
INVALID TIMESTAMP
World
"""
        asr_data = ASRData.from_srt(malformed)
        assert len(asr_data.segments) == 0  # 应跳过无效块

    def test_parse_srt_missing_text(self):
        """测试缺少文本的SRT块"""
        srt = """1
00:00:00,000 --> 00:00:01,000

2
00:00:01,000 --> 00:00:02,000
Valid
"""
        asr_data = ASRData.from_srt(srt)
        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "Valid"

    def test_parse_srt_97_percent_translation(self):
        """测试97%翻译(低于98%阈值)"""
        # 100个块，97个有翻译
        blocks = []
        for i in range(97):
            blocks.append(
                f"{i + 1}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nText{i}\nTrans{i}\n"
            )
        for i in range(97, 100):
            blocks.append(f"{i + 1}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nText{i}\n")

        srt = "\n".join(blocks)
        asr_data = ASRData.from_srt(srt)
        # 低于98%不应识别为翻译格式
        assert not asr_data.segments[0].translated_text

    def test_parse_target_above_bilingual_with_latin_tokens(self):
        """测试中文在上且含英文车型词时仍可拆分双语"""
        srt = """1
00:00:00,000 --> 00:00:02,000
你肯定还认得出这辆1986年的梅赛德斯-奔驰420 SEL
you should also recognize this 1986 Mercedes-Benz 420 SEL
"""
        asr_data = ASRData.from_srt(srt)
        assert len(asr_data.segments) == 1
        assert (
            asr_data.segments[0].text == "you should also recognize this 1986 Mercedes-Benz 420 SEL"
        )
        assert (
            asr_data.segments[0].translated_text == "你肯定还认得出这辆1986年的梅赛德斯-奔驰420 SEL"
        )

    def test_parse_target_above_bilingual_with_single_letter_source(self):
        srt = """1
00:00:00,000 --> 00:00:01,000
D挡
D.
"""

        asr_data = ASRData.from_srt(srt)

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "D."
        assert asr_data.segments[0].translated_text == "D挡"

    def test_parse_target_above_bilingual_with_numeric_punctuation_variant(self):
        srt = """1
00:00:00,000 --> 00:00:01,000
32
32.
"""

        asr_data = ASRData.from_srt(srt)

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "32."
        assert asr_data.segments[0].translated_text == "32"

    @pytest.mark.parametrize(
        ("first_line", "second_line", "source", "translated"),
        [
            (
                "Talkiatry.com/grayarea",
                "Talkiatry.com slash grayarea.",
                "Talkiatry.com slash grayarea.",
                "Talkiatry.com/grayarea",
            ),
            (
                "patreon dot com slash vox",
                "patreon.com/vox",
                "patreon dot com slash vox",
                "patreon.com/vox",
            ),
        ],
    )
    def test_parse_language_neutral_url_bilingual_pair(
        self,
        first_line: str,
        second_line: str,
        source: str,
        translated: str,
    ):
        srt = f"""1
00:00:00,000 --> 00:00:01,000
{first_line}
{second_line}
"""

        asr_data = ASRData.from_srt(srt)

        assert asr_data.segments[0].text == source
        assert asr_data.segments[0].translated_text == translated

    def test_parse_two_different_canonical_urls_as_source_only(self):
        srt = """1
00:00:00,000 --> 00:00:01,000
example.com/first
example.com/second
"""

        asr_data = ASRData.from_srt(srt)

        assert asr_data.segments[0].text == "example.com/first\nexample.com/second"
        assert asr_data.segments[0].translated_text == ""

    def test_parse_source_above_bilingual(self):
        """测试英文在上、中文在下的常规双语布局"""
        srt = """1
00:00:00,000 --> 00:00:02,000
It's time to sell my dirt cheap W126.
是时候卖掉我这辆便宜的 W126 了。
"""
        asr_data = ASRData.from_srt(srt)
        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "It's time to sell my dirt cheap W126."
        assert asr_data.segments[0].translated_text == "是时候卖掉我这辆便宜的 W126 了。"

    @pytest.mark.parametrize(
        ("first_line", "second_line"),
        [
            ("교황이 일시적인 호흡 곤란을 겪었습니다", "教皇一度出现呼吸困难"),
            ("教皇一度出现呼吸困难", "교황이 일시적인 호흡 곤란을 겪었습니다"),
        ],
    )
    def test_parse_korean_chinese_bilingual_in_either_layout(
        self, first_line: str, second_line: str
    ):
        srt = f"""1
00:00:00,000 --> 00:00:03,000
{first_line}
{second_line}
"""

        asr_data = ASRData.from_srt(srt)

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "교황이 일시적인 호흡 곤란을 겪었습니다"
        assert asr_data.segments[0].translated_text == "教皇一度出现呼吸困难"

    def test_parse_multiline_korean_chinese_bilingual_groups(self):
        srt = """1
00:00:00,000 --> 00:00:05,000
教皇一度出现呼吸困难
目前已经恢复稳定
교황이 일시적인 호흡 곤란을 겪었지만
현재는 안정을 되찾았습니다
"""

        asr_data = ASRData.from_srt(srt)

        assert asr_data.segments[0].text == (
            "교황이 일시적인 호흡 곤란을 겪었지만\n현재는 안정을 되찾았습니다"
        )
        assert asr_data.segments[0].translated_text == "教皇一度出现呼吸困难\n目前已经恢复稳定"

    def test_parse_decomposed_hangul_source(self):
        korean = unicodedata.normalize("NFD", "교황이 호흡 곤란을 겪었습니다")
        srt = f"""1
00:00:00,000 --> 00:00:03,000
教皇出现呼吸困难
{korean}
"""

        asr_data = ASRData.from_srt(srt)

        assert asr_data.segments[0].text == korean
        assert asr_data.segments[0].translated_text == "教皇出现呼吸困难"

    def test_parse_multiline_bilingual_groups(self):
        """测试多行原文和多行译文可以按语言组拆分"""
        srt = """1
00:00:00,000 --> 00:00:04,000
这是一辆非常特别的车
也是我一直想聊的车
This is a very special car
and one I have wanted to talk about.
"""
        asr_data = ASRData.from_srt(srt)
        assert len(asr_data.segments) == 1
        assert (
            asr_data.segments[0].text
            == "This is a very special car\nand one I have wanted to talk about."
        )
        assert asr_data.segments[0].translated_text == "这是一辆非常特别的车\n也是我一直想聊的车"

    def test_parse_multiline_single_language_stays_single(self):
        """测试纯英文多行字幕不会被误拆成双语"""
        srt = """1
00:00:00,000 --> 00:00:03,000
This is a very special car
and one I have wanted to talk about.
"""
        asr_data = ASRData.from_srt(srt)
        assert len(asr_data.segments) == 1
        assert (
            asr_data.segments[0].text
            == "This is a very special car\nand one I have wanted to talk about."
        )
        assert asr_data.segments[0].translated_text == ""

    def test_parse_json_non_numeric_keys(self):
        """测试JSON非数字键"""
        json_data = {
            "a": {
                "original_subtitle": "Test",
                "translated_subtitle": "",
                "start_time": 0,
                "end_time": 1000,
            }
        }
        with pytest.raises(ValueError):
            ASRData.from_json(json_data)

    def test_parse_vtt_empty_blocks(self):
        """测试VTT空块"""
        vtt = """WEBVTT

HEADER


1
00:00:01.000 --> 00:00:02.000
Text1


"""
        asr_data = ASRData.from_vtt(vtt)
        assert len(asr_data.segments) == 1


class TestHandleLongPath:
    """Windows 长路径前缀处理"""

    def test_non_windows_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr("subforge.core.asr.asr_data.platform.system", lambda: "Linux")
        long_path = "C:\\" + "a" * 300
        assert handle_long_path(long_path) == long_path

    def test_windows_short_path_unchanged(self, monkeypatch):
        monkeypatch.setattr("subforge.core.asr.asr_data.platform.system", lambda: "Windows")
        short_path = "C:\\Users\\me\\file.srt"
        assert handle_long_path(short_path) == short_path

    def test_windows_long_path_gets_prefix(self, monkeypatch):
        monkeypatch.setattr("subforge.core.asr.asr_data.platform.system", lambda: "Windows")
        monkeypatch.setattr("subforge.core.asr.asr_data.os.path.abspath", lambda p: p)
        long_path = "C:\\Users\\me\\" + "a" * 300 + ".srt"
        result = handle_long_path(long_path)
        assert result.startswith("\\\\?\\")
        assert result == "\\\\?\\" + long_path

    def test_windows_already_prefixed_path_is_idempotent(self, monkeypatch):
        """Regression: handle_long_path was double-prefixing already-prefixed paths.

        The startswith check used r"\\\\?\\\\" (5 chars) but the prefix added is
        "\\\\?\\" (4 chars), so a second call would re-prefix the path and produce
        the malformed "\\\\?\\\\\\?\\C:\\..." seen in issue #1089.
        """
        monkeypatch.setattr("subforge.core.asr.asr_data.platform.system", lambda: "Windows")
        monkeypatch.setattr("subforge.core.asr.asr_data.os.path.abspath", lambda p: p)
        long_path = "C:\\Users\\me\\" + "a" * 300 + ".srt"
        once = handle_long_path(long_path)
        twice = handle_long_path(once)
        assert twice == once
        assert "\\\\?\\\\" not in twice

    def test_windows_long_unc_path_uses_unc_prefix(self, monkeypatch):
        monkeypatch.setattr("subforge.core.asr.asr_data.platform.system", lambda: "Windows")
        long_path = "\\\\server\\share\\" + "a" * 300 + ".srt"
        monkeypatch.setattr("subforge.core.asr.asr_data.os.path.abspath", lambda p: p)

        assert handle_long_path(long_path) == "\\\\?\\UNC\\server\\share\\" + "a" * 300 + ".srt"


class TestConservativeTimingRepair:
    def test_keeps_reasonable_long_word_duration(self):
        data = ASRData(
            [
                ASRDataSeg("extraordinary", 1_000, 2_650),
                ASRDataSeg("car", 2_800, 3_100),
            ]
        )

        data.cap_abnormal_word_durations()

        assert data.segments[0].end_time == 2_650

    def test_refines_only_vad_confirmed_word_edges_near_pause(self):
        data = ASRData(
            [
                ASRDataSeg("This", 1_100, 1_300),
                ASRDataSeg("ends", 1_350, 1_700),
                ASRDataSeg("Next", 2_300, 2_600),
                ASRDataSeg("line", 2_650, 2_900),
            ]
        )

        data.refine_word_edges_with_speech_segments([(1_000, 2_000), (2_200, 3_000)])

        assert data.segments[0].start_time == 970
        assert data.segments[1].end_time == 2_030
        assert data.segments[2].start_time == 2_170
        assert data.segments[3].end_time == 3_030

    def test_high_confidence_pass_repairs_large_utterance_tail_error(self):
        data = ASRData(
            [
                ASRDataSeg("annual", 1_500, 1_800),
                ASRDataSeg("revenue.", 1_850, 2_050),
                ASRDataSeg("Next", 4_700, 4_900),
            ],
            granularity="word",
        )

        data.refine_word_edges_with_speech_segments(
            [(1_400, 4_250), (4_650, 5_000)],
            max_adjustment_ms=3_200,
        )

        assert data.segments[1].end_time == 4_280
        assert data.segments[2].start_time == 4_700

    def test_does_not_move_internal_word_boundary_during_continuous_speech(self):
        data = ASRData(
            [
                ASRDataSeg("one", 1_000, 1_250),
                ASRDataSeg("two", 1_280, 1_500),
                ASRDataSeg("three", 1_520, 1_800),
                ASRDataSeg("four", 1_820, 2_100),
            ]
        )

        data.refine_word_edges_with_speech_segments([(950, 2_150)])

        assert data.segments[1].end_time == 1_500
        assert data.segments[2].start_time == 1_520

    def test_caps_abnormal_word_duration_without_moving_following_words(self):
        data = ASRData(
            [
                ASRDataSeg("over", 618_520, 618_700),
                ASRDataSeg("200,000", 618_700, 628_690),
                ASRDataSeg("I'll", 628_690, 628_830),
            ]
        )

        data.cap_abnormal_word_durations()

        assert data.segments[0].start_time == 618_520
        assert data.segments[0].end_time == 618_700
        assert data.segments[1].start_time == 618_700
        assert data.segments[1].end_time == 620_100
        assert data.segments[2].start_time == 628_690

    @staticmethod
    def _aligned_sentence(
        text: str,
        start: int,
        end: int,
        *,
        translated_text: str = "",
        word_end: int | None = None,
    ) -> ASRDataSeg:
        atomic_end = end if word_end is None else word_end
        return ASRDataSeg(
            text,
            start,
            end,
            translated_text=translated_text,
            words=[
                ASRWord(
                    text=text,
                    start_time=start,
                    end_time=atomic_end,
                    timing_source="forced_alignment",
                )
            ],
            timestamp_granularity="sentence",
            timing_source="forced_alignment",
        )

    def test_extends_aligned_sentence_with_small_safe_tail(self):
        data = ASRData(
            [
                self._aligned_sentence(
                    "and 229 pound feet of torque.",
                    289_845,
                    290_569,
                    translated_text="能输出229磅英尺的扭矩",
                ),
                ASRDataSeg("Attached to this,", 293_120, 293_903, translated_text="与之匹配的"),
            ]
        )

        data.extend_sentence_tails_conservatively()

        assert data.segments[0].end_time == 290_729
        assert data.segments[1].start_time == 293_120

    def test_extension_is_independent_of_sentence_text_or_length(self):
        data = ASRData(
            [
                self._aligned_sentence(
                    "Odometer showing 186,764 miles but that is also",
                    607_689,
                    611_289,
                    translated_text="里程表显示186,764英里 不过这也是",
                ),
                ASRDataSeg("not accurate.", 612_540, 613_042, translated_text="不准确"),
            ]
        )

        data.extend_sentence_tails_conservatively()

        assert data.segments[0].end_time == 611_449
        assert data.segments[1].start_time == 612_540

    def test_uses_high_confidence_vad_for_a_larger_bounded_tail(self):
        data = ASRData(
            [
                self._aligned_sentence(
                    "The way these get off the line is unreal.",
                    970_000,
                    972_100,
                    translated_text="这车起步的方式太离谱了",
                ),
                ASRDataSeg("Next sentence.", 973_800, 974_500, translated_text="下一句"),
            ]
        )

        data.extend_sentence_tails_conservatively([(969_900, 972_620)])

        assert data.segments[0].end_time == 972_650
        assert data.segments[1].start_time == 973_800

    def test_uses_high_confidence_vad_for_multi_second_alignment_fallback(self):
        data = ASRData(
            [
                self._aligned_sentence(
                    "more than $1.2BN in annual revenue.",
                    181_982,
                    183_270,
                ),
                ASRDataSeg("The venue reportedly", 185_942, 186_985),
            ]
        )

        data.extend_sentence_tails_conservatively([(180_224, 185_296)])

        assert data.segments[0].end_time == 185_326
        assert data.segments[0].end_time < data.segments[1].start_time

    def test_does_not_extend_without_atomic_word_provenance(self):
        data = ASRData(
            [
                ASRDataSeg(
                    "This sentence is already readable enough.",
                    0,
                    3_800,
                    translated_text="这句已经足够长",
                ),
                ASRDataSeg("Next.", 4_200, 4_800, translated_text="下一句"),
                ASRDataSeg("Short.", 5_000, 5_700, translated_text="短句"),
                ASRDataSeg("Close next.", 6_000, 6_800, translated_text="下一句很近"),
            ]
        )

        data.extend_sentence_tails_conservatively()

        assert data.segments[0].end_time == 3_800
        assert data.segments[2].end_time == 5_700

    def test_does_not_extend_when_existing_tail_is_already_sufficient(self):
        data = ASRData(
            [
                self._aligned_sentence(
                    "If I had to guess I would say probably over 200,000.",
                    616_313,
                    620_100,
                    translated_text="如果非要猜，可能超过200,000",
                    word_end=619_900,
                ),
                ASRDataSeg(
                    "I'll tell you what though.", 628_694, 631_197, translated_text="不过我跟你说"
                ),
            ]
        )

        data.extend_sentence_tails_conservatively()

        assert data.segments[0].end_time == 620_100

    def test_bounds_vad_extension_that_crosses_the_next_cue(self):
        data = ASRData(
            [
                self._aligned_sentence("Current sentence.", 1_000, 2_000),
                ASRDataSeg("Next sentence.", 2_500, 3_000),
            ]
        )

        data.extend_sentence_tails_conservatively([(900, 2_700)])

        assert data.segments[0].end_time == 2_420
        assert data.segments[0].end_time < data.segments[1].start_time

    def test_caps_crossing_vad_extension_for_a_distant_next_cue(self):
        data = ASRData(
            [
                self._aligned_sentence("Current sentence.", 1_000, 2_000),
                ASRDataSeg("Next sentence.", 5_000, 5_500),
            ]
        )

        data.extend_sentence_tails_conservatively([(900, 5_200)])

        assert data.segments[0].end_time == 3_200
        assert data.segments[0].end_time < data.segments[1].start_time

    def test_extends_final_sentence_with_media_duration_bound(self):
        data = ASRData([self._aligned_sentence("Final sentence.", 1_000, 2_000)])

        data.extend_sentence_tails_conservatively(
            [(900, 2_420)],
            media_duration_ms=2_500,
        )

        assert data.segments[0].end_time == 2_450

    def test_final_sentence_never_exceeds_media_duration(self):
        data = ASRData([self._aligned_sentence("Final sentence.", 1_000, 2_000)])

        data.extend_sentence_tails_conservatively(
            [(900, 2_700)],
            media_duration_ms=2_300,
        )

        assert data.segments[0].end_time == 2_160
        assert data.segments[0].end_time <= 2_300

    def test_final_sentence_can_use_vad_without_media_duration(self):
        data = ASRData([self._aligned_sentence("Final sentence.", 1_000, 2_000)])

        data.extend_sentence_tails_conservatively([(900, 2_120)])

        assert data.segments[0].end_time == 2_150
